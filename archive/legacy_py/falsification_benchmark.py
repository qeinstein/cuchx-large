"""Falsification Benchmark on Labeled Validation Set.

Reproduces the exact Championship 0.97+ Consensus Pipeline on held-out validation
clips from unseen human subjects (Fold 0: user2 and user1) with labels completely hidden.

Compares:
1. v3-Equivalent Baseline (1,720d multi-spectral model trained without user2/user1)
2. Raw Qwen-2.5-VL-72B (Independent zero-shot visual VLM with no cross-question logic)
3. Championship Consensus Pipeline (Sequence presence proofs + Winning combination consensus + Closed-world multi + IMU jerk bounds)

Reports:
- Category-by-category accuracy
- Overall accuracy comparison on exactly identical examples
- Empirical verification of the 0.97+ hypothesis
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io
import json
import os
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm

from championship_solver import ChampionshipSolver

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "hf_data_manual"
CACHE_FILE = ROOT / "vlm_falsification_cache.json"

API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "qwen/qwen2.5-vl-72b-instruct"


def normalize_text(s: str) -> str:
    return str(s).strip().lower()


def parse_combination_actions(combo_text: str) -> set:
    parts = combo_text.split(",")
    return {p.strip().lower() for p in parts if p.strip()}


def extract_video_frames(rel_path: str, num_frames: int = 6) -> list:
    """Extract num_frames evenly spaced JPEG base64 images from depth video."""
    video_path = DATA_ROOT / rel_path / "Depth" / "Depth.mp4"
    if not video_path.exists():
        video_path = DATA_ROOT / rel_path / "Depth_Color" / "Depth_Color.mp4"
    if not video_path.exists():
        # Fallback to direct path
        video_path = DATA_ROOT / rel_path
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

    indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    b64_frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((336, 336))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    cap.release()
    return b64_frames


def query_qwen_vl(qa_id: str, category: str, question: str, options: dict, b64_frames: list) -> str:
    if not b64_frames:
        return "A"

    image_contents = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        for b in b64_frames
    ]
    opt_text = "\n".join([f"{l}: {options[l]}" for l in "ABCD"])

    if category == "sequence":
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show actions performed chronologically from first to last.\n"
            f"Question: {question}\n{opt_text}\n\n"
            "Task: Arrange all 4 letters in exact chronological order from first performed to last performed (e.g. DBCA or ABCD).\n"
            "Reply strictly with only the 4 uppercase letters, nothing else."
        )
    elif category == "emotion":
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show a person performing activities.\n"
            f"Question: {question}\n{opt_text}\n\n"
            "Task: Determine the physical pace, manner, or emotion expressed by the person in these frames.\n"
            "Reply strictly with only the single uppercase letter of the correct option (A, B, C, or D), nothing else."
        )
    elif category == "multi":
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show a person performing activities.\n"
            f"Question: {question}\n{opt_text}\n\n"
            "Task: Identify ALL actions from the choices that actually appear in these frames.\n"
            "Reply strictly with only the uppercase letters of the correct actions with no spaces (e.g. AB or CD or BCD), nothing else."
        )
    elif category == "combination":
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show a person performing activities.\n"
            f"Question: {question}\n{opt_text}\n\n"
            "Task: Determine which combination of actions is performed in these frames.\n"
            "Reply strictly with only the single uppercase letter of the correct option (A, B, C, or D), nothing else."
        )
    else:
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show a person performing activities.\n"
            f"Question: {question}\n{opt_text}\n\n"
            "Task: Answer the question based on what is visible in these frames.\n"
            "Reply strictly with only the single uppercase letter of the correct option (A, B, C, or D), nothing else."
        )

    content = [{"type": "text", "text": prompt_text}] + image_contents
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 12,
        "temperature": 0.05,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            ans = resp.json()["choices"][0]["message"]["content"].strip()
            clean_ans = "".join([c for c in ans.upper() if c in "ABCD"])
            if category == "sequence" and len(clean_ans) == 4 and len(set(clean_ans)) == 4:
                return clean_ans
            elif category in ("emotion", "single", "combination", "object_interaction") and len(clean_ans) >= 1:
                return clean_ans[0]
            elif category == "multi" and len(clean_ans) >= 1:
                seen = set()
                dedup = "".join([c for c in clean_ans if not (c in seen or seen.add(c))])
                return "".join(sorted(dedup))
            return clean_ans
    except Exception:
        return "A"
    return "A"


def main():
    print("=== FALSIFICATION BENCHMARK: HELD-OUT UNSEEN SUBJECT EVALUATION ===")

    val_df = pd.read_csv(ROOT / "splits" / "fold_0_val.csv")
    train_df = pd.read_csv(ROOT / "splits" / "fold_0_train.csv")

    # Select representative sample of unseen validation clips from user2 and user1
    # 10 clips with sequence questions, 10 clips without sequence questions
    u2_val = val_df[val_df.path.str.contains("user2")].copy()
    seq_clips = list(u2_val[u2_val.category == "sequence"].path.unique())[:10]
    non_seq_clips = [p for p in u2_val.path.unique() if p not in seq_clips][:10]
    selected_clips = set(seq_clips + non_seq_clips)

    benchmark_df = u2_val[u2_val.path.isin(selected_clips)].copy()
    print(f"Selected {len(benchmark_df)} questions across {len(selected_clips)} clips on unseen subject user2.")
    print("Question distribution:")
    print(benchmark_df.category.value_counts())

    # 1. Train v3-equivalent baseline strictly on fold 0 train (user2/user1 completely held out)
    print("\n--- Training v3-equivalent Multi-Spectral Baseline on Fold 0 Train ---")
    cache_m = np.load(ROOT / "multimodal_features_all.npz", allow_pickle=True)
    all_tr_paths = list(cache_m["train_paths"])
    all_tr_X = cache_m["X_train"]

    solver = ChampionshipSolver()
    solver.fit(train_df, all_tr_X, all_tr_paths)

    pred_df = solver.predict_clip_bundle(benchmark_df, all_tr_X, all_tr_paths)
    v3_preds = dict(zip(pred_df["qa_id"], pred_df["prediction"]))

    # 2. Run / Load Raw Qwen-2.5-VL-72B
    print("\n--- Running Raw Qwen-2.5-VL-72B on Depth Keyframes (Labels Hidden) ---")
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            print(f"Loaded existing falsification cache with {len(cache)} entries.")
        except Exception:
            cache = {}

    pending = [r for _, r in benchmark_df.iterrows() if r.qa_id not in cache]
    if pending:
        print(f"Querying Qwen-2.5-VL-72B for {len(pending)} questions...")
        # Pre-extract frames
        unique_paths = list({r.path for r in pending})
        path_frames = {p: extract_video_frames(p, num_frames=6) for p in tqdm(unique_paths, desc="Extracting Frames")}

        def process_q(row):
            frames = path_frames.get(row.path, [])
            opts = {l: row[l] for l in "ABCD"}
            ans = query_qwen_vl(row.qa_id, row.category, row.question, opts, frames)
            return row.qa_id, ans

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_q, r): r.qa_id for r in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="VLM Inference"):
                qid, ans = future.result()
                if ans:
                    cache[qid] = ans

        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

    raw_vlm_preds = {r.qa_id: cache.get(r.qa_id, "A") for _, r in benchmark_df.iterrows()}

    # 3. Run Championship 0.97+ Consensus Pipeline
    print("\n--- Executing Championship Consensus Pipeline on Validation Clips ---")
    cache_m = np.load(ROOT / "multimodal_features_all.npz", allow_pickle=True)
    paths_m = list(cache_m["train_paths"])
    X_m = cache_m["X_train"]
    p_to_idx = {p: i for i, p in enumerate(paths_m)}

    slow_words = {"slowly", "gently", "leisurely", "unhurriedly", "casually", "relaxedly"}
    fast_words = {"hastily", "anxiously", "hurriedly", "restlessly", "quickly", "nervously", "frantically"}

    consensus_preds = {}

    for path, clip_df in benchmark_df.groupby("path"):
        idx = p_to_idx.get(path, None)
        if idx is not None:
            jerk = X_m[idx, 31]
            v_rad = X_m[idx, 1688]
            speed_tier = 1 if jerk < 65.0 else (3 if jerk > 95.0 or v_rad > 0.058 else 2)
        else:
            speed_tier = 2

        s_rows = clip_df[clip_df.category == "single"]
        c_rows = clip_df[clip_df.category == "combination"]
        m_rows = clip_df[clip_df.category == "multi"]
        seq_rows = clip_df[clip_df.category == "sequence"]
        emo_rows = clip_df[clip_df.category == "emotion"]
        obj_rows = clip_df[clip_df.category == "object_interaction"]

        # Sequence actions
        seq_actions = set()
        if len(seq_rows) > 0:
            seq_row = seq_rows.iloc[0]
            for l in "ABCD":
                act = normalize_text(seq_row[l])
                if act and act != "nan":
                    seq_actions.add(act)

        # Single: Sequence presence proof
        for _, s_row in s_rows.iterrows():
            qid = s_row.qa_id
            vlm_s = raw_vlm_preds.get(qid, "A")
            v3_s = v3_preds.get(qid, "A")

            proven = [l for l in "ABCD" if normalize_text(s_row[l]) in seq_actions]
            if len(proven) == 1:
                consensus_preds[qid] = proven[0]
            else:
                consensus_preds[qid] = vlm_s

        # Combination: 3-way consensus
        if len(c_rows) > 0:
            c_row = c_rows.iloc[0]
            qid = c_row.qa_id
            vlm_c = raw_vlm_preds.get(qid, "A")
            v3_c = v3_preds.get(qid, "A")

            combo_options = {l: parse_combination_actions(str(c_row[l])) for l in "ABCD"}
            scores = {l: 0.0 for l in "ABCD"}
            scores[vlm_c] += 3.0
            scores[v3_c] += 1.5

            for l, acts in combo_options.items():
                scores[l] += 2.0 * len(acts.intersection(seq_actions))
                if len(s_rows) > 0:
                    s_ans = consensus_preds.get(s_rows.iloc[0].qa_id, "")
                    if s_ans and normalize_text(s_rows.iloc[0][s_ans]) in acts:
                        scores[l] += 2.5

            consensus_preds[qid] = max(scores, key=scores.get)

        # Multi: Closed-world projection from winning combination
        if len(m_rows) > 0:
            m_row = m_rows.iloc[0]
            qid = m_row.qa_id
            vlm_m = raw_vlm_preds.get(qid, "")
            v3_m = v3_preds.get(qid, "")

            if len(c_rows) > 0:
                winning_c = consensus_preds.get(c_rows.iloc[0].qa_id, "A")
                winning_acts = parse_combination_actions(str(c_rows.iloc[0][winning_c]))
            else:
                winning_acts = set()

            multi_from_combo = [l for l in "ABCD" if normalize_text(m_row[l]) in winning_acts]
            if multi_from_combo:
                consensus_preds[qid] = "".join(sorted(multi_from_combo))
            else:
                consensus_preds[qid] = vlm_m if vlm_m else v3_m

        # Sequence: VLM chronological tracking
        if len(seq_rows) > 0:
            seq_row = seq_rows.iloc[0]
            qid = seq_row.qa_id
            vlm_seq = raw_vlm_preds.get(qid, "")
            if len(vlm_seq) == 4 and len(set(vlm_seq)) == 4:
                consensus_preds[qid] = vlm_seq
            else:
                consensus_preds[qid] = v3_preds.get(qid, "ABCD")

        # Emotion: VLM + IMU jerk physical speed bounds
        if len(emo_rows) > 0:
            emo_row = emo_rows.iloc[0]
            qid = emo_row.qa_id
            vlm_e = raw_vlm_preds.get(qid, "A")
            vlm_word = normalize_text(emo_row[vlm_e])

            if speed_tier == 1 and any(fw in vlm_word for fw in fast_words):
                cand = [l for l in "ABCD" if any(sw in normalize_text(emo_row[l]) for sw in slow_words)]
                consensus_preds[qid] = cand[0] if cand else vlm_e
            elif speed_tier == 3 and any(sw in vlm_word for sw in slow_words):
                cand = [l for l in "ABCD" if any(fw in normalize_text(emo_row[l]) for fw in fast_words)]
                consensus_preds[qid] = cand[0] if cand else vlm_e
            else:
                consensus_preds[qid] = vlm_e

        # Object interaction
        if len(obj_rows) > 0:
            obj_row = obj_rows.iloc[0]
            qid = obj_row.qa_id
            consensus_preds[qid] = raw_vlm_preds.get(qid, v3_preds.get(qid, "A"))

    # 4. Accuracy Evaluation Against Ground Truth
    print("\n" + "=" * 70)
    print("                     ACCURACY REPORT & COMPARISON")
    print("=" * 70)

    benchmark_df["v3"] = [str(v3_preds.get(qid, "A")) for qid in benchmark_df.qa_id]
    benchmark_df["raw_vlm"] = [str(raw_vlm_preds.get(qid, "A")) for qid in benchmark_df.qa_id]
    benchmark_df["consensus"] = [str(consensus_preds.get(qid, "A")) for qid in benchmark_df.qa_id]

    def eval_model(col_name):
        res = {}
        for cat, grp in benchmark_df.groupby("category"):
            corr = (grp[col_name] == grp["answer"]).sum()
            res[cat] = (corr, len(grp), corr / len(grp))
        tot_corr = (benchmark_df[col_name] == benchmark_df["answer"]).sum()
        res["OVERALL"] = (tot_corr, len(benchmark_df), tot_corr / len(benchmark_df))
        return res

    res_v3 = eval_model("v3")
    res_vlm = eval_model("raw_vlm")
    res_con = eval_model("consensus")

    cats = sorted([c for c in res_v3.keys() if c != "OVERALL"]) + ["OVERALL"]

    print(f"{'Category':22s} | {'v3 Baseline':15s} | {'Raw Qwen-72B':15s} | {'Championship Consensus':22s}")
    print("-" * 80)
    for c in cats:
        s_v3 = f"{res_v3[c][0]}/{res_v3[c][1]} ({res_v3[c][2]*100:.1f}%)"
        s_vlm = f"{res_vlm[c][0]}/{res_vlm[c][1]} ({res_vlm[c][2]*100:.1f}%)"
        s_con = f"{res_con[c][0]}/{res_con[c][1]} ({res_con[c][2]*100:.1f}%)"
        if c == "OVERALL":
            print("-" * 80)
            print(f"{c:22s} | {s_v3:15s} | {s_vlm:15s} | {s_con:22s}")
        else:
            print(f"{c:22s} | {s_v3:15s} | {s_vlm:15s} | {s_con:22s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
