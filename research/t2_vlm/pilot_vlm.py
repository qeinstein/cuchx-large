"""T2 VLM pilot: frontier-VLM zero-shot accuracy on a fixed 30Q diagnostic subset.

Model: anthropic/claude-sonnet-5 via OpenRouter (key loaded at runtime from the
repo's existing benchmark_vlm_judge.py; no secret is copied here).
Visual: 16-frame 4x4 contact sheet from Depth_Color (fallback Depth).
Output: research/t2_vlm/pilot_results.csv + stdout summary.
"""
import base64
import csv
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "hf_data_manual"
OUT_CSV = Path(__file__).parent / "pilot_results.csv"

MODEL = "anthropic/claude-sonnet-5"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
N_PER_CAT = 6
CATS = ["single", "multi", "combination", "sequence", "emotion"]
SEED = 20260914
WORKERS = 3


def load_key():
    src = (ROOT / "benchmark_vlm_judge.py").read_text()
    return re.search(r'API_KEY\s*=\s*["\'](.+?)["\']', src).group(1)


def resolve_video(rel_path):
    base = DATA / rel_path
    for cand in [base / "Depth_Color" / "Depth_Color.mp4", base / "Depth" / "Depth.mp4"]:
        if cand.exists():
            return cand
    mp4s = sorted(base.glob("**/*.mp4"))
    return mp4s[0] if mp4s else None


def contact_sheet_b64(vpath, n=16, thumb=300):
    cap = cv2.VideoCapture(str(vpath))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((240, 320, 3), np.uint8)
        fr = cv2.resize(fr, (thumb, thumb * fr.shape[0] // max(1, fr.shape[1])))
        frames.append(fr)
    cap.release()
    h = min(f.shape[0] for f in frames)
    frames = [f[:h] for f in frames]
    rows = [np.hstack(frames[r * 4:(r + 1) * 4]) for r in range(4)]
    sheet = np.vstack(rows)
    rgb = cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def prompt_for(row):
    cat = row["category"]
    opts = "\n".join(f"{L}: {row[L]}" for L in "ABCD" if row[L].strip())
    head = (
        "You are an expert activity-recognition judge. The attached image is a 16-frame "
        "contact sheet (chronological, top-left to bottom-right) from a DEPTH video "
        "(false-color; brighter = closer). No RGB is available. Judge motion and posture only.\n\n"
        f"Question: {row['question']}\nOptions:\n{opts}\n"
    )
    if cat == "multi":
        tail = (
            "\nMultiple options may apply. For EACH option letter, first write one word "
            "(PRESENT or ABSENT) with a 3-6 word reason. Then on the LAST line write "
            "exactly: FINAL: <concatenated present letters in A-D order, e.g. FINAL: ABD>"
        )
    elif cat == "sequence":
        tail = (
            "\nList the actions in chronological order, then on the LAST line write exactly: "
            "FINAL: <4 letters in chronological order, e.g. FINAL: DBCA>"
        )
    else:
        tail = (
            "\nReply with brief reasoning (<=30 words), then on the LAST line write exactly: "
            "FINAL: <single letter>"
        )
    return head + tail


def parse_final(text):
    m = re.findall(r"FINAL:\s*([A-D]{1,4})", text.upper())
    return m[-1] if m else ""


def call_model(key, prompt, b64):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 250,
    }
    t0 = time.time()
    r = requests.post(API_URL, headers=headers, json=body, timeout=120)
    dt = time.time() - t0
    r.raise_for_status()
    d = r.json()
    txt = d["choices"][0]["message"]["content"] or ""
    use = d.get("usage", {})
    return txt, dt, use.get("prompt_tokens", 0), use.get("completion_tokens", 0)


def run_one(key, row):
    vpath = resolve_video(row["path"])
    if not vpath:
        return {**row, "pred": "", "correct": 0, "err": "no-video", "raw": ""}
    b64 = contact_sheet_b64(vpath)
    if not b64:
        return {**row, "pred": "", "correct": 0, "err": "decode-fail", "raw": ""}
    last = ""
    for attempt in range(3):
        try:
            txt, dt, pt, ct = call_model(key, prompt_for(row), b64)
            pred = parse_final(txt)
            ok = 1 if pred == row["answer"].strip().upper() else 0
            return {**row, "pred": pred, "correct": ok, "err": "",
                    "raw": txt.replace("\n", " | ")[:600],
                    "dt": round(dt, 1), "pt": pt, "ct": ct}
        except Exception as e:
            last = str(e)[:120]
            time.sleep(2 * (attempt + 1))
    return {**row, "pred": "", "correct": 0, "err": last, "raw": ""}


def main():
    key = load_key()
    rows = list(csv.DictReader(open(ROOT / "training_qa.csv", encoding="utf-8-sig")))
    rng = random.Random(SEED)
    subset = []
    for c in CATS:
        pool = [r for r in rows if r["category"] == c]
        subset += rng.sample(pool, N_PER_CAT)
    print(f"Pilot: model={MODEL} n={len(subset)} seed={SEED}", flush=True)
    results = []
    tot_pt = tot_ct = 0.0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, key, r): r for r in subset}
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            tot_pt += res.get("pt", 0)
            tot_ct += res.get("ct", 0)
            r0 = futs[f]
            print(f"{r0['qa_id']} {r0['category']:12s} ans={r0['answer']:4s} "
                  f"pred={res['pred']:4s} ok={res['correct']} err={res['err']} "
                  f"({res.get('dt', '?')}s)", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qa_id", "category", "answer", "pred",
                                          "correct", "err", "dt", "pt", "ct", "raw"])
        w.writeheader()
        for r in sorted(results, key=lambda x: x["qa_id"]):
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print("\n--- per-category ---")
    for c in CATS:
        rs = [r for r in results if r["category"] == c]
        ok = sum(r["correct"] for r in rs)
        print(f"{c:12s} {ok}/{len(rs)} = {ok / len(rs):.3f}")
    ok = sum(r["correct"] for r in results)
    print(f"OVERALL {ok}/{len(results)} = {ok / len(results):.3f}")
    # claude-sonnet-5 pricing $2/$10 per MTok
    print(f"tokens in={tot_pt:.0f} out={tot_ct:.0f} "
          f"est_cost=${(tot_pt * 2 + tot_ct * 10) / 1e6:.4f}")


if __name__ == "__main__":
    main()
