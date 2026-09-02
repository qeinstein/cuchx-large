"""Build Championship Grandmaster Submission v1.

Fuses:
1. 1,720-dimensional Multi-Spectral Feature Engine (IMU, Skeleton, DINOv2, Thermal, Radar)
2. Closed-World Belief Propagation Solver (Combination, Single, HARn)
3. Qwen-2.5-VL-72B Vision-Language Keyframe Oracle (Sequence, Emotion, Multi)
4. IMU Jerk & Doppler Velocity Cadence Monotonicity Constraints
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TEST_QA = ROOT / "test_qa.csv"
VLM_CACHE = ROOT / "vlm_predictions_cache.json"
BASE_V3 = ROOT / "submission_v3.csv"
OUT_CSV = ROOT / "submission_championship_v1.csv"


def main():
    print("=== Assembling Championship Grandmaster Submission v1 ===")
    test_df = pd.read_csv(TEST_QA)
    v3_df = pd.read_csv(BASE_V3)
    v3_map = dict(zip(v3_df.qa_id, v3_df.prediction))

    with open(VLM_CACHE) as f:
        vlm_map = json.load(f)

    # 1. Closed-world candidate action pool per clip
    clip_pools = {}
    for p, grp in test_df.groupby("path"):
        pool = set()
        for _, r in grp[grp.category.isin(["single", "multi", "combination", "sequence"])].iterrows():
            for l in "ABCD":
                opt = str(r[l]).strip().lower()
                if opt and opt != "nan" and not any(k in opt for k in ["person", "none", "cannot"]):
                    pool.add(opt)
        clip_pools[p] = pool

    # 2. Extract IMU jerk & Radar cadence speed per clip
    cache_m = np.load(ROOT / "multimodal_features_all.npz", allow_pickle=True)
    paths_m = list(cache_m["test_paths"])
    X_m = cache_m["X_test"]
    p_to_idx = {p: i for i, p in enumerate(paths_m)}

    clip_cadence = {}
    for p in test_df.path.unique():
        if p in p_to_idx:
            idx = p_to_idx[p]
            # IMU Right Arm Jerk (feature 31)
            jerk = X_m[idx, 31]
            # Radar mean velocity (feature 1688)
            v_rad = X_m[idx, 1688]
            if jerk < 65.0:
                z = 1  # Slow
            elif jerk > 95.0 or v_rad > 0.058:
                z = 3  # Fast
            else:
                z = 2  # Normal
            clip_cadence[p] = z
        else:
            clip_cadence[p] = 2

    # Map words to cadence tier
    slow_words = {"slowly", "gently", "leisurely", "unhurriedly", "casually", "relaxedly"}
    fast_words = {"hastily", "anxiously", "hurriedly", "restlessly", "quickly", "nervously", "frantically"}
    normal_words = {"steadily", "calmly", "neatly", "intently", "seriously", "attentively", "naturally", "gracefully"}

    predictions = []
    stats = {"v3_kept": 0, "vlm_sequence": 0, "vlm_emotion": 0, "vlm_multi": 0, "cadence_overrides": 0}

    for _, row in test_df.iterrows():
        qid = row.qa_id
        cat = row.category
        v3_ans = v3_map.get(qid, "A")
        p = row.path

        if cat == "sequence":
            # Sequence: Prioritize Qwen-2.5-VL-72B visual chronological ordering
            vlm_ans = vlm_map.get(qid, "")
            if len(vlm_ans) == 4 and len(set(vlm_ans)) == 4:
                final_ans = vlm_ans
                stats["vlm_sequence"] += 1
            else:
                final_ans = v3_ans
                stats["v3_kept"] += 1

        elif cat == "emotion":
            # Emotion: Fused Qwen-2.5-VL + Physical Cadence Monotonicity
            vlm_ans = vlm_map.get(qid, "")
            z = clip_cadence.get(p, 2)

            # Check if VLM answer physically contradicts measured IMU jerk
            if vlm_ans in "ABCD":
                vlm_word = str(row[vlm_ans]).strip().lower()
                # If clip is slow (Z=1) but VLM picked fast, or clip is fast (Z=3) but VLM picked slow
                if z == 1 and any(fw in vlm_word for fw in fast_words):
                    # Override to slow option
                    cand = [l for l in "ABCD" if any(sw in str(row[l]).lower() for sw in slow_words)]
                    final_ans = cand[0] if cand else vlm_ans
                    stats["cadence_overrides"] += 1
                elif z == 3 and any(sw in vlm_word for sw in slow_words):
                    # Override to fast option
                    cand = [l for l in "ABCD" if any(fw in str(row[l]).lower() for fw in fast_words)]
                    final_ans = cand[0] if cand else vlm_ans
                    stats["cadence_overrides"] += 1
                else:
                    final_ans = vlm_ans
                    stats["vlm_emotion"] += 1
            else:
                final_ans = v3_ans
                stats["v3_kept"] += 1

        elif cat == "multi":
            # Multi: Qwen-2.5-VL visual detection validated against clip candidate pool
            vlm_ans = vlm_map.get(qid, "")
            pool = clip_pools.get(p, set())

            if vlm_ans:
                # Keep letters whose action is in the pool
                valid_letters = [
                    l for l in vlm_ans if str(row[l]).strip().lower() in pool
                ]
                if valid_letters:
                    final_ans = "".join(sorted(valid_letters))
                    stats["vlm_multi"] += 1
                else:
                    final_ans = v3_ans
                    stats["v3_kept"] += 1
            else:
                final_ans = v3_ans
                stats["v3_kept"] += 1

        else:
            # Combination, Single, Object Interaction: Kept from high-confidence 1,720d solver
            final_ans = v3_ans
            stats["v3_kept"] += 1

        predictions.append({"qa_id": qid, "prediction": final_ans})

    out_df = pd.DataFrame(predictions)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nGenerated {OUT_CSV} with {len(out_df)} rows.")
    print("Breakdown of predictions:")
    for k, v in stats.items():
        print(f"  {k:22s}: {v}")

    # Integrity verification
    assert len(out_df) == 682, f"Expected 682 rows, got {len(out_df)}"
    assert set(out_df.columns) == {"qa_id", "prediction"}
    assert out_df["prediction"].isna().sum() == 0, "Null predictions found!"
    print("ALL INTEGRITY CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    main()
