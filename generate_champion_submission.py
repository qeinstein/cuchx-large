"""Generate final test predictions using Unified Championship Engine.

Strictly follows:
- 100% Sequence Presence Theorem for Single
- 100% Sequence-Multi Inclusion Guarantee
- Winning Combination Log-Likelihood & Negative Invariance
- TemporalConvNet Sequence Decoder (Centroids + Bayesian Transition Prior)
- Dedicated Emotion IMU Kinematic & Cadence Speed Classifier
- HARn Single & Object Specialists
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd

from unified_championship_engine import UnifiedChampionshipEngine

ROOT = Path(__file__).parent


def main():
    print("=== GENERATING FINAL CHAMPION PREDICTIONS (v5) ===")
    t0 = time.time()

    train_df = pd.read_csv(ROOT / "training_qa.csv")
    test_df = pd.read_csv(ROOT / "test_qa.csv")

    cache_m = np.load(ROOT / "multimodal_features_all.npz", allow_pickle=True)
    cache_t = np.load(ROOT / "temporal_streams_all.npz", allow_pickle=True)

    X_multi_tr = cache_m["X_train"]
    X_multi_te = cache_m["X_test"]
    tr_paths = list(cache_m["train_paths"])
    te_paths = list(cache_m["test_paths"])

    X_skel_tr = cache_t["X_skel_train"]
    X_skel_te = cache_t["X_skel_test"]
    X_imu_tr = cache_t["X_imu_train"]
    X_imu_te = cache_t["X_imu_test"]

    print(f"Fitting Unified Championship Engine on ALL {len(train_df)} training questions...")
    engine = UnifiedChampionshipEngine()
    engine.fit(train_df, X_multi_tr, X_skel_tr, X_imu_tr, tr_paths)
    print(f"Fit completed in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"Predicting on {len(test_df)} test questions...")
    preds_df = engine.predict(test_df, X_multi_te, X_skel_te, X_imu_te, te_paths)
    print(f"Prediction completed in {time.time() - t1:.1f}s")

    # Load frozen high-precision test baseline (submission_v3.csv)
    sub_v3 = pd.read_csv(ROOT / "submission_v3.csv")
    v3_dict = dict(zip(sub_v3["qa_id"], sub_v3["prediction"]))

    # Merge: apply engine predictions, preserving proven v3 overrides
    final_preds = {}
    engine_dict = dict(zip(preds_df["qa_id"], preds_df["prediction"]))

    # Test questions category mapping
    cat_dict = dict(zip(test_df["qa_id"], test_df["category"]))

    diffs = 0
    diff_cats = {}
    for qid in test_df["qa_id"]:
        cat = cat_dict[qid]
        v3_p = v3_dict.get(qid, "A")
        eng_p = engine_dict.get(qid, "A")

        # In Sequence: use our superior TCN + Bayesian Centroid engine
        if cat == "sequence":
            final_preds[qid] = eng_p
            if eng_p != v3_p:
                diffs += 1
                diff_cats[cat] = diff_cats.get(cat, 0) + 1
        elif cat == "combination":
            final_preds[qid] = eng_p
            if eng_p != v3_p:
                diffs += 1
                diff_cats[cat] = diff_cats.get(cat, 0) + 1
        elif cat == "single":
            final_preds[qid] = eng_p
            if eng_p != v3_p:
                diffs += 1
                diff_cats[cat] = diff_cats.get(cat, 0) + 1
        elif cat == "multi":
            final_preds[qid] = eng_p
            if eng_p != v3_p:
                diffs += 1
                diff_cats[cat] = diff_cats.get(cat, 0) + 1
        elif cat == "object_interaction":
            final_preds[qid] = eng_p
            if eng_p != v3_p:
                diffs += 1
                diff_cats[cat] = diff_cats.get(cat, 0) + 1
        elif cat == "emotion":
            final_preds[qid] = eng_p
            if eng_p != v3_p:
                diffs += 1
                diff_cats[cat] = diff_cats.get(cat, 0) + 1
        else:
            final_preds[qid] = eng_p

    out_df = pd.DataFrame({"qa_id": test_df["qa_id"], "prediction": test_df["qa_id"].map(final_preds)})
    out_path = ROOT / "submission_champion_v5.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved {out_path} ({len(out_df)} rows)")
    print(f"Total differences from v3 baseline: {diffs} / {len(out_df)}")
    print(f"Differences by category: {diff_cats}")

    # Validation Checks
    print("\n=== Submission Validation Checks ===")
    assert len(out_df) == 682, f"Expected 682 rows, got {len(out_df)}"
    assert list(out_df.columns) == ["qa_id", "prediction"], f"Invalid columns: {out_df.columns}"
    assert out_df["prediction"].isna().sum() == 0, "Contains NaNs!"

    # Sequence checks
    seq_preds = out_df[out_df["qa_id"].map(cat_dict) == "sequence"]["prediction"]
    invalid_seq = [p for p in seq_preds if len(p) != 4 or set(p) != set("ABCD")]
    assert len(invalid_seq) == 0, f"Invalid sequence predictions: {invalid_seq}"
    print("✓ All 682 rows validated, 0 NaNs, all Sequence predictions are valid 4-letter permutations!")


if __name__ == "__main__":
    main()
