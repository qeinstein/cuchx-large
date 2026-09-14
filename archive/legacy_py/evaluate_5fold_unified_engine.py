"""Evaluate Unified Championship Engine across all 5 subject-held-out validation folds.

Outputs exact 5-fold cross-validation metrics across all 4,087 samples:
- overall accuracy
- per-category accuracy (single, combination, multi, sequence, emotion, object_interaction)
- delta vs v3 baseline
- per-subject accuracy
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd

from unified_championship_engine import UnifiedChampionshipEngine

ROOT = Path(__file__).parent


def main():
    print("=== 5-FOLD CROSS-VALIDATION OF UNIFIED CHAMPIONSHIP ENGINE ===")
    cache_m = np.load(ROOT / "multimodal_features_all.npz", allow_pickle=True)
    cache_t = np.load(ROOT / "temporal_streams_all.npz", allow_pickle=True)

    X_multi = cache_m["X_train"]
    all_paths = list(cache_m["train_paths"])
    X_skel = cache_t["X_skel_train"]
    X_imu = cache_t["X_imu_train"]

    oof_predictions = []
    all_results = []
    cat_results = []
    subj_results = []

    t_start = time.time()

    for fold in range(5):
        t0 = time.time()
        train_df = pd.read_csv(ROOT / "splits" / f"fold_{fold}_train.csv")
        val_df = pd.read_csv(ROOT / "splits" / f"fold_{fold}_val.csv")

        print(f"\n--- Fold {fold} (Train: {len(train_df)}, Val: {len(val_df)}) ---")
        engine = UnifiedChampionshipEngine()
        engine.fit(train_df, X_multi, X_skel, X_imu, all_paths)

        pred_df = engine.predict(val_df, X_multi, X_skel, X_imu, all_paths)
        val_df["pred"] = val_df["qa_id"].map(dict(zip(pred_df["qa_id"], pred_df["prediction"])))
        val_df["correct"] = (val_df["pred"] == val_df["answer"]).astype(int)
        val_df["fold"] = fold

        oof_predictions.append(val_df)

        fold_acc = val_df["correct"].mean() * 100
        print(f"Fold {fold} Acc: {val_df['correct'].sum()}/{len(val_df)} ({fold_acc:.2f}%) in {time.time() - t0:.1f}s")

        all_results.append({
            "fold": fold,
            "samples": len(val_df),
            "correct": val_df["correct"].sum(),
            "acc": fold_acc
        })

        for cat, grp in val_df.groupby("category"):
            cat_results.append({
                "fold": fold,
                "category": cat,
                "correct": grp["correct"].sum(),
                "total": len(grp),
                "acc": grp["correct"].mean() * 100
            })

        for subj, grp in val_df.groupby("subject_id"):
            subj_results.append({
                "fold": fold,
                "subject": subj,
                "correct": grp["correct"].sum(),
                "total": len(grp),
                "acc": grp["correct"].mean() * 100
            })

    full_oof = pd.concat(oof_predictions, ignore_index=True)
    full_oof.to_csv(ROOT / "oof_unified_championship_engine.csv", index=False)

    print("\n" + "=" * 80)
    print("                     5-FOLD CROSS-VALIDATION SUMMARY REPORT")
    print("=" * 80)

    total_correct = full_oof["correct"].sum()
    total_samples = len(full_oof)
    overall_acc = total_correct / total_samples * 100

    print(f"Total OOF Accuracy: {total_correct}/{total_samples} ({overall_acc:.2f}%)")
    print(f"Baseline was: 3013/4087 (73.72%)")
    print(f"Net Gain across all folds: {total_correct - 3013:+d} correct answers (+{overall_acc - 73.72:.2f}%)")

    print("\n=== Category Performance Comparison ===")
    print(f"{'Category':22s} | {'Baseline v3':15s} | {'Unified Engine':16s} | {'Delta':10s}")
    print("-" * 75)

    baseline_cat = {
        "combination": (691, 790, 87.47),
        "emotion": (423, 809, 52.29),
        "multi": (613, 809, 75.77),
        "object_interaction": (113, 133, 84.96),
        "sequence": (116, 308, 37.66),
        "single": (1057, 1238, 85.38)
    }

    df_cat = pd.DataFrame(cat_results)
    for cat in sorted(baseline_cat.keys()):
        grp = full_oof[full_oof.category == cat]
        corr = grp["correct"].sum()
        tot = len(grp)
        acc = corr / tot * 100
        b_corr, b_tot, b_acc = baseline_cat[cat]
        delta = acc - b_acc
        print(f"{cat:22s} | {b_corr:4d}/{b_tot:4d} ({b_acc:5.2f}%) | {corr:4d}/{tot:4d} ({acc:5.2f}%) | {delta:+6.2f}% ({corr - b_corr:+d})")

    print("\n=== Per-Subject Generalization ===")
    df_subj = pd.DataFrame(subj_results).groupby("subject").agg({"correct": "sum", "total": "sum"}).reset_index()
    df_subj["acc"] = df_subj["correct"] / df_subj["total"] * 100
    print(df_subj.to_string(index=False))
    print(f"\nTotal elapsed time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
