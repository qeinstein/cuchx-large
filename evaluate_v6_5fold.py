"""Evaluate Championship v6 Engine across all 5 subject-held-out validation folds.

Outputs exact 5-fold cross-validation metrics across all 4,087 samples:
- overall accuracy
- per-category accuracy (single, combination, multi, sequence, emotion, object_interaction)
- delta vs v5 baseline (74.50%, 3045/4087)
- per-subject accuracy
- saves verified oof_championship_v6_verified.csv
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd

from championship_v6_engine import ChampionshipV6Engine

ROOT = Path(__file__).parent


def main():
    print("=== 5-FOLD CROSS-VALIDATION OF CHAMPIONSHIP v6 ENGINE ===")
    cache_ext = np.load(ROOT / "extended_features_all.npz", allow_pickle=True)
    cache_t = np.load(ROOT / "temporal_streams_all.npz", allow_pickle=True)

    X_all = cache_ext["X_train"]
    all_paths = list(cache_ext["train_paths"])
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
        engine = ChampionshipV6Engine()
        engine.fit(train_df, X_all, X_skel, X_imu, all_paths)

        pred_df = engine.predict(val_df, X_all, X_skel, X_imu, all_paths)
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
    full_oof.to_csv(ROOT / "oof_championship_v6_verified.csv", index=False)

    print("\n" + "=" * 85)
    print("                 CHAMPIONSHIP v6 5-FOLD BENCHMARK REPORT (4,087 SAMPLES)")
    print("=" * 85)

    total_correct = full_oof["correct"].sum()
    total_samples = len(full_oof)
    overall_acc = total_correct / total_samples * 100

    print(f"Championship v6 Total OOF Accuracy: {total_correct}/{total_samples} ({overall_acc:.2f}%)")
    print(f"Baseline v3 was: 3013/4087 (73.72%)")
    print(f"Baseline v5 was: 3045/4087 (74.50%)")
    print(f"Net Gain vs v3: {total_correct - 3013:+d} correct answers ({overall_acc - 73.72:+.2f}%)")
    print(f"Net Gain vs v5: {total_correct - 3045:+d} correct answers ({overall_acc - 74.50:+.2f}%)")

    print("\n=== Category Performance Comparison ===")
    print(f"{'Category':22s} | {'v5 Baseline':15s} | {'v6 Engine':16s} | {'Delta vs v5':12s}")
    print("-" * 75)

    v5_baseline_cat = {
        "combination": (703, 790, 88.99),
        "emotion": (395, 809, 48.83),
        "multi": (612, 809, 75.65),
        "object_interaction": (114, 133, 85.71),
        "sequence": (146, 308, 47.40),
        "single": (1075, 1238, 86.83)
    }

    for cat in sorted(v5_baseline_cat.keys()):
        grp = full_oof[full_oof.category == cat]
        corr = grp["correct"].sum()
        tot = len(grp)
        acc = corr / tot * 100
        b_corr, b_tot, b_acc = v5_baseline_cat[cat]
        delta = acc - b_acc
        print(f"{cat:22s} | {b_corr:4d}/{b_tot:4d} ({b_acc:5.2f}%) | {corr:4d}/{tot:4d} ({acc:5.2f}%) | {delta:+6.2f}% ({corr - b_corr:+d})")

    print(f"\nTotal elapsed time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
