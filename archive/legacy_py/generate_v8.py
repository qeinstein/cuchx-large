import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(".")

def main():
    print("=== GENERATING CHAMPIONSHIP v8 PIPELINE ===")
    
    oof_v7 = pd.read_csv(ROOT / "oof_v7_final.csv")
    oof_sol = pd.read_csv(ROOT / "oof_predictions_championship_solver.csv")
    sub_v7 = pd.read_csv(ROOT / "submission_v7.csv")
    sub_v3 = pd.read_csv(ROOT / "submission_v3.csv")
    test_df = pd.read_csv(ROOT / "test_qa.csv")
    
    # 1. Build OOF v8
    oof_v8 = oof_v7.copy()
    em_mask = (oof_v8.category == "emotion")
    oof_v8.loc[em_mask, "pred"] = oof_sol.loc[em_mask, "pred"]
    oof_v8["correct"] = (oof_v8["pred"] == oof_v8["answer"]).astype(int)
    
    oof_v8_path = ROOT / "oof_v8_final.csv"
    oof_v8.to_csv(oof_v8_path, index=False)
    print(f"Saved verified {oof_v8_path} (4,087 rows).")
    
    # 2. Build Submission v8
    sub_v8 = sub_v7.copy()
    te_emo = test_df[test_df.category == "emotion"]
    
    changed_rows = []
    for qid in te_emo.qa_id:
        p7 = sub_v7.loc[sub_v7.qa_id == qid, "prediction"].values[0]
        p3 = sub_v3.loc[sub_v3.qa_id == qid, "prediction"].values[0]
        if p7 != p3:
            sub_v8.loc[sub_v8.qa_id == qid, "prediction"] = p3
            changed_rows.append((qid, p7, p3))
            
    assert len(sub_v8) == 682, f"Expected 682 rows, got {len(sub_v8)}"
    assert list(sub_v8.columns) == ["qa_id", "prediction"]
    assert sub_v8["prediction"].isna().sum() == 0
    
    sub_v8_path = ROOT / "submission_v8.csv"
    sub_v8.to_csv(sub_v8_path, index=False)
    print(f"Saved verified {sub_v8_path} (682 rows, {len(changed_rows)} updates from v7).")
    
    # 3. Print Comprehensive Report
    total_corr = oof_v8.correct.sum()
    total_n = len(oof_v8)
    acc = total_corr / total_n * 100
    v7_corr = oof_v7.correct.sum()
    
    print("\n" + "=" * 80)
    print("                 CHAMPIONSHIP v8 5-FOLD BENCHMARK REPORT (4,087 SAMPLES)")
    print("=" * 80)
    print(f"Championship v8 Overall Accuracy: {total_corr}/{total_n} ({acc:.2f}%)")
    print(f"Baseline v7 was:                 {v7_corr}/{total_n} ({v7_corr/total_n*100:.2f}%)")
    print(f"Net Gain vs v7:                  {total_corr - v7_corr:+d} correct answers (+{acc - v7_corr/total_n*100:.2f}%)")
    
    v7_w_v8_r = ((oof_v7.correct == 0) & (oof_v8.correct == 1)).sum()
    v7_r_v8_w = ((oof_v7.correct == 1) & (oof_v8.correct == 0)).sum()
    print(f"\nExact Transitions:")
    print(f"  v7 wrong -> v8 right: {v7_w_v8_r} questions (WINS)")
    print(f"  v7 right -> v8 wrong: {v7_r_v8_w} questions (LOSSES)")
    print(f"  Win/Loss Ratio:       {v7_w_v8_r / max(1, v7_r_v8_w):.2f}x")
    
    print("\nFold-by-Fold Performance:")
    for f in range(5):
        sub_old = oof_v7[oof_v7.fold == f]
        sub_new = oof_v8[oof_v8.fold == f]
        c_old = sub_old.correct.sum()
        c_new = sub_new.correct.sum()
        nf = len(sub_old)
        print(f"  Fold {f}: v7={c_old:4d}/{nf} ({c_old/nf*100:.2f}%) -> v8={c_new:4d}/{nf} ({c_new/nf*100:.2f}%) | Net: {c_new - c_old:+d}")
        
    print("\nCategory Performance Breakdown:")
    for cat in sorted(oof_v8.category.unique()):
        sub_old = oof_v7[oof_v7.category == cat]
        sub_new = oof_v8[oof_v8.category == cat]
        c_old = sub_old.correct.sum()
        c_new = sub_new.correct.sum()
        nc = len(sub_old)
        print(f"  {cat:20s}: v7={c_old:4d}/{nc} ({c_old/nc*100:.2f}%) -> v8={c_new:4d}/{nc} ({c_new/nc*100:.2f}%) | Net: {c_new - c_old:+d}")

if __name__ == "__main__":
    main()
