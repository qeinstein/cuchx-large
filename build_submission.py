"""Build improved submission by fusing exact sequence-multi constraints and calibrated sensor specialists."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent


def main():
    cache = np.load(ROOT / "sensor_features_all.npz", allow_pickle=True)
    X_train, train_paths = cache["X_train"], list(cache["train_paths"])
    X_test, test_paths = cache["X_test"], list(cache["test_paths"])
    path_to_idx = {p: i for i, p in enumerate(train_paths)}
    test_path_to_idx = {p: i for i, p in enumerate(test_paths)}

    train_df = pd.read_csv(ROOT / "training_qa.csv")
    test_df = pd.read_csv(ROOT / "test_qa.csv").drop(columns=["prediction"], errors="ignore")
    parent_sub = pd.read_csv(ROOT / "submission_077777.csv")

    new_sub = parent_sub.copy()
    pred_map = dict(zip(new_sub["qa_id"], new_sub["prediction"]))
    changes = []

    # 1. Exact Sequence-Multi Consistency Guarantees
    seq_clips = test_df[test_df.category == "sequence"]["path"].unique()
    for p in seq_clips:
        g = test_df[test_df.path == p]
        seq_r = g[g.category == "sequence"].iloc[0]
        seq_actions = set(str(seq_r[l]).strip().lower() for l in "ABCD")

        multi_rows = g[g.category == "multi"]
        if len(multi_rows):
            m_row = multi_rows.iloc[0]
            qid = m_row["qa_id"]
            curr_pred = pred_map[qid]
            curr_letters = set(curr_pred)

            guaranteed = [l for l in "ABCD" if str(m_row[l]).strip().lower() in seq_actions]
            combined = sorted(curr_letters.union(guaranteed))
            new_pred = "".join(combined)
            if new_pred != curr_pred:
                changes.append((qid, "multi_seq_guarantee", curr_pred, new_pred))
                pred_map[qid] = new_pred

    # 2. Train HARn Action Specialist
    tr_harn = train_df[(train_df.source == "HARn") & (train_df.category == "single")].copy()
    tr_harn["target"] = tr_harn.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)
    X_tr_harn = np.array([X_train[path_to_idx[p]][120:] for p in tr_harn.path])
    scaler_harn = StandardScaler()
    X_tr_harn_s = scaler_harn.fit_transform(np.nan_to_num(X_tr_harn))
    rf_harn = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_harn.fit(X_tr_harn_s, tr_harn.target)
    harn_classes = list(rf_harn.classes_)

    for idx, row in test_df[(test_df.source == "HARn") & (test_df.category == "single")].iterrows():
        p = row["path"]
        if p in test_path_to_idx:
            x = scaler_harn.transform(np.nan_to_num(X_test[test_path_to_idx[p] : test_path_to_idx[p] + 1, 120:]))
            probs = rf_harn.predict_proba(x)[0]
            scores = [
                probs[harn_classes.index(str(row[opt]).strip().lower())]
                if str(row[opt]).strip().lower() in harn_classes
                else 0.0
                for opt in ["A", "B", "C"]
            ]
            sorted_s = sorted(scores, reverse=True)
            best_opt = ["A", "B", "C"][int(np.argmax(scores))]
            qid = row["qa_id"]
            curr_pred = pred_map[qid]
            if best_opt != curr_pred and (sorted_s[0] - sorted_s[1]) >= 0.25:
                changes.append((qid, "HARn_single_override", curr_pred, best_opt))
                pred_map[qid] = best_opt

    # 3. Train HARn Object Specialist
    tr_obj = train_df[(train_df.source == "HARn") & (train_df.category == "object_interaction")].copy()
    tr_obj["target"] = tr_obj.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)
    X_tr_obj = np.array([X_train[path_to_idx[p]] for p in tr_obj.path])
    scaler_obj = StandardScaler()
    X_tr_obj_s = scaler_obj.fit_transform(np.nan_to_num(X_tr_obj))
    rf_obj = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_obj.fit(X_tr_obj_s, tr_obj.target)
    obj_classes = list(rf_obj.classes_)

    for idx, row in test_df[(test_df.source == "HARn") & (test_df.category == "object_interaction")].iterrows():
        p = row["path"]
        if p in test_path_to_idx:
            x = scaler_obj.transform(np.nan_to_num(X_test[test_path_to_idx[p] : test_path_to_idx[p] + 1]))
            probs = rf_obj.predict_proba(x)[0]
            scores = [
                probs[obj_classes.index(str(row[opt]).strip().lower())]
                if str(row[opt]).strip().lower() in obj_classes
                else 0.0
                for opt in ["A", "B", "C", "D"]
            ]
            sorted_s = sorted(scores, reverse=True)
            best_opt = ["A", "B", "C", "D"][int(np.argmax(scores))]
            qid = row["qa_id"]
            curr_pred = pred_map[qid]
            if best_opt != curr_pred and (sorted_s[0] - sorted_s[1]) >= 0.25:
                changes.append((qid, "HARn_obj_override", curr_pred, best_opt))
                pred_map[qid] = best_opt

    print(f"Total deliberate changes: {len(changes)}")
    for ch in changes:
        print(f"  {ch[0]}: {ch[1]} ({ch[2]} -> {ch[3]})")

    new_sub["prediction"] = new_sub["qa_id"].map(pred_map)

    # Format verification
    assert len(new_sub) == 682, f"Expected 682 rows, got {len(new_sub)}"
    assert list(new_sub.columns) == ["qa_id", "prediction"], f"Invalid columns: {new_sub.columns}"
    assert new_sub["prediction"].isna().sum() == 0, "Null predictions found!"

    out_file = ROOT / "submission_v1.csv"
    new_sub.to_csv(out_file, index=False)
    print(f"Successfully generated {out_file} ({len(new_sub)} rows)!")


if __name__ == "__main__":
    main()
