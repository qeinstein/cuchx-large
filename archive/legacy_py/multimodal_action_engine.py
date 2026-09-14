"""Unified Multimodal Action and Reasoning Engine.

Fuses 1688-dimensional features (Sensor + DINOv2 + ResNet + Thermal) with exact sequence
theorems and cross-question consistency constraints to generate optimal predictions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent


def main():
    print("Loading unified 1688-dimensional multimodal cache...")
    cache = np.load(ROOT / "multimodal_features_all.npz", allow_pickle=True)
    X_train = cache["X_train"]  # (1333, 1688)
    X_test = cache["X_test"]  # (208, 1688)

    train_paths = list(cache["train_paths"])
    test_paths = list(cache["test_paths"])
    path_to_idx = {p: i for i, p in enumerate(train_paths)}
    test_path_to_idx = {p: i for i, p in enumerate(test_paths)}

    train_df = pd.read_csv(ROOT / "training_qa.csv")
    test_df = pd.read_csv(ROOT / "test_qa.csv").drop(columns=["prediction"], errors="ignore")

    # Start from our proven peak baseline submission_v3.csv (LB: 0.78947)
    base_sub = pd.read_csv(ROOT / "submission_v3.csv")
    pred_map = dict(zip(base_sub["qa_id"], base_sub["prediction"]))
    changes = []

    # ---------------------------------------------------------
    # 1. Exact Sequence-Multi Consistency Guarantees
    # ---------------------------------------------------------
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
                changes.append((qid, "seq_guarantee", curr_pred, new_pred))
                pred_map[qid] = new_pred

    # ---------------------------------------------------------
    # 2. Train 40-Action Unified Multimodal Classifier (1688 dims)
    # ---------------------------------------------------------
    hau = train_df[train_df.source == "HAU"]
    vocab = sorted(
        list(
            set(
                str(r[r.answer]).strip().lower()
                for _, r in hau[hau.category == "single"].iterrows()
                if r.answer in "ABCD"
            )
        )
    )
    act_to_idx = {a: i for i, a in enumerate(vocab)}

    clip_labels = {}
    for p, g in hau.groupby("path"):
        vec = np.zeros(len(vocab), dtype=np.float32)
        s = g[g.category == "single"]
        if len(s):
            vec[act_to_idx[str(s.iloc[0][s.iloc[0]["answer"]]).strip().lower()]] = 1.0
        m = g[g.category == "multi"]
        if len(m):
            for l in m.iloc[0]["answer"]:
                if l in "ABCD":
                    a = str(m.iloc[0][l]).strip().lower()
                    if a in act_to_idx:
                        vec[act_to_idx[a]] = 1.0
        c = g[g.category == "combination"]
        if len(c):
            for x in str(c.iloc[0][c.iloc[0]["answer"]]).split(","):
                a = x.strip().lower()
                if a in act_to_idx:
                    vec[act_to_idx[a]] = 1.0
        seq = g[g.category == "sequence"]
        if len(seq):
            for l in "ABCD":
                a = str(seq.iloc[0][l]).strip().lower()
                if a in act_to_idx:
                    vec[act_to_idx[a]] = 1.0
        clip_labels[p] = vec

    hau_clips = [p for p in hau["path"].unique() if p in clip_labels]
    X_hau_tr = np.array([X_train[path_to_idx[p]] for p in hau_clips])
    Y_hau_tr = np.array([clip_labels[p] for p in hau_clips])

    scaler_m = StandardScaler()
    X_hau_tr_s = scaler_m.fit_transform(np.nan_to_num(X_hau_tr))
    X_test_s = scaler_m.transform(np.nan_to_num(X_test))

    print(f"Training unified ExtraTrees classifier on {X_hau_tr.shape} across 40 classes...")
    clf_m = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
    clf_m.fit(X_hau_tr_s, Y_hau_tr)

    probs_list = clf_m.predict_proba(X_test_s)
    te_probs = np.array([p[:, 1] if p.shape[1] > 1 else np.zeros(len(p)) for p in probs_list]).T

    # ---------------------------------------------------------
    # 3. High-Confidence Multi-Action Additions
    # ---------------------------------------------------------
    for idx, r in test_df[test_df.category == "multi"].iterrows():
        p = r["path"]
        if p in test_path_to_idx:
            c_idx = test_path_to_idx[p]
            probs = te_probs[c_idx]
            opt_p = [
                probs[act_to_idx[str(r[l]).strip().lower()]] if str(r[l]).strip().lower() in act_to_idx else 0.0
                for l in "ABCD"
            ]

            qid = r["qa_id"]
            curr_pred = pred_map[qid]
            strong_additions = set(curr_pred)

            for i, l in enumerate("ABCD"):
                if opt_p[i] >= 0.28:
                    strong_additions.add(l)

            new_pred = "".join(sorted(strong_additions))
            if new_pred != curr_pred:
                changes.append((qid, "multi_confidence_add", curr_pred, new_pred))
                pred_map[qid] = new_pred

    # ---------------------------------------------------------
    # 4. HARn Single Action Specialist (Skeleton + DINOv2)
    # ---------------------------------------------------------
    tr_harn = train_df[(train_df.source == "HARn") & (train_df.category == "single")].copy()
    tr_harn["target"] = tr_harn.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)

    # Skeleton: 120..280, DINOv2: 280..664
    X_tr_harn = np.array([X_train[path_to_idx[p]][120:664] for p in tr_harn.path])
    scaler_harn = StandardScaler()
    X_tr_harn_s = scaler_harn.fit_transform(np.nan_to_num(X_tr_harn))
    X_te_harn_s = scaler_harn.transform(np.nan_to_num(X_test[:, 120:664]))

    rf_harn = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_harn.fit(X_tr_harn_s, tr_harn.target)
    harn_classes = list(rf_harn.classes_)

    for idx, row in test_df[(test_df.source == "HARn") & (test_df.category == "single")].iterrows():
        p = row["path"]
        if p in test_path_to_idx:
            c_idx = test_path_to_idx[p]
            probs = rf_harn.predict_proba(X_te_harn_s[c_idx : c_idx + 1])[0]
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

    # ---------------------------------------------------------
    # 5. HARn Object Interaction Specialist (Skeleton + DINOv2)
    # ---------------------------------------------------------
    tr_obj = train_df[(train_df.source == "HARn") & (train_df.category == "object_interaction")].copy()
    tr_obj["target"] = tr_obj.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)

    X_tr_obj = np.array([X_train[path_to_idx[p]][:664] for p in tr_obj.path])
    scaler_obj = StandardScaler()
    X_tr_obj_s = scaler_obj.fit_transform(np.nan_to_num(X_tr_obj))
    X_te_obj_s = scaler_obj.transform(np.nan_to_num(X_test[:, :664]))

    rf_obj = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_obj.fit(X_tr_obj_s, tr_obj.target)
    obj_classes = list(rf_obj.classes_)

    for idx, row in test_df[(test_df.source == "HARn") & (test_df.category == "object_interaction")].iterrows():
        p = row["path"]
        if p in test_path_to_idx:
            c_idx = test_path_to_idx[p]
            probs = rf_obj.predict_proba(X_te_obj_s[c_idx : c_idx + 1])[0]
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

    print(f"\nTotal updates relative to baseline: {len(changes)}")
    counts = {}
    for ch in changes:
        counts[ch[1]] = counts.get(ch[1], 0) + 1
    for k, v in counts.items():
        print(f"  {k}: {v} questions")

    out_sub = base_sub.copy()
    out_sub["prediction"] = out_sub["qa_id"].map(pred_map)

    assert len(out_sub) == 682
    assert list(out_sub.columns) == ["qa_id", "prediction"]
    assert out_sub["prediction"].isna().sum() == 0

    out_file = ROOT / "submission_v5.csv"
    out_sub.to_csv(out_file, index=False)
    print(f"\nSuccessfully generated {out_file} (682 rows)!")


if __name__ == "__main__":
    main()
