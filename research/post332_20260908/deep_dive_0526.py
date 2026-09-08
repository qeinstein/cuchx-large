#!/usr/bin/env python3
"""Focused, label-free audit of test_0526 (phone versus keyboard).

The exact-template donor group is evaluated both at full test support and under
leave-one-user-out training transfer.  Feature models are diagnostics only; this
script never edits a submission.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
QID = "test_0526"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sig(r):
    return (r["source"], r["category"], r["question"].strip(),
            tuple(sorted(r[L].strip() for L in "ABCD")))


def semantic(r):
    a = str(r["answer"])
    return r[a[0]].strip()


def donor_audit():
    tr = [r for r in read(ROOT / "training_qa.csv")
          if r["category"] == "object_interaction"]
    for r in tr:
        r["user"] = re.search(r"(user\d+)", r["path"]).group(1)
    test = next(r for r in read(ROOT / "test_qa.csv") if r["qa_id"] == QID)
    group = [r for r in tr if sig(r) == sig(test)]
    held = []
    for r in group:
        donors = [x for x in group if x["user"] != r["user"]]
        counts = Counter(semantic(x) for x in donors)
        top = counts.most_common()
        # The tie break is irrelevant to the reported accuracy here: both
        # possible top labels are wrong for the held-out row in this group.
        pred = top[0][0]
        held.append({"qa_id": r["qa_id"], "user": r["user"],
                     "truth": semantic(r), "prediction": pred,
                     "support": len(donors), "counts": dict(counts),
                     "correct": int(pred == semantic(r))})
    return {
        "test_options": {L: test[L].strip() for L in "ABCD"},
        "test_full_donor_support": len(group),
        "test_full_donor_counts": dict(Counter(semantic(x) for x in group)),
        "leave_user_out": held,
        "leave_user_out_accuracy": sum(x["correct"] for x in held) / len(held),
    }


def feature_diagnostics():
    # Keep this optional dependency isolated: it is a diagnostic disagreement
    # check, not part of the production solver.
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    meta = pd.read_csv(ROOT / "champ/meta.csv")
    feats = pd.read_csv(ROOT / "champ/feats.csv")
    meta = meta.merge(feats, on="unit_dir", how="left")
    cols = [c for c in meta if c.startswith(("sk_", "imu_", "rad_"))] + ["nf", "f0"]
    d = meta[(meta.kind == "train_harn") & meta.action.isin(
        ["17_Tap_the_keyboard", "24_Use_a_mobile_phone", "26_Play_games"])].copy()
    raw = d[cols].replace([np.inf, -np.inf], np.nan)
    imp, scl = SimpleImputer(), StandardScaler()
    X = scl.fit_transform(imp.fit_transform(raw))
    y = np.where(d.action.eq("17_Tap_the_keyboard"), "typing", "phone")
    groups = d.user.astype(str).to_numpy()
    acc = {}
    for k in (3, 5, 11):
        ok = []
        for ii, jj in GroupKFold(5).split(X, y, groups):
            model = KNeighborsClassifier(k, weights="distance").fit(X[ii], y[ii])
            ok.extend((model.predict(X[jj]) == y[jj]).tolist())
        acc[str(k)] = {"correct": int(sum(ok)), "n": len(ok), "accuracy": sum(ok) / len(ok)}
    q = meta[meta.qa_path == "LM_test_0016"]
    xx = scl.transform(imp.transform(q[cols].replace([np.inf, -np.inf], np.nan)))
    test = {}
    model = KNeighborsClassifier(11, weights="distance").fit(X, y)
    test["knn11_prediction"] = str(model.predict(xx)[0])
    test["knn11_probabilities"] = {str(c): float(p) for c, p in zip(model.classes_, model.predict_proba(xx)[0])}
    row = q.iloc[0]
    return {"training_rows": len(d), "classes": dict(Counter(y)),
            "subject_disjoint_knn": acc, "test": test,
            "clip_features": {c: float(row[c]) for c in ("nf", "f0", "sk_v_mean",
                                                            "sk_vwrist_mean", "imu_gyr_mean")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-features", action="store_true",
                   help="only run exact-template transfer (no sklearn diagnostic)")
    a = p.parse_args()
    result = {"qa_id": QID, "donor_group": donor_audit()}
    phone = OUT / "phone_typing_probe.json"
    if phone.exists():
        result["cached_phone_typing_probe"] = next(
            x for x in json.loads(phone.read_text())["test"] if x["clip"] == "LM_test_0016")
    if not a.no_features:
        result["feature_diagnostics"] = feature_diagnostics()
    result["interpretation"] = (
        "Exact option donors are a 3-keyboard/3-phone tie, and leave-user-out transfer "
        "is intentionally evaluated on every row of that group.  Feature diagnostics "
        "disagree (KNN versus aggregate/logistic); no object correction is approved."
    )
    (OUT / "deep_dive_0526.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
