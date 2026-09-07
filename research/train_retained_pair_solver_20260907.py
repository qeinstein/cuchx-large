"""
train_retained_pair_solver_20260907.py
Empirical 2nd-generation Retained-Pair Solver.
Evaluates 3-way pair classifier:
  Class 0: (0, 1) [Trials 1 & 2 retained, Trial 3 withheld]
  Class 1: (0, 2) [Trials 1 & 3 retained, Trial 2 withheld]
  Class 2: (1, 2) [Trials 2 & 3 retained, Trial 1 withheld]
Evaluated with 5-fold Subject-Disjoint CV across 265 training sessions.
"""

import sys, os, itertools, math, joblib
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, "champ")
from core import load_all, PHYS

tr, te, meta = load_all()
meta_idx = meta.set_index("qa_path")
feats = pd.read_csv("champ/feats.csv").set_index("unit_dir")

e = tr[tr.category == "emotion"].copy()
e["manner"] = e.apply(lambda r: str(r.get(r.answer)).strip(), axis=1)

sessions_by_user = defaultdict(list)
for (u, aa, bb), g in e.groupby(["user", "aa", "bb"]):
    if len(g) == 3:
        g = g.sort_values("cc")
        sessions_by_user[u].append(((aa, bb), [r for _, r in g.iterrows()]))

users = sorted(sessions_by_user.keys())
print(f"Total users with 3-trial sessions: {len(users)}, total sessions: {sum(len(v) for v in sessions_by_user.values())}")

pair_data = []
for u in users:
    for (aa, bb), rows in sessions_by_user[u]:
        c = []
        for r in rows:
            p = r.path
            mr = meta_idx.loc[p] if p in meta_idx.index else None
            u_dir = mr.get("unit_dir") if mr is not None else None
            f = feats.loc[u_dir] if (pd.notna(u_dir) and u_dir in feats.index) else {}
            c.append({
                "path": p,
                "manner": r.manner,
                "trial": int(r.path.split("/")[-1].split("-")[-1]) if len(r.path.split("/")[-1].split("-")) == 3 else int(r.cc),
                "nf": float(mr.get("nf", np.nan)) if mr is not None else np.nan,
                "t0": float(mr.get("t0", np.nan)) if mr is not None else np.nan,
                "imu": float(f.get("imu_acc_std", np.nan)),
                "spd": float(f.get("sk_v_mean", np.nan)),
            })
        
        pairs = [
            (0, 1, 0),
            (0, 2, 1),
            (1, 2, 2),
        ]
        for i0, i1, target_class in pairs:
            c0, c1 = c[i0], c[i1]
            gap = abs(c1["t0"] - c0["t0"]) if (np.isfinite(c0["t0"]) and np.isfinite(c1["t0"])) else np.nan
            nf_ratio = c1["nf"] / (c0["nf"] + 1e-4) if (np.isfinite(c0["nf"]) and np.isfinite(c1["nf"])) else np.nan
            imu_ratio = c1["imu"] / (c0["imu"] + 1e-4) if (np.isfinite(c0["imu"]) and np.isfinite(c1["imu"])) else np.nan
            spd_ratio = c1["spd"] / (c0["spd"] + 1e-4) if (np.isfinite(c0["spd"]) and np.isfinite(c1["spd"])) else np.nan
            
            pair_data.append({
                "user": u, "aa": aa, "bb": bb, "pair_type": target_class,
                "p0": c0["path"], "p1": c1["path"],
                "m0": c0["manner"], "m1": c1["manner"],
                "t0_true": c0["trial"], "t1_true": c1["trial"],
                "triad": tuple(sorted([c[0]["manner"], c[1]["manner"], c[2]["manner"]])),
                "nf0": c0["nf"], "nf1": c1["nf"],
                "nf_diff": c1["nf"] - c0["nf"],
                "nf_ratio": nf_ratio,
                "nf_max": max(c0["nf"], c1["nf"]),
                "nf_min": min(c0["nf"], c1["nf"]),
                "imu0": c0["imu"], "imu1": c1["imu"],
                "imu_diff": c1["imu"] - c0["imu"],
                "imu_ratio": imu_ratio,
                "spd0": c0["spd"], "spd1": c1["spd"],
                "spd_diff": c1["spd"] - c0["spd"],
                "spd_ratio": spd_ratio,
                "gap": gap
            })

df = pd.DataFrame(pair_data)
print(f"Total simulated pair samples: {len(df)}")
print(df.pair_type.value_counts())

feature_cols = [
    "nf0", "nf1", "nf_diff", "nf_ratio", "nf_max", "nf_min",
    "imu0", "imu1", "imu_diff", "imu_ratio",
    "spd0", "spd1", "spd_diff", "spd_ratio",
    "gap"
]

X = df[feature_cols].copy()
for col in feature_cols:
    X[col] = X[col].fillna(X[col].median())

y = df["pair_type"].values
groups = df["user"].values

gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(df), dtype=int)
oof_probs = np.zeros((len(df), 3), dtype=float)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    clf = HistGradientBoostingClassifier(random_state=42, max_depth=4, min_samples_leaf=10)
    clf.fit(X_tr, y_tr)
    
    oof_probs[val_idx] = clf.predict_proba(X_val)
    oof_preds[val_idx] = clf.predict(X_val)

acc = accuracy_score(y, oof_preds)
print(f"\n=== Subject-Disjoint 5-Fold OOF Pair Classification Accuracy: {acc:.4f} ({np.sum(y == oof_preds)}/{len(df)}) ===")
print("\nClassification Report:")
print(classification_report(y, oof_preds, target_names=["(0, 1) [T1+T2]", "(0, 2) [T1+T3]", "(1, 2) [T2+T3]"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y, oof_preds))

final_pair_model = HistGradientBoostingClassifier(random_state=42, max_depth=4, min_samples_leaf=10)
final_pair_model.fit(X, y)
joblib.dump({"model": final_pair_model, "feature_cols": feature_cols, "median_impute": X.median().to_dict()}, "scratch/retained_pair_model.joblib")
print("\nSaved final pair model to scratch/retained_pair_model.joblib")
