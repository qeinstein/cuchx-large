#!/usr/bin/env python3
"""EXP-EMO-DINO-TRAJ-002: integrate the trajectory descriptor into the champion's OWN
pairwise objective, instead of training an independent override policy.

Rationale
---------
`probe_dino_trajectory.py` showed the descriptor has real oracle-level orientation signal
(0.91-0.96 on true-label pairs) but an independently-trained swap policy is strongly
net-negative against the champion's actual disagreement subset (worst -127, best still -8).
That is a calibration-shift failure of the *override* formulation, not proof the signal is
useless.  Session 2 found the analogous thing for DTW features: killed as a standalone
override, but +0.9pp when added as extra columns to the champion's existing pairwise
HistGradientBoostingClassifier (`champ/emopair.py`) and evaluated on the SAME orientation task
the champion already solves internally, fit end-to-end together with its other evidence.

This script repeats exactly that integration test for the DINO trajectory descriptor:
arm 1 = champion's pairwise feature set alone (block_features diffs/ranks + slot priors);
arm 2 = arm 1 + signed difference/ratio of the trajectory descriptor between the two clips.
Both arms are HistGradientBoostingClassifier, fit subject-disjoint, on the pairwise manner
ORIENTATION task (same construction as champ/emopair.py's actual training objective).

This is still training-only / analysis-only.  No submission is written.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAMP = ROOT / "champ"
sys.path.insert(0, str(CHAMP))

from core import load_all, opts, gt_letters, mgroup, GROUPS, PHYS, block_features  # noqa: E402
import emopair as EP  # noqa: E402
import core as CORE  # noqa: E402

sys.path.insert(0, str(HERE))
from probe_dino_trajectory import dino_trajectory_descriptor  # noqa: E402

tr, te, meta = load_all()
em = tr[tr.category == "emotion"].copy()
em["blk"] = list(zip(em.user, em.aa, em.bb))
em["lab"] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
ufold = {}
for f in range(5):
    for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique():
        ufold[u] = f
mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}

ACT = ["single", "multi", "combination", "sequence"]
tpool = {}
import collections
tpool = collections.defaultdict(set)
for _, r in tr[(tr.source == "HAU") & tr.category.isin(ACT)].iterrows():
    for L in str(r["answer"]):
        if L in "ABCD":
            for p in str(r[L]).split(","):
                tpool[(r.user, r.aa, r.bb)].add(p.strip())

mm = CORE.fit_manner(tr, meta)

DN = np.load(CHAMP / "dino_frames.npz")
_desc_cache = {}


def descriptor(path):
    if path not in _desc_cache:
        if path in DN:
            try:
                _desc_cache[path] = dino_trajectory_descriptor(np.asarray(DN[path]))
            except Exception:
                _desc_cache[path] = None
        else:
            _desc_cache[path] = None
    return _desc_cache[path]


DCOLS = [f"traj_{i}" for i in range(123)]

X = []
Y = []
U = []
for b, g in em.groupby("blk"):
    g = g.sort_values("cc")
    rows = [r for _, r in g.iterrows()]
    k = len(rows)
    if k < 2:
        continue
    bf = block_features([r.path for r in rows], mfeat, k)
    labs = [r.lab for r in rows]
    ctx = EP.pool_context(tpool.get(b))
    for i, j in itertools.combinations(range(k), 2):
        if labs[i] == labs[j]:
            continue
        di, dj = descriptor(rows[i].path), descriptor(rows[j].path)
        traj = {}
        if di is not None and dj is not None:
            diff = (di - dj).astype(float)
            ratio = diff / (np.abs(di) + np.abs(dj) + 1e-6)
            traj = {**{f"traj_d_{n}": v for n, v in enumerate(diff)},
                    **{f"traj_r_{n}": v for n, v in enumerate(ratio)}}
        # IMPORTANT: traj is a fixed physical fact about the (clip_i, clip_j) pair and must NOT
        # depend on which orientation (ma, mb) is being scored -- exactly like champ/emopair.py
        # keeps bf_i/bf_j and its DTW/PHYS diff features fixed across the correct/swap rows.
        # An earlier version of this script negated `traj` for the swap row, which encodes the
        # label into the feature (trivial leakage) and inflated the measured lift; this version
        # matches the champion's own convention and lets the classifier learn the interaction
        # with grp_i/grp_j itself, as it already does for DTW and the physical-difference PHYS
        # columns.
        for (ma, mb, y) in [(labs[i], labs[j], 1), (labs[j], labs[i], 0)]:
            f = EP.pair_row(bf[i], bf[j], i, j, k, ma, mb, mm, ctx)
            f.update(traj)
            X.append(f)
            Y.append(y)
        U.append(b[0])
        U.append(b[0])

X = pd.DataFrame(X)
Y = np.array(Y)
U = np.array(U)
traj_cols = [c for c in X.columns if c.startswith("traj_")]
print(f"pair rows {len(X)} ({len(X)//2} pairs)  champion cols {X.shape[1]-len(traj_cols)}  traj cols {len(traj_cols)}")
cov = X[traj_cols].notna().any(axis=1).mean() if traj_cols else 0.0
print(f"fraction of pairs with usable trajectory descriptor: {cov:.3f}")


def ev(cols, label):
    Xv = X[cols].to_numpy(float)
    p = np.zeros(len(Y))
    per_fold = {}
    for f in range(5):
        m = np.array([ufold.get(u, -1) == f for u in U])
        if m.sum() == 0 or (~m).sum() == 0:
            continue
        c = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=5, l2_regularization=1.0, random_state=0
        ).fit(Xv[~m], Y[~m])
        p[m] = c.predict_proba(Xv[m])[:, 1]
    ok = 0
    n = 0
    fold_acc = {}
    fold_n = {}
    for t in range(0, len(Y), 2):
        n += 1
        c = int(p[t] > p[t + 1])
        ok += c
        f = ufold.get(U[t], -1)
        fold_n[f] = fold_n.get(f, 0) + 1
        fold_acc[f] = fold_acc.get(f, 0) + c
    print(f"  {label:44s} orientation acc {ok/n:.4f}  ({ok}/{n})  nfeat={len(cols)}")
    for f in sorted(fold_n):
        print(f"      fold {f}: {fold_acc[f]}/{fold_n[f]} = {fold_acc[f]/fold_n[f]:.4f}")
    return ok / n


champ_cols = [c for c in X.columns if c not in traj_cols]
print("\n=== pairwise manner-orientation accuracy, subject-disjoint ===")
base_acc = ev(champ_cols, "champion pairwise features [BASELINE]")
traj_acc = ev(traj_cols, "DINO trajectory descriptor ONLY")
comb_acc = ev(champ_cols + traj_cols, "champion + trajectory  [COMBINED]")

out_dir = HERE / "results_joint"
out_dir.mkdir(exist_ok=True)
pd.DataFrame(
    [
        dict(arm="champion_only", accuracy=base_acc, n_pairs=len(Y) // 2),
        dict(arm="trajectory_only", accuracy=traj_acc, n_pairs=len(Y) // 2),
        dict(arm="combined", accuracy=comb_acc, n_pairs=len(Y) // 2),
    ]
).to_csv(out_dir / "orientation_accuracy_summary.csv", index=False)
print(f"\nwrote {out_dir / 'orientation_accuracy_summary.csv'}")
