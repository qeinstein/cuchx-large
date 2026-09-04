#!/usr/bin/env python3
"""EXP-EMO-DINO-TRAJ-003: does DTW ALIGNMENT of the raw DINO-Depth sequence (not a pooled
trajectory summary) carry manner signal the champion is missing?

Motivation
----------
Two independent probes this session (`probe_dino_trajectory.py`'s decision-level override,
and `probe_joint_integration.py`'s corrected joint-objective test) found that a rich, 8-phase-
pooled summary of the frame-delta trajectory contributes exactly zero weight once combined
honestly with the champion's existing evidence.  One remaining hypothesis, distinct from a
fixed-length pooled summary: manner differences might live in *where two clips' timelines
align or diverge* (a hesitation localized to one third of the clip, one trial's segments
running fractionally faster throughout) -- structure that pooling into 8 coarse phases can
smear out, but that dynamic time warping preserves as a per-frame alignment path.

Session 2 tested exactly this for SKELETON sequences (`emolab/dtw_feats.py`,
`emolab/pairtest.py`): standalone override killed, joint-objective integration gave a small,
real, all-folds-improved gain (0.9738 -> 0.9825, +0.9pp).  This script repeats that identical
methodology on the DINO-Depth (T, 384) sequence instead of the (T, 17, 3) skeleton, reusing
`emolab.dtw_feats.dtw_path`/`path_feats` unmodified.

Design safeguards carried over from the other two probes in this directory:
* every input reaching the warping/feature step is native-rate and label-free;
* the pairwise classifier is fit subject-disjoint, 5-fold;
* the promotion bar is champion + trajectory-DTW beating champion alone on the SAME
  orientation task the champion already solves internally, not an independent override policy;
* the feature that describes a (clip_i, clip_j) pair is held FIXED across the correct/swapped
  label rows (see RESULTS_joint_integration.md's leakage retraction) -- verified here by
  reusing dtw_feats.path_feats's existing directed-pair convention without introducing any
  new sign flip.
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
EMOLAB = ROOT / "emolab"
sys.path.insert(0, str(CHAMP))
sys.path.insert(0, str(EMOLAB))

from core import load_all, gt_letters, block_features, PHYS  # noqa: E402
import emopair as EP  # noqa: E402
import core as CORE  # noqa: E402
from dtw_feats import dtw_path, path_feats  # noqa: E402

tr, te, meta = load_all()
em = tr[tr.category == "emotion"].copy()
em["blk"] = list(zip(em.user, em.aa, em.bb))
em["lab"] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
ufold = {}
for f in range(5):
    for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique():
        ufold[u] = f
mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}

import collections

ACT = ["single", "multi", "combination", "sequence"]
tpool = collections.defaultdict(set)
for _, r in tr[(tr.source == "HAU") & tr.category.isin(ACT)].iterrows():
    for L in str(r["answer"]):
        if L in "ABCD":
            for p in str(r[L]).split(","):
                tpool[(r.user, r.aa, r.bb)].add(p.strip())

mm = CORE.fit_manner(tr, meta)

DN = np.load(CHAMP / "dino_frames.npz")


def dino_seq(path, maxT=420):
    """L2-normalised DINO frame sequence, subsampled like emolab.dtw_feats.pose_seq."""
    if path not in DN:
        return None, 1.0
    A = np.asarray(DN[path], dtype=np.float32)
    if len(A) < 8:
        return None, 1.0
    ratio = 1.0
    if len(A) > maxT:
        idx = np.linspace(0, len(A) - 1, maxT).astype(int)
        ratio = len(A) / maxT
        A = A[idx]
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    return A / n, ratio


_dtw_cache = {}


def dino_dtw_feats(pa, pb):
    """Symmetric-cache directed DTW path features between two clips' DINO sequences."""
    key = (pa, pb)
    if key in _dtw_cache:
        return _dtw_cache[key]
    A, ra = dino_seq(pa)
    B, rb = dino_seq(pb)
    if A is None or B is None:
        _dtw_cache[key] = {}
        return {}
    try:
        f = path_feats(A, B, ra, rb)
    except Exception:
        f = {}
    f = {f"dinodtw_{k}": v for k, v in f.items()}
    _dtw_cache[key] = f
    return f


X = []
Y = []
U = []
COVERED = 0
TOTAL = 0
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
        TOTAL += 1
        # directed feature computed ONCE per (clip_i, clip_j) physical pair, then held fixed
        # for both the correct and swapped orientation rows below (no label-dependent sign flip)
        dfeat = dino_dtw_feats(rows[i].path, rows[j].path)
        if dfeat:
            COVERED += 1
        for (ma, mb, y) in [(labs[i], labs[j], 1), (labs[j], labs[i], 0)]:
            f = EP.pair_row(bf[i], bf[j], i, j, k, ma, mb, mm, ctx)
            f.update(dfeat)
            X.append(f)
            Y.append(y)
        U.append(b[0])
        U.append(b[0])

X = pd.DataFrame(X)
Y = np.array(Y)
U = np.array(U)
dtw_cols = [c for c in X.columns if c.startswith("dinodtw_")]
print(f"pair rows {len(X)} ({len(X)//2} pairs)  champion cols {X.shape[1]-len(dtw_cols)}  dino-dtw cols {len(dtw_cols)}")
print(f"pairs with a usable DINO-DTW feature: {COVERED}/{TOTAL} = {COVERED/max(TOTAL,1):.3f}")


def ev(cols, label):
    Xv = X[cols].to_numpy(float)
    p = np.zeros(len(Y))
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


champ_cols = [c for c in X.columns if c not in dtw_cols]
print("\n=== pairwise manner-orientation accuracy, subject-disjoint ===")
base_acc = ev(champ_cols, "champion pairwise features [BASELINE]")
dtw_acc = ev(dtw_cols, "DINO-DTW path features ONLY")
comb_acc = ev(champ_cols + dtw_cols, "champion + DINO-DTW  [COMBINED]")

out_dir = HERE / "results_dino_dtw"
out_dir.mkdir(exist_ok=True)
pd.DataFrame(
    [
        dict(arm="champion_only", accuracy=base_acc, n_pairs=len(Y) // 2),
        dict(arm="dino_dtw_only", accuracy=dtw_acc, n_pairs=len(Y) // 2),
        dict(arm="combined", accuracy=comb_acc, n_pairs=len(Y) // 2),
    ]
).to_csv(out_dir / "orientation_accuracy_summary.csv", index=False)
print(f"\nwrote {out_dir / 'orientation_accuracy_summary.csv'}")
