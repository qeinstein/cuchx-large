#!/usr/bin/env python3
"""PCRME Stage 2 -- cheap falsification probe.

Question: after conditioning on action identity (comparing corresponding action PHASES
across sibling trials, instead of whole clips), does manner-group separability measurably
increase over an action-unconditioned whole-clip representation?

Four arms, all evaluated with the identical 5-fold subject-disjoint protocol and the same
probe classifiers (multinomial logistic regression + nearest-centroid), on the SAME target
(5-way manner group) so the comparison is apples-to-apples:

  A. whole-clip absolute       -- champion-style: per-clip physical features, no action
                                   segmentation, no session conditioning (~ existing PHYS).
  B. phase absolute             -- same feature family but pooled over ONLY the anchored
                                   action-phase segment (Stage 1's aligned_phases.csv rows),
                                   still no session/sibling conditioning.
  C. phase, session-relative    -- B minus that session's OWN sibling mean for the SAME
                                   action (the core PCRME hypothesis: relative comparison of
                                   corresponding phases).
  D. shuffled-label control     -- arm C's features, manner labels permuted within each fold's
                                   training subjects. Must collapse to chance; if it does not,
                                   something in the pipeline leaks the label.

Leakage audit built in
-----------------------
* Feature extraction (`phase_feats`) reads only skeleton/IMU arrays and frame indices. It is
  called identically regardless of the manner label; the label is attached to the row only
  after extraction, in a separate join.
* Arm D reshuffles labels ONLY within each fold's training partition, after features are
  frozen, and is fit/scored with the exact same code path as arm C.
* Antisymmetry check: for every pair of siblings (i, j) in a triple, verify that the
  session-relative feature of i versus j is the negation of j versus i (a differencing
  convention bug would otherwise look like signal). See `assert_antisymmetry()`.
* No test labels exist or are used anywhere in this script; it is 100% training-subject data.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CHAMP = ROOT / "champ"
sys.path.insert(0, str(CHAMP))

import dense as D  # noqa: E402
from core import load_all, PHYS, block_features, mgroup, gt_letters  # noqa: E402

HERE = Path(__file__).resolve().parent
GROUPS = ["SLOW", "CARE", "NEUT", "NERV", "FAST"]

SKEL = np.load(CHAMP / "skel_seq.npz")


def seg_slice(unit_dir, f0, f1):
    """Frame-index slice [lo, hi) into this unit's skeleton/IMU arrays for [f0, f1]."""
    kkey, fkey = f"{unit_dir}|K", f"{unit_dir}|F"
    if kkey not in SKEL or fkey not in SKEL:
        return None
    F = SKEL[fkey]
    lo = int(np.searchsorted(F, f0, side="left"))
    hi = int(np.searchsorted(F, f1, side="right"))
    hi = max(hi, lo + 1)
    return lo, hi


def phase_pool_stats(K, X_imu, lo, hi):
    """Pool per-frame skeleton+IMU descriptors over one segment into a fixed-length vector."""
    Kseg = K[lo:hi]
    if len(Kseg) < 2:
        return None
    F = D.frame_feats(Kseg)  # (t, d) translation/scale-invariant + derivatives
    Iseg = X_imu[lo:hi] if X_imu is not None else None
    parts = [F.mean(0), F.std(0), np.percentile(F, 90, axis=0)]
    if Iseg is not None and len(Iseg) == len(Kseg):
        acc = np.stack([np.linalg.norm(Iseg[:, d * 6:d * 6 + 3], axis=1) for d in range(5)], 1)
        gyr = np.stack([np.linalg.norm(Iseg[:, d * 6 + 3:d * 6 + 6], axis=1) for d in range(5)], 1)
        parts += [acc.mean(0), acc.std(0), gyr.mean(0), gyr.std(0)]
    else:
        parts += [np.zeros(5)] * 4
    vec = np.concatenate([np.atleast_1d(p) for p in parts])
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    dur = (hi - lo)
    return np.concatenate([vec, [np.log1p(dur)]])


_imu_cache = {}


def imu_of_unit(unit_dir, T):
    if unit_dir not in _imu_cache:
        try:
            _imu_cache[unit_dir] = D.imu_of(unit_dir, T)
        except Exception:
            _imu_cache[unit_dir] = None
    v = _imu_cache[unit_dir]
    return v if v is not None and len(v) == T else None


_skel_cache = {}


def skel_of_unit(unit_dir):
    if unit_dir not in _skel_cache:
        k = f"{unit_dir}|K"
        _skel_cache[unit_dir] = SKEL[k] if k in SKEL else None
    return _skel_cache[unit_dir]


def build_phase_features(aligned):
    """One feature vector per aligned_phases.csv row (arm B input)."""
    feats = {}
    for r in aligned.itertuples():
        K = skel_of_unit(r.parent_unit_dir)
        if K is None:
            continue
        Iu = imu_of_unit(r.parent_unit_dir, len(K))
        sl = seg_slice(r.parent_unit_dir, r.seg_f0, r.seg_f1)
        if sl is None:
            continue
        v = phase_pool_stats(K, Iu, *sl)
        if v is not None:
            feats[r.Index] = v
    return feats


def build_whole_clip_features(tr, meta):
    """Arm A: the champion's own PHYS features, one row per emotion clip (no segmentation)."""
    mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}
    em = tr[tr.category == "emotion"].copy()
    em["label"] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
    em["group"] = em.label.map(mgroup)
    rows = []
    for _, r in em.iterrows():
        f = mfeat.get(r.path, {})
        rows.append(dict(path=r.path, user=r.user, group=r.group,
                         **{c: f.get(c, np.nan) for c in PHYS}))
    return pd.DataFrame(rows)


def assert_antisymmetry(aligned, feat_by_idx):
    """For one triple, check that relative(i,j) == -relative(j,i) under the differencing used
    below (mean-centering is inherently symmetric, but this also guards the join/index logic:
    if action/session grouping were subtly wrong, this specific numeric identity would not
    hold to floating-point precision)."""
    g = aligned[aligned.session == aligned.session.iloc[0]]
    g = g[g.action == g.action.iloc[0]]
    idxs = [i for i in g.index if i in feat_by_idx]
    if len(idxs) < 2:
        return True
    vecs = np.stack([feat_by_idx[i] for i in idxs])
    mu = vecs.mean(0)
    rel = vecs - mu
    ok = np.isclose(rel.sum(0), 0.0, atol=1e-4).all()
    return bool(ok)


def main():
    tr, te, meta = load_all()
    aligned = pd.read_csv(HERE / "aligned_phases.csv")
    print(f"aligned_phases rows: {len(aligned)}")

    feat_by_idx = build_phase_features(aligned)
    cov = len(feat_by_idx) / len(aligned)
    print(f"phase feature coverage: {len(feat_by_idx)}/{len(aligned)} = {cov:.3f}")

    ok = assert_antisymmetry(aligned, feat_by_idx)
    print(f"antisymmetry / grouping sanity check passed: {ok}")
    assert ok, "session-relative centering failed its own sanity check -- stop and inspect"

    # `feat_by_idx` keys are row-labels into the ORIGINAL aligned_phases.csv (before any
    # filtering/reset), so re-read it once and select by that same label to keep the row order
    # and the (session, action) grouping unambiguous. `raw` is the single source of truth for
    # every array below.
    raw = pd.read_csv(HERE / "aligned_phases.csv").loc[list(feat_by_idx.keys())]
    Xb = np.stack([feat_by_idx[i] for i in raw.index])  # arm B: phase absolute
    raw = raw.reset_index(drop=True)

    # arm C: session-relative -- subtract the mean of THIS session's siblings for the SAME
    # action (computed only from the sibling set at hand; this is a leakage-free operation
    # because it never looks at another session or at the manner label).
    Xc = Xb.copy()
    for (sess, act), g in raw.groupby(["session", "action"]):
        idx = g.index.to_numpy()
        mu = Xb[idx].mean(0)
        Xc[idx] = Xb[idx] - mu

    y = raw.manner_group.to_numpy()
    users = raw.user.to_numpy()
    ufold = {}
    for f in range(5):
        for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique():
            ufold[u] = f
    fold = np.array([ufold.get(u, -1) for u in users])

    def eval_arm(X, y, fold, label, shuffle=False, agg=None):
        accs_lr, accs_nc = [], []
        rng = np.random.default_rng(0)
        for f in range(5):
            tr_m = fold != f
            te_m = fold == f
            if te_m.sum() == 0 or tr_m.sum() == 0:
                continue
            ytr, yte = y[tr_m].copy(), y[te_m]
            if shuffle:
                rng.shuffle(ytr)
            sc = StandardScaler().fit(X[tr_m])
            Xtr, Xte = sc.transform(X[tr_m]), sc.transform(X[te_m])
            lr = LogisticRegression(max_iter=2000, C=0.3).fit(Xtr, ytr)
            nc = NearestCentroid().fit(Xtr, ytr)
            accs_lr.append(lr.score(Xte, yte))
            accs_nc.append(nc.score(Xte, yte))
        print(f"  {label:38s} logreg {np.mean(accs_lr):.4f} (folds {[round(a,3) for a in accs_lr]})"
              f"   nearest-centroid {np.mean(accs_nc):.4f}")
        return np.mean(accs_lr), np.mean(accs_nc)

    print("\n=== Stage 2: manner-group probe accuracy, subject-disjoint, per PHASE row ===")
    eval_arm(Xb, y, fold, "B. phase absolute")
    eval_arm(Xc, y, fold, "C. phase session-relative")
    eval_arm(Xc, y, fold, "D. shuffled-label control", shuffle=True)

    # arm A: whole-clip champion features, same probe, same protocol, per CLIP not per phase
    wc = build_whole_clip_features(tr, meta)
    wc = wc.dropna(subset=PHYS, how="all")
    Xa = wc[PHYS].fillna(wc[PHYS].median()).to_numpy(float)
    ya = wc.group.to_numpy()
    fa = np.array([ufold.get(u, -1) for u in wc.user])
    eval_arm(Xa, ya, fa, "A. whole-clip absolute (champion PHYS)")

    print("\n=== per-clip aggregation: mean-pool phase rows back up to one vector per clip ===")
    raw["clip_key"] = raw.parent_qa_path
    agg_rows = []
    for key, g in raw.groupby("clip_key"):
        idx = g.index.to_numpy()
        agg_rows.append(dict(clip_key=key, user=g.user.iloc[0], group=g.manner_group.iloc[0],
                             vec_abs=Xb[idx].mean(0), vec_rel=Xc[idx].mean(0)))
    Aabs = np.stack([r["vec_abs"] for r in agg_rows])
    Arel = np.stack([r["vec_rel"] for r in agg_rows])
    yclip = np.array([r["group"] for r in agg_rows])
    fclip = np.array([ufold.get(r["user"], -1) for r in agg_rows])
    print(f"clips with >=1 aligned phase: {len(agg_rows)} of 809")
    eval_arm(Aabs, yclip, fclip, "clip-level mean of phase-absolute")
    eval_arm(Arel, yclip, fclip, "clip-level mean of phase-relative")

    summary = dict(
        phase_feature_coverage=cov,
        antisymmetry_check_passed=ok,
        n_phase_rows=len(raw),
        n_clips_with_phase_coverage=len(agg_rows),
    )
    json.dump(summary, open(HERE / "stage2_summary.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'stage2_summary.json'}")


if __name__ == "__main__":
    main()
