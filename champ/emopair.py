"""ATTACK 2 - pairwise manner-assignment discriminator.

Error diagnosis that motivates this: of the 78 held-out emotion errors, 73 already have the
true manner inside the candidate set, and 62 (79%) are a pure pairwise SWAP within the
session - two clips received each other's manner.  Every confused pair shows up symmetrically
(Carefully->Calmly 3 and Calmly->Carefully 3, ...).  So absolute manner recognition is not
the failure; the relative ordering of two clips of the same session is.

Rather than scoring each (clip, manner) independently with a 5-group classifier, learn
    P( clip i gets manner a AND clip j gets manner b | clip i vs clip j, a, b, slots )
from the DIFFERENCE of the two clips' physical features.  Both clips are the same subject
performing the same script, so the differences are far cleaner than absolute values.

Manner evidence is conditioned on action identity via the session's recovered action pool:
"slowly while walking" is not physically "slowly while wiping", so the pool's locomotion /
manipulation composition is supplied as features.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import PHYS, block_features, mgroup, gt_letters, opts, GROUPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Optional: DTW warping-path features between the two clips.  Two trials of one session are the
# same script performed twice, so the optimal warping path between them is a content-free
# measurement of relative tempo, including where one trial pauses and the other does not.
# Held-out pairwise manner-orientation accuracy 0.9738 -> 0.9825 in an isolated A/B.
# 0 = champion behaviour.
W_DTW = float(os.environ.get('CHAMP_EMO_DTW', 0.0))
DTW_COLS = ['dtw_logratio', 'dtw_hv', 'dtw_dev_mean', 'dtw_dev_absmean', 'dtw_slope_med',
            'dtw_run_hv', 'dtw_res_mean', 'dtw_cost', 'dtw_slope_std', 'dtw_horiz', 'dtw_vert']
_DTW = None


def dtw_of(pa, pb):
    """Directed DTW features for the ordered clip pair (pa, pb); {} when unavailable."""
    global _DTW
    if not W_DTW:
        return {}
    if _DTW is None:
        f = os.path.join(ROOT, 'champ', 'dtw_pairs_bypath.csv')
        _DTW = {}
        if os.path.exists(f):
            d = pd.read_csv(f)
            cols = [c for c in DTW_COLS if c in d.columns]
            for r in d.itertuples():
                _DTW[(r.pa, r.pb)] = {c: getattr(r, c) for c in cols}
    return _DTW.get((pa, pb), {})

# Optional: frozen DINOv2-Depth frame-delta TRAJECTORY descriptor (research/
# dino_temporal_style_probe_20260904/probe_dino_trajectory.py).  Unlike the static dino_v_*
# summary stats already in PHYS, this differences L2-normalised frame embeddings first, so it
# removes static scene/subject appearance and keeps native-rate velocity/acceleration/rhythm/
# phase shape.
#
# MEASURED AND KILLED (2026-09-04, probe_joint_integration.py).  A standalone override policy
# trained on it was already strongly net-negative against the champion's actual disagreements
# (results/flip_audit_by_threshold.csv, worst -127, best still -8).  Added as extra columns to
# THIS classifier and fit jointly with the existing evidence -- the correct way to test whether
# it helps the joint objective, per the DTW precedent above -- held-out pairwise manner-
# orientation is 0.9738 with or without it, bit-identical in every one of the 5 folds: the
# classifier assigns it zero weight.  (An earlier run of that probe reported +1.4pp from a
# label-leakage bug -- the feature was sign-flipped in lockstep with the swap label being
# predicted -- and that number is retracted; see the probe's corrected RESULTS.)
# Kept here, default-off, only so the negative result is reproducible.  0 = champion behaviour.
W_TRAJ = float(os.environ.get('CHAMP_EMO_TRAJ', 0.0))
_TRAJ = None


def traj_of(pa, pb):
    """Signed difference/ratio of the trajectory descriptor for (pa, pb); {} when unavailable."""
    global _TRAJ
    if not W_TRAJ:
        return {}
    if _TRAJ is None:
        f = os.path.join(ROOT, 'champ', 'dino_traj_desc.npz')
        _TRAJ = np.load(f) if os.path.exists(f) else {}
    if pa not in _TRAJ or pb not in _TRAJ:
        return {}
    da, db = np.asarray(_TRAJ[pa], float), np.asarray(_TRAJ[pb], float)
    diff = da - db
    ratio = diff / (np.abs(da) + np.abs(db) + 1e-6)
    out = {}
    for n in range(len(diff)):
        out[f'traj_d_{n}'] = float(diff[n])
        out[f'traj_r_{n}'] = float(ratio[n])
    return out

# gross-motion vs fine-manipulation split of the 40-action vocabulary, used only to
# characterise a session so the speed evidence can be read in the right context
LOCOMOTION = {'Walking', 'Running', 'Squats', 'Lunges', 'Jumping jacks', 'Stretching',
              'Sitting down', 'Standing up', 'Lying down', 'Mopping', 'Sweeping'}


def pool_context(pool):
    """Two scalars describing what kind of session this is."""
    if not pool:
        return dict(pool_n=np.nan, pool_loco=np.nan)
    return dict(pool_n=len(pool),
                pool_loco=sum(1 for a in pool if a in LOCOMOTION) / len(pool))


def pair_row(bf_i, bf_j, i, j, k, ma, mb, mm, ctx, paths=None):
    """Feature row for the hypothesis 'clip i -> manner ma, clip j -> manner mb'."""
    f = {}
    if W_DTW and paths is not None and len(paths) == 2:
        for c, v in dtw_of(paths[0], paths[1]).items():
            f[c] = v
    if W_TRAJ and paths is not None and len(paths) == 2:
        for c, v in traj_of(paths[0], paths[1]).items():
            f[c] = v
    for c in PHYS:
        ai, aj = bf_i.get(f'a_{c}', np.nan), bf_j.get(f'a_{c}', np.nan)
        f[f'd_{c}'] = ai - aj
        f[f'r_{c}'] = (ai - aj) / (abs(ai) + abs(aj) + 1e-9)
        f[f'zi_{c}'] = bf_i.get(f'z_{c}', np.nan)
        f[f'zj_{c}'] = bf_j.get(f'z_{c}', np.nan)
    f['slot_i'] = i; f['slot_j'] = j; f['k'] = k; f['slot_gap'] = j - i
    ga, gb = mgroup(ma), mgroup(mb)
    f['grp_i'] = GROUPS.index(ga); f['grp_j'] = GROUPS.index(gb)
    f['same_grp'] = int(ga == gb)
    # slot priors for this hypothesis and for the swap, and their difference
    lp = np.log(max(mm['ppm'](ma, i, k), 1e-9)) + np.log(max(mm['ppm'](mb, j, k), 1e-9))
    lps = np.log(max(mm['ppm'](mb, i, k), 1e-9)) + np.log(max(mm['ppm'](ma, j, k), 1e-9))
    f['slot_ll'] = lp; f['slot_ll_swap'] = lps; f['slot_ll_margin'] = lp - lps
    f['pmg_i'] = np.log(max(mm['pmg'].get(ma, 1e-4), 1e-6))
    f['pmg_j'] = np.log(max(mm['pmg'].get(mb, 1e-4), 1e-6))
    f.update(ctx)
    return f


def fit(tr, meta, mm, pool_of=None):
    """Train the pairwise discriminator on the given (training) subjects only."""
    mfeat = mm['mfeat']
    e = tr[tr.category == 'emotion']
    X, y = [], []
    augment_subblocks = os.environ.get('CHAMP_EMO_PAIR_SUBBLOCKS', '0') == '1'
    for (u, a, b), g in e.groupby(['user', 'aa', 'bb']):
        g = g.sort_values('cc')
        rows = [r for _, r in g.iterrows()]
        if len(rows) < 2:
            continue
        variants = [rows]
        if augment_subblocks and len(rows) >= 3:
            # The real test has 21 two-clip blocks whose option intersection still exposes
            # three protocol manners.  Train the pair head on the same visible k=2 regime,
            # while retaining the full-session observations for k=3+ inference.
            variants += [list(v) for v in itertools.combinations(rows, 2)]
        for rows_v in variants:
            k = len(rows_v)
            blk = [r.path for r in rows_v]
            bf = block_features(blk, mfeat, k)
            labs = [str(r[gt_letters(r)[0]]).strip() for r in rows_v]
            ctx = pool_context((pool_of or {}).get((u, a, b)))
            for i, j in itertools.combinations(range(k), 2):
                if labs[i] == labs[j]:
                    continue
                pp = [blk[i], blk[j]]
                # correct orientation
                X.append(pair_row(bf[i], bf[j], i, j, k, labs[i], labs[j], mm, ctx, pp)); y.append(1)
                # swapped orientation
                X.append(pair_row(bf[i], bf[j], i, j, k, labs[j], labs[i], mm, ctx, pp)); y.append(0)
    Xd = pd.DataFrame(X)
    # a column that is entirely NaN (e.g. pool context when no pool is supplied) breaks
    # the histogram binner, so drop those before fitting and remember the surviving set
    Xd = Xd.loc[:, Xd.notna().any()]
    cols = list(Xd.columns)
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=5, l2_regularization=1.0,
        categorical_features=[cols.index(c) for c in
                              ('slot_i', 'slot_j', 'k', 'grp_i', 'grp_j')],
        random_state=0)
    clf.fit(Xd.to_numpy(float), y)
    return dict(clf=clf, cols=cols)


def pair_logodds(pm, bf, k, cand, mm, ctx, clip_opts, paths=None):
    """log-odds table: lo[(i, j, a, b)] for 'clip i -> cand[a], clip j -> cand[b]'."""
    rows, key = [], []
    for i, j in itertools.combinations(range(k), 2):
        for a, b in itertools.permutations(range(len(cand)), 2):
            if cand[a] not in clip_opts[i] or cand[b] not in clip_opts[j]:
                continue
            pp = [paths[i], paths[j]] if paths is not None else None
            rows.append(pair_row(bf[i], bf[j], i, j, k, cand[a], cand[b], mm, ctx, pp))
            key.append((i, j, a, b))
    if not rows:
        return {}
    P = pm['clf'].predict_proba(
        pd.DataFrame(rows).reindex(columns=pm['cols']).to_numpy(float))[:, 1]
    P = np.clip(P, 1e-4, 1 - 1e-4)
    return {kk: float(np.log(p / (1 - p))) for kk, p in zip(key, P)}
