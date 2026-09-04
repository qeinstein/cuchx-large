"""A/B on the task that actually produces the errors: pairwise manner ORIENTATION.
88% of the champion's 75 emotion errors are within-block swaps, so this is the decision
that matters.  Arm 1 = the champion's pairwise feature set.  Arm 2 = + DTW warping-path
features between the two clips.  Subject-disjoint throughout.
"""
import os, sys, itertools, collections, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'champ')
from core import (load_all, opts, gt_letters, mgroup, GROUPS, PHYS, block_features)
import emopair as EP
from sklearn.ensemble import HistGradientBoostingClassifier

tr, te, meta = load_all()
em = tr[tr.category == 'emotion'].copy()
em['blk'] = list(zip(em.user, em.aa, em.bb))
em['lab'] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
ufold = {}
for f in range(5):
    for u in pd.read_csv('splits/fold_%d_val.csv' % f).subject_id.unique(): ufold[u] = f
mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}

# true session pool (same object the champion's emopair receives)
ACT = ['single', 'multi', 'combination', 'sequence']
tpool = collections.defaultdict(set)
for _, r in tr[(tr.source == 'HAU') & tr.category.isin(ACT)].iterrows():
    for L in str(r['answer']):
        if L in 'ABCD':
            for p in str(r[L]).split(','): tpool[(r.user, r.aa, r.bb)].add(p.strip())

# slot priors, fitted once on all users (identical in both arms, so it cannot explain a delta)
import core as CORE
mm = CORE.fit_manner(tr, meta)

# DTW features
DT = pd.read_csv('emolab/dtw_pairs.csv')
DCOLS = [c for c in DT.columns if c.startswith('dtw_')]
dmap = {}
for r in DT.itertuples():
    dmap[(r.key, r.i, r.j)] = {c: getattr(r, c) for c in DCOLS}

X = []; Y = []; U = []; PAIR = []
for b, g in em.groupby('blk'):
    g = g.sort_values('cc'); rows = [r for _, r in g.iterrows()]; k = len(rows)
    if k < 2: continue
    paths = sorted(g.path.unique())
    key = f'train|{b[0]}|{b[1]}|{b[2]}|' + ','.join(paths)
    bf = block_features([r.path for r in rows], mfeat, k)
    labs = [r.lab for r in rows]
    ctx = EP.pool_context(tpool.get(b))
    for i, j in itertools.combinations(range(k), 2):
        if labs[i] == labs[j]: continue
        for (ma, mb, y, si, sj) in [(labs[i], labs[j], 1, i, j), (labs[j], labs[i], 0, i, j)]:
            f = EP.pair_row(bf[i], bf[j], i, j, k, ma, mb, mm, ctx)
            d = dmap.get((key, i, j), {}) if y == 1 else dmap.get((key, j, i), {})
            # for the swapped hypothesis, the relevant warp direction is reversed
            f.update({c: d.get(c, np.nan) for c in DCOLS})
            X.append(f); Y.append(y); U.append(b[0]); PAIR.append((str(b), i, j))
X = pd.DataFrame(X); Y = np.array(Y); U = np.array(U)
print('pair rows %d (%d pairs)  champion cols %d  dtw cols %d' % (
    len(X), len(X) // 2, X.shape[1] - len(DCOLS), len(DCOLS)))

def ev(cols, label):
    Xv = X[cols].to_numpy(float)
    p = np.zeros(len(Y))
    for f in range(5):
        m = np.array([ufold.get(u, -1) == f for u in U])
        c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5,
                                           l2_regularization=1.0, random_state=0).fit(Xv[~m], Y[~m])
        p[m] = c.predict_proba(Xv[m])[:, 1]
    # per-pair decision: correct orientation must outscore the swap
    ok = 0; n = 0
    for t in range(0, len(Y), 2):
        n += 1; ok += int(p[t] > p[t + 1])
    print(f'  {label:44s} orientation acc {ok/n:.4f}  ({ok}/{n})  nfeat={len(cols)}')
    return p, ok / n

champ_cols = [c for c in X.columns if c not in DCOLS]
print('\n=== pairwise manner-orientation accuracy, subject-disjoint ===')
ev(champ_cols, 'champion pairwise features [BASELINE]')
ev(DCOLS, 'DTW warping-path features ONLY')
ev(champ_cols + DCOLS, 'champion + DTW  [COMBINED]')
dsub = [c for c in DCOLS if c in ('dtw_logratio','dtw_hv','dtw_dev_mean','dtw_dev_absmean',
                                  'dtw_slope_med','dtw_run_hv','dtw_res_mean','dtw_cost',
                                  'dtw_slope_std','dtw_horiz','dtw_vert')]
ev(champ_cols + dsub, 'champion + DTW(compact 11)')
