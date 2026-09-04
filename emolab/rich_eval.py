import os, sys, itertools, collections, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'champ')
from core import load_all, opts, gt_letters, mgroup, GROUPS, PHYS, block_features
from sklearn.ensemble import HistGradientBoostingClassifier

tr, te, meta = load_all()
R = pd.read_csv('emolab/rich_hau.csv')
num = [c for c in R.columns if c != 'unit_dir' and R[c].dtype.kind in 'fiu']
# keep features present for >=80% of units
keep = [c for c in num if R[c].notna().mean() > 0.8 and R[c].nunique() > 3]
print('rich features kept: %d of %d' % (len(keep), len(num)))
R = R[["unit_dir"] + keep].rename(columns=lambda c: c.replace(".","p"))
keep = [c.replace(".","p") for c in keep]
u2p = dict(zip(meta.unit_dir, meta.qa_path))
R['qa_path'] = R.unit_dir.map(u2p)
rich = {p: {c: v for c, v in zip(keep, row)} for p, row in zip(R.qa_path, R[keep].to_numpy(float)) if isinstance(p, str)}

em = tr[tr.category == 'emotion'].copy(); em['blk'] = list(zip(em.user, em.aa, em.bb))
em['lab'] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
em['grp'] = [mgroup(l) for l in em.lab]
ufold = {}
for f in range(5):
    for u in pd.read_csv('splits/fold_%d_val.csv' % f).subject_id.unique(): ufold[u] = f
mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}

def rich_block(blk, cols):
    M = np.array([[rich.get(b, {}).get(c, np.nan) for c in cols] for b in blk], float)
    with np.errstate(all='ignore'):
        mu = np.nanmean(M, 0); sd = np.nanstd(M, 0) + 1e-9
        Z = (M - mu) / sd
        Rk = np.argsort(np.argsort(np.where(np.isnan(M), -np.inf, M), 0), 0) / max(1, len(blk) - 1)
    return [{**{f'A_{c}': M[i, j] for j, c in enumerate(cols)},
             **{f'Z_{c}': Z[i, j] for j, c in enumerate(cols)},
             **{f'R_{c}': Rk[i, j] for j, c in enumerate(cols)}} for i in range(len(blk))]

def build(use_champ=True, use_rich=True, rich_cols=None):
    rc = rich_cols if rich_cols is not None else keep
    X = []; y = []; u = []; bk = []
    for b, g in em.groupby('blk'):
        g = g.sort_values('cc'); blk = [r.path for _, r in g.iterrows()]; k = len(blk)
        bf = block_features(blk, mfeat, k) if use_champ else [{} for _ in blk]
        rf = rich_block(blk, rc) if use_rich else [{} for _ in blk]
        for i, (_, r) in enumerate(g.iterrows()):
            d = dict(bf[i]); d.update(rf[i])
            if not use_champ: d.update(pos=i, k=k, posfrac=i / max(1, k - 1))
            X.append(d); y.append(r.grp); u.append(r.user); bk.append(b)
    return pd.DataFrame(X), np.array(y), np.array(u), bk

def ev(X, y, u, label, ret=False):
    X = X.loc[:, X.notna().any()]
    Xv = X.to_numpy(float)
    oof = np.empty(len(y), object); P = np.zeros((len(y), len(GROUPS)))
    for f in range(5):
        m = np.array([ufold.get(uu, -1) == f for uu in u])
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(Xv[~m], y[~m]); oof[m] = clf.predict(Xv[m])
        pp = clf.predict_proba(Xv[m])
        for j, c in enumerate(clf.classes_): P[m, GROUPS.index(c)] = pp[:, j]
    acc = (oof == y).mean()
    print(f'  {label:50s} {acc:.4f}  ({(oof==y).sum()}/{len(y)})  nfeat={X.shape[1]}')
    return (oof, P, acc, X.shape[1]) if ret else acc

print('\n=== 5-way manner-GROUP accuracy, subject-disjoint ===')
Xc, y, u, bk = build(True, False); ev(Xc, y, u, 'champion PHYS block features (baseline)')
Xr, _, _, _ = build(False, True); ev(Xr, y, u, 'rich radar+IMU only (+position)')
Xa, _, _, _ = build(True, True); o, P, acc, nf = ev(Xa, y, u, 'champion + rich  [COMBINED]', ret=True)
np.save('emolab/P_rich.npy', P)
pd.DataFrame(dict(y=y, u=u, blk=[str(b) for b in bk])).to_csv('emolab/rows.csv', index=False)
# groupwise
Xrr = [c for c in keep if c.startswith('rr_')]
Xii = [c for c in keep if c.startswith('iu_')]
Xr1, _, _, _ = build(True, True, Xrr); ev(Xr1, y, u, 'champion + radar only')
Xr2, _, _, _ = build(True, True, Xii); ev(Xr2, y, u, 'champion + IMU only')
print('\n  (radar cols %d, imu cols %d)' % (len(Xrr), len(Xii)))
