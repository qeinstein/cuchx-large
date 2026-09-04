import os, sys, itertools, collections, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,'champ')
from core import load_all, opts, gt_letters, mgroup, GROUPS, PHYS, block_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

tr, te, meta = load_all()
R = pd.read_csv('emolab/rich_hau.csv')
R.columns = [c.replace('.', 'p') for c in R.columns]
num = [c for c in R.columns if c != 'unit_dir' and R[c].dtype.kind in 'fiu']
keep = [c for c in num if R[c].notna().mean() > 0.85 and R[c].nunique() > 5]
u2p = dict(zip(meta.unit_dir, meta.qa_path))
R['qa_path'] = R.unit_dir.map(u2p)
rich = {p: dict(zip(keep, row)) for p, row in zip(R.qa_path, R[keep].to_numpy(float))
        if isinstance(p, str)}
em = tr[tr.category == 'emotion'].copy(); em['blk'] = list(zip(em.user, em.aa, em.bb))
em['lab'] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
em['grp'] = [mgroup(l) for l in em.lab]
ufold = {}
for f in range(5):
    for u in pd.read_csv('splits/fold_%d_val.csv' % f).subject_id.unique(): ufold[u] = f
mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}

def rb(blk, cols):
    M = np.array([[rich.get(b, {}).get(c, np.nan) for c in cols] for b in blk], float)
    with np.errstate(all='ignore'):
        mu = np.nanmean(M, 0); sd = np.nanstd(M, 0) + 1e-9; Z = (M - mu) / sd
        Rk = np.argsort(np.argsort(np.where(np.isnan(M), -np.inf, M), 0), 0) / max(1, len(blk) - 1)
    return [{**{f'A_{c}': M[i, j] for j, c in enumerate(cols)},
             **{f'Z_{c}': Z[i, j] for j, c in enumerate(cols)},
             **{f'R_{c}': Rk[i, j] for j, c in enumerate(cols)}} for i in range(len(blk))]

def build(champ=True, rcols=None):
    X = []; y = []; u = []; bk = []; pth = []
    for b, g in em.groupby('blk'):
        g = g.sort_values('cc'); blk = [r.path for _, r in g.iterrows()]; k = len(blk)
        bf = block_features(blk, mfeat, k) if champ else [dict(pos=i, k=k, posfrac=i/max(1,k-1)) for i in range(k)]
        rf = rb(blk, rcols) if rcols else [{} for _ in blk]
        for i, (_, r) in enumerate(g.iterrows()):
            d = dict(bf[i]); d.update(rf[i]); X.append(d); y.append(r.grp); u.append(r.user)
            bk.append(b); pth.append(r.path)
    return pd.DataFrame(X), np.array(y), np.array(u), bk, pth

def ev(X, y, u, label, it=250):
    X = X.loc[:, X.notna().any()]; Xv = X.to_numpy(float)
    oof = np.empty(len(y), object); P = np.zeros((len(y), len(GROUPS)))
    for f in range(5):
        m = np.array([ufold.get(uu, -1) == f for uu in u])
        c = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.06, max_depth=4,
                                           l2_regularization=1.0, random_state=0).fit(Xv[~m], y[~m])
        oof[m] = c.predict(Xv[m]); pp = c.predict_proba(Xv[m])
        for j, cc in enumerate(c.classes_): P[m, GROUPS.index(cc)] = pp[:, j]
    a = (oof == y).mean()
    print(f'  {label:46s} {a:.4f} ({(oof==y).sum()}/{len(y)}) nf={X.shape[1]}', flush=True)
    return P, a

t0 = time.time()
Xc, y, u, bk, pth = build(True, None)
print('=== 5-way manner-group accuracy, subject-disjoint ===', flush=True)
Pb, ab = ev(Xc, y, u, 'champion PHYS features [BASELINE]')
# univariate screen of the rich features, fitted on all users (screen only, then re-check nested)
Xr_all, _, _, _, _ = build(False, keep)
Xr_all = Xr_all.loc[:, Xr_all.notna().any()]
Xf = Xr_all.fillna(Xr_all.median())
mi = mutual_info_classif(Xf.to_numpy(float), y, random_state=0)
ord_ = np.argsort(-mi); names = list(Xr_all.columns)
print('\ntop 25 rich features by mutual information with manner group:')
for i in ord_[:25]: print(f'   {names[i]:34s} {mi[i]:.4f}')
def base_of(n):
    return n[2:] if n[:2] in ('A_','Z_','R_') else None
base_rich = sorted({base_of(names[i]) for i in ord_[:200]} - {None} & set(keep))
print('\nrich base columns implied by top-200 derived features:', len(base_rich), flush=True)
for n in [20, 40, 80]:
    sel = sorted(({base_of(names[i]) for i in ord_[:n*3]} - {None}) & set(keep))
    Xa, _, _, _, _ = build(True, sel)
    ev(Xa, y, u, f'champion + rich(top~{len(sel)} base cols)')
print('%.0fs' % (time.time()-t0))
np.save('emolab/P_base.npy', Pb)
pd.DataFrame(dict(y=y, u=u, blk=[str(b) for b in bk], path=pth)).to_csv('emolab/rows.csv', index=False)
