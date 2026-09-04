import os, sys, itertools, collections, pickle, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from prior import load_sources, OrderPrior
from pairclf import clip_stats, feat, OPT2CLS as O2C
from fast import Block, PERMS, PIDX
from sklearn.ensemble import HistGradientBoostingClassifier

tr, te, meta = load_all()
cache = pickle.load(open('seqlab/cache.pkl', 'rb'))
byblk = collections.defaultdict(list); 
for k, v in cache.items(): byblk[v['blk']].append(k)
hau = tr[tr.source == 'HAU'].copy(); hau['blk'] = list(zip(hau.user, hau.aa, hau.bb))
clips_of = {b: sorted(g.path.unique(), key=lambda p: p.split('-')[-1]) for b, g in hau.groupby('blk')}
path_user = dict(zip(hau.path, hau.user))
S = pd.read_csv('seqlab/onsets.csv')
Ssrc, seg, qa = load_sources(); OP = OrderPrior(Ssrc, seg, qa)
fold = {}
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q] = f
ufold = {}
for f in range(5):
    for u in pd.read_csv(f'splits/fold_{f}_val.csv').subject_id.unique(): ufold[u] = f
print('users per fold', collections.Counter(ufold.values()))

# ---- cache clip stats
t0 = time.time()
ST = {}
for p in hau.path.unique():
    lp = DC.logits('oof', p)
    if lp is not None: ST[p] = clip_stats(lp)
print('clip stats %d in %.1fs' % (len(ST), time.time()-t0))

# ---- training pairs from HARn segment onsets
segord = collections.defaultdict(list)
for r in S.itertuples(): segord[(r.user, r.trial)].append((r.on, r.act))
TRAIN = []
for (u, t), v in segord.items():
    p = f'HAU/{u}/{t}'
    if p not in ST: continue
    v = sorted(v)
    for i in range(len(v)):
        for j in range(len(v)):
            if i == j or v[i][1] == v[j][1] or v[i][0] == v[j][0]: continue
            ax, ay = v[i][1], v[j][1]
            if ax not in O2C or ay not in O2C: continue
            TRAIN.append((u, p, ax, ay, int(v[i][0] < v[j][0])))
print('training ordered pairs from segments:', len(TRAIN))

ONMU_ALL = {a: float(g.on.mean()) for a, g in S.groupby('act')}
def build_rows(items, prlo, onmu):
    X = []; y = []
    for (u, p, ax, ay, lab) in items:
        X.append(feat(ST[p], ax, ay, prlo, onmu)); y.append(lab)
    return pd.DataFrame(X), np.array(y)

MODELS = {}
for f in range(5):
    tru = [it for it in TRAIN if ufold.get(it[0], -1) != f]
    # prior/onset stats from training users only
    hold = [u for u, ff in ufold.items() if ff == f]
    Str = S[~S.user.isin(hold)]
    onmu = {a: float(g.on.mean()) for a, g in Str.groupby('act')}
    prlo = {}
    keys = set(tuple(sorted(k)) for k in list(seg) + list(qa))
    for (x, yy) in keys:
        nxy = sum(v for uu, v in seg.get((x, yy), {}).items() if uu not in hold) \
            + sum(v for uu, v in qa.get((x, yy), {}).items() if uu not in hold)
        nyx = sum(v for uu, v in seg.get((yy, x), {}).items() if uu not in hold) \
            + sum(v for uu, v in qa.get((yy, x), {}).items() if uu not in hold)
        n = nxy + nyx
        if n == 0: continue
        pp = (nxy + 1) / (n + 2); lo = float(np.log(pp / (1 - pp)))
        prlo[(x, yy)] = lo; prlo[(yy, x)] = -lo
    Xd, yv = build_rows(tru, prlo, onmu)
    cols = list(Xd.columns)
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=6,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xd.to_numpy(float), yv)
    MODELS[f] = dict(clf=clf, cols=cols, prlo=prlo, onmu=onmu)
    print(f'  fold {f}: fitted on {len(tru)} pairs ({len(hold)} users held out)')
pickle.dump({f: dict(prlo=MODELS[f]['prlo'], onmu=MODELS[f]['onmu']) for f in MODELS},
            open('seqlab/fold_priors.pkl','wb'))

# ---- sanity: held-out pairwise accuracy of the learned model
print('\n=== learned pairwise-order model, held-out accuracy on segment pairs ===')
allok=[];alln=[]
for f in range(5):
    hold=[u for u,ff in ufold.items() if ff==f]
    te_it=[it for it in TRAIN if it[0] in hold]
    if not te_it: continue
    M=MODELS[f]; Xd,yv=build_rows(te_it,M['prlo'],M['onmu'])
    pp=M['clf'].predict_proba(Xd.reindex(columns=M['cols']).to_numpy(float))[:,1]
    ok=((pp>0.5)==(yv==1)).mean(); allok.append(((pp>0.5)==(yv==1)).sum()); alln.append(len(yv))
    print(f'  fold {f}: n={len(yv):5d} acc={ok:.4f}')
print(f'  pooled: {sum(allok)}/{sum(alln)} = {sum(allok)/sum(alln):.4f}')
pickle.dump(MODELS, open('seqlab/pairmodels.pkl','wb'))
