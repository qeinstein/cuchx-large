"""MECHANISM Y, with nested subject-disjoint weight selection and a proper flip audit.

`run17.py` established that nested HARn children give usable order observations once the
harvesting works (385 observations over 304 parent clips, child action correct 366/385 =
0.951), and that adding them to mechanism S's joint decoder helps.  But it reported the best
score over a weight grid evaluated on the same folds it selected on, which is exactly the
overfitting this project audits for elsewhere.  This re-runs it with the weight chosen inside
each fold, on the other four folds only, so the reported number is honest.

Also reports the decision-level audit the project requires before anything ships: flips
against mechanism S (the version actually in the 0.93859 champion), W->R, R->W, flip
precision, net, and per-fold consistency.
"""
import os, sys, itertools, collections, json, pickle
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'champ'))
from seqcore import *
from fast import Block, PERMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = pickle.load(open(os.path.join(ROOT, 'seqlab', 'res11.pkl'), 'rb'))
EV, KS = R['EV'], R['KS']
cache = pickle.load(open(os.path.join(ROOT, 'seqlab', 'cache.pkl'), 'rb'))
tr, te, meta = load_all()
au = pd.read_csv(os.path.join(ROOT, 'champ', 'audit_champ.csv')).set_index('qa_id')

hau = tr[tr.source == 'HAU'].copy()
hau['blk'] = list(zip(hau.user, hau.aa, hau.bb))
clips_of = {b: sorted(g.path.unique(), key=lambda p: p.split('-')[-1])
            for b, g in hau.groupby('blk')}
byblk = collections.defaultdict(list)
for k, v in cache.items():
    byblk[v['blk']].append(k)
BL = {}
for blk, ks in byblk.items():
    ks = sorted(ks); qo = [cache[k]['opts'] for k in ks]
    BL[blk] = Block(sorted(set().union(*[set(o) for o in qo])), qo)
fold = {}
for f in range(5):
    for q in pd.read_csv(os.path.join(ROOT, 'splits', f'fold_{f}_val.csv')).qa_id:
        fold[q] = f

# ------------------------------------------------------------------ observations
mi = meta.set_index('qa_path')
hl = [(r.qa_path, r.t0, r.t1, r.f0, r.f1) for r in meta[meta.kind == 'train_hau'].itertuples()
      if np.isfinite(r.t0)]
nest = {}
for r in meta[meta.kind == 'train_harn'].itertuples():
    if not np.isfinite(r.t0):
        continue
    c = [h for h in hl if h[1] <= r.t0 + 0.5 and h[2] >= r.t1 - 0.5
         and h[3] <= r.f0 and h[4] >= r.f1]
    if len(c) == 1:
        nest[r.qa_path] = c[0][0]

VOC = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
hs = tr[(tr.source == 'HARn') & (tr.category == 'single')]
p2f = {}
for r in hs.itertuples():
    L = [x for x in str(r.answer) if x in 'ABCD']
    if L:
        p2f.setdefault(str(getattr(r, L[0])).strip(), collections.Counter())[r.harn_action] += 1
p2f = {k: v.most_common(1)[0][0] for k, v in p2f.items()}

obs = collections.defaultdict(list)
nok = nbad = 0
for r in hs.itertuples():
    par = nest.get(r.path)
    if par is None or r.qa_id not in au.index:
        continue
    pv = au.loc[r.qa_id, 'pred']
    if pv not in 'ABCD':
        continue
    act = VOC.get(p2f.get(str(getattr(r, pv)).strip()))
    if act is None or act not in OPT2CLS:
        continue
    p, c = mi.loc[par], mi.loc[r.path]
    span = p.f1 - p.f0
    if not np.isfinite(span) or span <= 0:
        continue
    obs[par].append((float((c.f0 - p.f0) / span), act))
    nok += int(au.loc[r.qa_id, 'correct'] == 1)
    nbad += int(au.loc[r.qa_id, 'correct'] == 0)
print(f'observations: {sum(len(v) for v in obs.values())} over {len(obs)} parent clips '
      f'(child action correct {nok}/{nok+nbad} = {nok/max(nok+nbad,1):.3f})')


def child_pairs(blk, w=1.0):
    pts = [pt for p in clips_of.get(blk, []) for pt in obs.get(p, [])]
    out = collections.Counter()
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j or pts[i][1] == pts[j][1]:
                continue
            if pts[i][0] < pts[j][0]:
                out[(pts[i][1], pts[j][1])] += 1
    Rm = {}
    for (x, y) in list(out):
        n1, n2 = out[(x, y)], out.get((y, x), 0)
        if n1 + n2 == 0:
            continue
        pr = (n1 + 0.5) / (n1 + n2 + 1.0)
        Rm[(x, y)] = w * float(np.log(pr / (1 - pr)))
    return Rm


W = (4.0, 4.0, 1.0)
GRID = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
_cp = {blk: child_pairs(blk) for blk in KS}


def run(wy):
    preds = {}
    for blk, ks in KS.items():
        e = EV[blk]
        pm = [(W[0], e['h05']), (W[1], e['h10']), (W[2], e['learn'])]
        if wy:
            pm.append((wy, _cp[blk]))
        labs, _, _ = BL[blk].best(pm, [(0.0, np.zeros(24)) for _ in ks])
        for k, l in zip(ks, labs):
            preds[k] = l
    return preds


P = {wy: run(wy) for wy in GRID}
ans = {k: cache[k]['ans'] for k in cache}


def score(pred, keys):
    return sum(pred[k] == ans[k] for k in keys)


keys_by_fold = collections.defaultdict(list)
for k in cache:
    keys_by_fold[fold.get(k, -1)].append(k)

print('\n=== full-grid scores (SELECTION-CONTAMINATED, for reference only) ===')
for wy in GRID:
    print(f'  w_y={wy:<5} {score(P[wy], cache)}/305  '
          f'per-fold {{{", ".join(f"{f}: {score(P[wy], ks)}" for f, ks in sorted(keys_by_fold.items()))}}}')

print('\n=== NESTED selection: weight chosen on the other four folds ===')
nested, chosen = {}, {}
for f, ks in sorted(keys_by_fold.items()):
    other = [k for g, kk in keys_by_fold.items() if g != f for k in kk]
    best = max(GRID, key=lambda wy: (score(P[wy], other), -wy))
    chosen[f] = best
    for k in ks:
        nested[k] = P[best][k]
    print(f'  fold {f}: chose w_y={best:<5} -> held-out {score(P[best], ks)}/{len(ks)}')
base = P[0.0]
print(f'\n  mechanism S alone      {score(base, cache)}/305 = {score(base, cache)/305:.4f}')
print(f'  S + Y, nested weights  {sum(nested[k]==ans[k] for k in cache)}/305 = '
      f'{sum(nested[k]==ans[k] for k in cache)/305:.4f}')

print('\n=== decision-level audit, nested S+Y vs mechanism S ===')
rows = []
for k in cache:
    rows.append(dict(qa=k, fold=fold.get(k, -1), ans=ans[k], base=base[k], new=nested[k],
                     bc=base[k] == ans[k], nc=nested[k] == ans[k],
                     ncov=len([pt for p in clips_of.get(cache[k]['blk'], [])
                               for pt in obs.get(p, [])])))
A = pd.DataFrame(rows)
fl = A[A.base != A.new]
wr = int(((~fl.bc) & fl.nc).sum()); rw = int((fl.bc & (~fl.nc)).sum())
print(f'  flips {len(fl)}  W->R {wr}  R->W {rw}  W->W {len(fl)-wr-rw}  '
      f'precision {wr/max(wr+rw,1):.3f}  net {wr-rw:+d}')
print('  per-fold net: ' + str({int(f): int(((~g.bc) & g.nc).sum() - (g.bc & (~g.nc)).sum())
                                for f, g in fl.groupby('fold')}))
print('\n  by observation coverage of the block:')
A['cov'] = pd.cut(A.ncov, [-1, 0, 1, 3, 100], labels=['0', '1', '2-3', '4+'])
print(A.groupby('cov', observed=True).agg(n=('qa', 'size'), S=('bc', 'sum'),
                                          SY=('nc', 'sum')).to_string())
A.to_csv(os.path.join(ROOT, 'seqlab', 'audit_run18.csv'), index=False)
json.dump(dict(S=score(base, cache), SY_nested=int(sum(nested[k] == ans[k] for k in cache)),
               chosen={int(k): v for k, v in chosen.items()},
               flips=len(fl), wr=wr, rw=rw, precision=wr / max(wr + rw, 1), net=wr - rw),
          open(os.path.join(ROOT, 'seqlab', 'run18_summary.json'), 'w'), indent=2)
print('\nwrote seqlab/audit_run18.csv and seqlab/run18_summary.json')
