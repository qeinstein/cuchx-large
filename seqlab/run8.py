import os, sys, itertools, collections, pickle, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from prior import load_sources, OrderPrior
from fast import Block, PERMS, PIDX

tr, te, meta = load_all()
cache = pickle.load(open('seqlab/cache.pkl', 'rb'))
byblk = collections.defaultdict(list)
for k, v in cache.items(): byblk[v['blk']].append(k)
hau = tr[tr.source == 'HAU'].copy(); hau['blk'] = list(zip(hau.user, hau.aa, hau.bb))
clips_of = {b: sorted(g.path.unique(), key=lambda p: p.split('-')[-1]) for b, g in hau.groupby('blk')}
fold = {}
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q] = f

def pairlo(lp, acts, tau=1.0, conf=False):
    ci = [OPT2CLS[a] for a in acts]
    sub = lp[ci]
    q = np.exp((sub - sub.max(1, keepdims=True)) / tau)
    q = q / (q.sum(1, keepdims=True) + 1e-12)
    C = np.cumsum(q, axis=1)
    n = len(acts); raw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: raw[i, j] = (q[j][1:] * C[i][:-1]).sum()
    # per-action presence confidence: max posterior over time of that class
    P = np.exp(lp); tot = P.sum(0) + 1e-12
    cf = np.array([float((P[c] / tot).max()) for c in ci]) if conf else np.ones(n)
    R = {}
    for i in range(n):
        for j in range(i + 1, n):
            p = raw[i, j] / (raw[i, j] + raw[j, i] + 1e-12)
            p = float(np.clip(p, 1e-3, 1 - 1e-3))
            lo = np.log(p / (1 - p)) * float(np.sqrt(cf[i] * cf[j]))
            R[(acts[i], acts[j])] = lo; R[(acts[j], acts[i])] = -lo
    return R

TAUS = [0.5, 1.0, 2.0]
BL, KS, XC = {}, {}, {}
for blk, ks in byblk.items():
    ks = sorted(ks); KS[blk] = ks
    qopts = [cache[k]['opts'] for k in ks]
    U = sorted(set().union(*[set(o) for o in qopts]))
    BL[blk] = Block(U, qopts)
    Ua = [a for a in U if a in OPT2CLS]
    for tau in TAUS:
        for cf in [False, True]:
            acc = collections.Counter(); ncl = 0
            for p in clips_of.get(blk, []):
                lp = DC.logits('oof', p)
                if lp is None: continue
                ncl += 1
                for kk, v in pairlo(lp, Ua, tau, cf).items(): acc[kk] += v
            XC[(blk, tau, cf)] = {k: v / max(ncl, 1) for k, v in acc.items()}
S, seg, qa = load_sources(); OP = OrderPrior(S, seg, qa)
PR = {}
for u in set(b[0] for b in byblk):
    for wq in [0.0, 1.0]:
        PR[(u, wq)] = OP.build(u, w_qa=wq, gauss=False)

def run(w_dp, w_xc, w_pr, tau=1.0, cf=False, wq=1.0):
    preds = {}
    for blk, ks in KS.items():
        labs, _, _ = BL[blk].best([(w_pr, PR[(blk[0], wq)]), (w_xc, XC[(blk, tau, cf)])],
                                  [(w_dp, cache[k]['dp']) for k in ks])
        for k, l in zip(ks, labs): preds[k] = l
    return preds
def acc_by_fold(p):
    r = collections.Counter(); n = collections.Counter()
    for k, v in cache.items():
        f = fold.get(k, -1); n[f] += 1; r[f] += (p[k] == v['ans'])
    return r, n

GRID = [(wd, wx, wp, tau, cf, wq)
        for wd in [0.0, 0.5, 1.0]
        for wx in [1, 2, 4, 8, 16]
        for wp in [0.5, 1, 2, 4]
        for tau in TAUS for cf in [False, True] for wq in [1.0]]
print('grid size', len(GRID))
t0 = time.time()
RES = {}
for g in GRID:
    p = run(*g)
    r, n = acc_by_fold(p)
    RES[g] = (sum(r.values()), {f: r[f] for f in sorted(n)}, {f: n[f] for f in sorted(n)})
print('evaluated in %.1fs' % (time.time()-t0))
tot = sorted(RES.items(), key=lambda x: -x[1][0])
print('\ntop 12 configs by pooled OOF:')
for g, v in tot[:12]:
    print(f'  wd={g[0]} wx={g[1]:<3} wp={g[2]:<4} tau={g[3]} conf={int(g[4])} -> {v[0]}/305  per-fold {v[1]}')

print('\n=== HONEST nested selection: pick weights on the other 4 folds, score the held-out fold ===')
tot_ok = 0; totn = 0; chosen = {}
for f in range(5):
    best, bs = None, -1
    for g, v in RES.items():
        o = v[0] - v[1][f]
        if o > bs: bs, best = o, g
    tot_ok += RES[best][1][f]; totn += RES[best][2][f]; chosen[f] = best
    print(f'  fold {f}: chose wd={best[0]} wx={best[1]} wp={best[2]} tau={best[3]} cf={int(best[4])}'
          f'  -> held-out {RES[best][1][f]}/{RES[best][2][f]}')
print(f'  NESTED-CV TOTAL {tot_ok}/{totn} = {tot_ok/totn:.4f}   (champion 172/305 = 0.5639)')
pickle.dump(dict(RES=RES, chosen=chosen), open('seqlab/res8.pkl','wb'))
