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

def pairlo_over(lp, acts, tau=1.0):
    ci = [OPT2CLS[a] for a in acts]
    sub = lp[ci]
    q = np.exp((sub - sub.max(1, keepdims=True)) / tau)
    q = q / (q.sum(1, keepdims=True) + 1e-12)
    C = np.cumsum(q, axis=1)
    n = len(acts); raw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: raw[i, j] = (q[j][1:] * C[i][:-1]).sum()
    R = {}
    for i in range(n):
        for j in range(i + 1, n):
            p = raw[i, j] / (raw[i, j] + raw[j, i] + 1e-12)
            lo = float(np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3))))
            R[(acts[i], acts[j])] = lo; R[(acts[j], acts[i])] = -lo
    return R

t0 = time.time()
BL, XC, OWN, KS = {}, {}, {}, {}
for blk, ks in byblk.items():
    ks = sorted(ks); KS[blk] = ks
    qopts = [cache[k]['opts'] for k in ks]
    U = sorted(set().union(*[set(o) for o in qopts]))
    BL[blk] = Block(U, qopts)
    acc = collections.Counter(); ncl = 0
    for p in clips_of.get(blk, []):
        lp = DC.logits('oof', p)
        if lp is None: continue
        ncl += 1
        for kk, v in pairlo_over(lp, [a for a in U if a in OPT2CLS]).items(): acc[kk] += v
    XC[blk] = ({k: v / max(ncl, 1) for k, v in acc.items()}, ncl)
    # own-clip pairwise (only that question's clip)
    own = {}
    for k in ks:
        lp = DC.logits('oof', tr.set_index('qa_id').loc[k, 'path'])
        own[k] = pairlo_over(lp, cache[k]['opts']) if lp is not None else {}
    OWN[blk] = own
print('precompute %.1fs  mean |U| %.2f  max %d' % (time.time()-t0,
      np.mean([BL[b].n for b in BL]), max(BL[b].n for b in BL)))
S, seg, qa = load_sources(); OP = OrderPrior(S, seg, qa)
PR = {u: OP.build(u, w_qa=1.0, gauss=False) for u in set(b[0] for b in byblk)}

def run(w_dp=1.0, w_xc=0.0, w_pr=1.0, w_own=0.0):
    preds = {}
    for blk, ks in KS.items():
        B = BL[blk]
        pm = [(w_pr, PR[blk[0]]), (w_xc, XC[blk][0])]
        if w_own:
            merged = collections.Counter()
            for k in ks:
                for kk, v in OWN[blk][k].items(): merged[kk] += v
            pm.append((w_own, dict(merged)))
        labs, _, _ = B.best(pm, [(w_dp, cache[k]['dp']) for k in ks])
        for k, l in zip(ks, labs): preds[k] = l
    return preds
def sc(p): return sum(p[k] == cache[k]['ans'] for k in cache)

print('\nbaseline centroid %d/305   prev best (dp+prior) %d/305' % (
    sum(v['cen']==v['ans'] for v in cache.values()), sc(run(1.0,0,1.0))))
print('\n=== w_xc sweep (w_dp=1, w_pr=1) ===')
for w in [0,1,2,4,8,16,32,64,128]: print(f'  w_xc={w:<5} {sc(run(1.0,w,1.0))}')
print('\n=== cross-clip only ===')
for w in [1,4,16,64]: print(f'  w_dp=0 w_xc={w:<4} w_pr=1 -> {sc(run(0.0,w,1.0))}')
print('\n=== grid: w_xc x w_pr  (w_dp=1) ===')
print('         ' + ''.join(f'{v:>7}' for v in [0.25,0.5,1,2,4]))
for w_xc in [0,2,4,8,16,32]:
    print(f'  xc={w_xc:<5}' + ''.join(f'{sc(run(1.0,w_xc,w)):>7d}' for w in [0.25,0.5,1,2,4]))
