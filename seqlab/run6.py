import os, sys, itertools, collections, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from prior import load_sources, OrderPrior

tr, te, meta = load_all()
s = seq_rows(tr)
cache = pickle.load(open('seqlab/cache.pkl', 'rb'))
byblk = collections.defaultdict(list)
for k, v in cache.items(): byblk[v['blk']].append(k)
hau = tr[tr.source == 'HAU'].copy(); hau['blk'] = list(zip(hau.user, hau.aa, hau.bb))
clips_of = {b: sorted(g.path.unique(), key=lambda p: p.split('-')[-1]) for b, g in hau.groupby('blk')}

def pairlo_over(lp, acts, tau=1.0):
    """log-odds that a precedes b, for every pair in acts, from one clip's dense logits."""
    ci = [OPT2CLS[a] for a in acts]
    sub = lp[ci]
    q = np.exp((sub - sub.max(1, keepdims=True)) / tau)
    q = q / (q.sum(1, keepdims=True) + 1e-12)
    C = np.cumsum(q, axis=1)
    n = len(acts); out = {}
    for i in range(n):
        for j in range(n):
            if i == j: continue
            out[(i, j)] = float((q[j][1:] * C[i][:-1]).sum())
    R = {}
    for i in range(n):
        for j in range(i + 1, n):
            p = out[(i, j)] / (out[(i, j)] + out[(j, i)] + 1e-12)
            p = float(np.clip(p, 1e-3, 1 - 1e-3))
            lo = np.log(p / (1 - p))
            R[(acts[i], acts[j])] = lo; R[(acts[j], acts[i])] = -lo
    return R

# precompute per-block cross-clip evidence
XC = {}
for blk, ks in byblk.items():
    U = sorted(set().union(*[set(cache[k]['opts']) for k in ks]))
    U = [a for a in U if a in OPT2CLS]
    acc = collections.Counter()
    nclip = 0
    for p in clips_of.get(blk, []):
        lp = DC.logits('oof', p)
        if lp is None: continue
        nclip += 1
        for kk, v in pairlo_over(lp, U).items(): acc[kk] += v
    XC[blk] = (dict(acc), nclip, U)
print('blocks %d, mean clips with logits %.2f, mean |U| %.2f' % (
    len(XC), np.mean([v[1] for v in XC.values()]), np.mean([len(v[2]) for v in XC.values()])))

S, seg, qa = load_sources(); OP = OrderPrior(S, seg, qa)

def run(w_dp=1.0, w_xc=0.0, w_pr=1.0, prkw=None):
    prkw = prkw or dict(w_qa=1.0, gauss=False)
    preds = {}
    pidx = {p: i for i, p in enumerate(PERMS)}
    for blk, ks in byblk.items():
        ks = sorted(ks); pr = OP.build(blk[0], **prkw)
        xc, ncl, U = XC[blk]
        qs = [cache[k]['opts'] for k in ks]
        Uu = sorted(set().union(*[set(o) for o in qs]))
        best, bs = None, -np.inf
        for perm in itertools.permutations(Uu):
            tot = 0.0
            for i in range(len(perm)):
                for j in range(i + 1, len(perm)):
                    tot += w_pr * pr.get((perm[i], perm[j]), 0.0)
                    tot += w_xc * xc.get((perm[i], perm[j]), 0.0) / max(ncl, 1)
            pos = {x: i for i, x in enumerate(perm)}
            ind = []
            for o, k in zip(qs, ks):
                lab = ''.join(sorted('ABCD', key=lambda L: pos[o[ord(L)-65]]))
                ind.append(lab)
                tot += w_dp * cache[k]['dp'][pidx[lab]]
            if tot > bs: bs, best = tot, ind
        for k, l in zip(ks, best): preds[k] = l
    return preds
def sc(p): return sum(p[k] == cache[k]['ans'] for k in cache)

print('\n=== add cross-clip dense pairwise evidence (w_dp=1, w_pr=1) ===')
for w_xc in [0, 1, 2, 4, 8, 16, 32, 64]:
    print(f'  w_xc={w_xc:<4} {sc(run(1.0, w_xc, 1.0))}/305')
print('\n=== cross-clip ONLY (no per-question DP) ===')
for w_xc in [1, 4, 16, 64]:
    print(f'  w_dp=0 w_xc={w_xc:<4} w_pr=1 -> {sc(run(0.0, w_xc, 1.0))}/305')
print('\n=== grid w_xc x w_pr at w_dp=1 ===')
print('        ' + ''.join(f'{v:>7}' for v in [0.5,1,2,4]))
for w_xc in [2,4,8,16,32]:
    print(f'  xc={w_xc:<4}' + ''.join(f'{sc(run(1.0,w_xc,w)):>7d}' for w in [0.5,1,2,4]))
