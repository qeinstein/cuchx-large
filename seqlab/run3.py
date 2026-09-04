import os, sys, itertools, collections, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *

tr, te, meta = load_all()
s = seq_rows(tr).set_index('qa_id')
cache = pickle.load(open('seqlab/cache.pkl', 'rb'))
raw = json.load(open('seqlab/harn_pairs.json'))
PC = {tuple(k.split('|||')): v for k, v in raw.items()}      # (x,y) -> {user: n}

def prior_lo(exclude_user, alpha=1.0, nmin=0, cmin=0.0):
    """log-odds that x precedes y, leaving out one user."""
    out = {}
    keys = set(tuple(sorted(k)) for k in PC)
    for (x, y) in keys:
        nxy = sum(v for u, v in PC.get((x, y), {}).items() if u != exclude_user)
        nyx = sum(v for u, v in PC.get((y, x), {}).items() if u != exclude_user)
        n = nxy + nyx
        if n == 0: continue
        p = (nxy + alpha) / (n + 2 * alpha)
        if n < nmin or max(p, 1 - p) < cmin:
            continue
        lo = np.log(p / (1 - p))
        out[(x, y)] = lo; out[(y, x)] = -lo
    return out

byblk = collections.defaultdict(list)
for k, v in cache.items(): byblk[v['blk']].append(k)
user_of = {k: v['blk'][0] for k, v in cache.items()}
priors = {u: None for u in set(user_of.values())}

def evaluate(lam, nmin=0, cmin=0.0, sname='dp', dpscale=1.0, joint=True, label=''):
    preds = {}
    for blk, ks in byblk.items():
        ks = sorted(ks); u = blk[0]
        pr = prior_lo(u, nmin=nmin, cmin=cmin)
        if joint:
            lab, _, U = joint_decode([cache[k]['opts'] for k in ks],
                                     [cache[k][sname] * dpscale for k in ks],
                                     prior=pr, wprior=lam)
            for k, l in zip(ks, lab): preds[k] = l
        else:
            for k in ks:
                o = cache[k]['opts']; sv = cache[k][sname] * dpscale
                best, bs = None, -np.inf
                for p in PERMS:
                    idx = [ord(c) - 65 for c in p]
                    t = sv[PERMS.index(p)] + lam * sum(pr.get((o[idx[i]], o[idx[j]]), 0.0)
                                                       for i in range(4) for j in range(i + 1, 4))
                    if t > bs: bs, best = t, p
                preds[k] = best
    ok = sum(preds[k] == cache[k]['ans'] for k in cache)
    print(f'  {label:52s} {ok:3d}/{len(cache)} = {ok/len(cache):.4f}')
    return preds, ok

print('=== reference ===')
print('  champion centroid                                    %d/%d = %.4f' % (
    sum(v['cen'] == v['ans'] for v in cache.values()), len(cache),
    np.mean([v['cen'] == v['ans'] for v in cache.values()])))
print('\n=== prior only (no dense evidence) ===')
evaluate(1.0, joint=False, dpscale=0.0, label='independent, prior only')
evaluate(1.0, joint=True, dpscale=0.0, label='joint, prior only')
print('\n=== dense DP + prior, INDEPENDENT ===')
for lam in [0.5, 1, 2, 4, 8, 16, 32]:
    evaluate(lam, joint=False, label=f'independent  lam={lam}')
print('\n=== dense DP + prior, JOINT block decode ===')
for lam in [0.5, 1, 2, 4, 8, 16, 32, 64]:
    evaluate(lam, joint=True, label=f'joint  lam={lam}')
