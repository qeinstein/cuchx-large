import os, sys, itertools, collections, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from prior import load_sources, OrderPrior

cache = pickle.load(open('seqlab/cache.pkl', 'rb'))
byblk = collections.defaultdict(list)
for k, v in cache.items(): byblk[v['blk']].append(k)
S, seg, qa = load_sources()
OP = OrderPrior(S, seg, qa)
print('prior sources: seg pairs %d, qa pairs %d, union keys %d' % (len(seg), len(qa), len(OP.keys)))

def run(lam, temp=1.0, joint=True, **kw):
    preds = {}
    for blk, ks in byblk.items():
        ks = sorted(ks); pr = OP.build(blk[0], **kw)
        if joint:
            lab, _, U = joint_decode([cache[k]['opts'] for k in ks],
                                     [cache[k]['dp']/temp for k in ks], prior=pr, wprior=lam)
            for k, l in zip(ks, lab): preds[k] = l
        else:
            for k in ks:
                o = cache[k]['opts']; sv = cache[k]['dp']/temp
                best, bs = None, -np.inf
                for t, p in enumerate(PERMS):
                    idx = [ord(c)-65 for c in p]
                    v = sv[t] + lam*sum(pr.get((o[idx[i]], o[idx[j]]), 0.0)
                                        for i in range(4) for j in range(i+1,4))
                    if v > bs: bs, best = v, p
                preds[k] = best
    return preds
def sc(p): return sum(p[k]==cache[k]['ans'] for k in cache)

print('\n=== prior variants, joint, lam swept ===')
cfgs = {
 'seg counts only        ': dict(w_qa=0.0, gauss=False),
 'seg+qa counts          ': dict(w_qa=1.0, gauss=False),
 'seg+qa, gauss-shrunk   ': dict(w_qa=1.0, gauss=True, kappa=6.0),
 'seg+qa, gauss k=2      ': dict(w_qa=1.0, gauss=True, kappa=2.0),
 'seg+qa, gauss k=20     ': dict(w_qa=1.0, gauss=True, kappa=20.0),
}
best=(None,-1)
for name,kw in cfgs.items():
    row=[]
    for lam in [0.5,0.75,1.0,1.5,2.0,3.0]:
        v=sc(run(lam,**kw)); row.append(v)
        if v>best[1]: best=((name,lam,kw),v)
    print(f'  {name} ' + ' '.join(f'{l}:{v}' for l,v in zip([0.5,0.75,1.0,1.5,2.0,3.0],row)))
print('\nBEST', best[0][0], 'lam=',best[0][1], '->', best[1], '/', len(cache))
pickle.dump(best[0], open('seqlab/best_cfg.pkl','wb'))
