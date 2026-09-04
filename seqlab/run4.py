import os, sys, itertools, collections, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *

cache = pickle.load(open('seqlab/cache.pkl', 'rb'))
PC = {tuple(k.split('|||')): v for k, v in json.load(open('seqlab/harn_pairs.json')).items()}
byblk = collections.defaultdict(list)
for k, v in cache.items(): byblk[v['blk']].append(k)
fold = {}
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q] = f

# scale of the DP evidence
d = np.concatenate([v['dp'] - v['dp'].max() for v in cache.values()])
print('DP score spread: p50 %.1f p90 %.1f min %.1f' % (np.percentile(d,50), np.percentile(d,90), d.min()))

_pcache = {}
def prior_lo(u, alpha=1.0, nmin=0, cmin=0.0):
    key = (u, alpha, nmin, cmin)
    if key in _pcache: return _pcache[key]
    out = {}
    for (x, y) in set(tuple(sorted(k)) for k in PC):
        nxy = sum(v for uu, v in PC.get((x,y),{}).items() if uu != u)
        nyx = sum(v for uu, v in PC.get((y,x),{}).items() if uu != u)
        n = nxy + nyx
        if n == 0: continue
        p = (nxy+alpha)/(n+2*alpha)
        if n < nmin or max(p,1-p) < cmin: continue
        lo = np.log(p/(1-p)); out[(x,y)] = lo; out[(y,x)] = -lo
    _pcache[key] = out; return out

def run(temp, lam, nmin=0, cmin=0.0, sname='dp'):
    preds = {}
    for blk, ks in byblk.items():
        ks = sorted(ks)
        pr = prior_lo(blk[0], nmin=nmin, cmin=cmin)
        lab, _, U = joint_decode([cache[k]['opts'] for k in ks],
                                 [cache[k][sname]/temp for k in ks], prior=pr, wprior=lam)
        for k, l in zip(ks, lab): preds[k] = l
    return preds

def score(preds):
    return sum(preds[k]==cache[k]['ans'] for k in cache)

print('\n=== 2D sweep: DP temperature x prior weight (joint) ===')
print('       ' + ''.join(f'{l:>7}' for l in [0.25,0.5,1,2,4,8]))
best=(None,-1)
for temp in [1,2,4,8,16,32]:
    row=[]
    for lam in [0.25,0.5,1,2,4,8]:
        v=score(run(temp,lam)); row.append(v)
        if v>best[1]: best=((temp,lam),v)
    print(f'  T={temp:<4d}' + ''.join(f'{v:>7d}' for v in row))
print('best', best)

print('\n=== support/confidence filtering of the prior (at best T,lam) ===')
T,L=best[0]
for nmin,cmin in [(0,0.0),(3,0.0),(5,0.0),(10,0.0),(0,0.7),(0,0.8),(0,0.9),(5,0.9),(10,0.9),(10,0.95),(20,0.95)]:
    print(f'  nmin={nmin:2d} cmin={cmin:.2f}: {score(run(T,L,nmin,cmin))}/305')

print('\n=== fold consistency & flip audit for the best config ===')
preds=run(T,L)
rows=[]
for k,v in cache.items():
    rows.append(dict(qa=k,fold=fold.get(k,-1),blk=str(v['blk']),ans=v['ans'],
                     base=v['cen'],new=preds[k],bc=v['cen']==v['ans'],nc=preds[k]==v['ans']))
R=pd.DataFrame(rows)
print(f"  base {R.bc.sum()}  new {R.nc.sum()}  net {R.nc.sum()-R.bc.sum():+d}")
fl=R[R.base!=R.new]
wr=int(((~fl.bc)&fl.nc).sum()); rw=int((fl.bc&(~fl.nc)).sum())
print(f"  flips {len(fl)}  W->R {wr}  R->W {rw}  W->W {len(fl)-wr-rw}  precision {wr/max(wr+rw,1):.3f}  net/100 {100*(wr-rw)/len(fl):.1f}")
for f in sorted(R.fold.unique()):
    dd=R[R.fold==f]; fd=dd[dd.base!=dd.new]
    w=int(((~fd.bc)&fd.nc).sum()); l=int((fd.bc&(~fd.nc)).sum())
    print(f"    fold {f}: base {dd.bc.sum():3d}/{len(dd):3d} new {dd.nc.sum():3d} net {dd.nc.sum()-dd.bc.sum():+3d}  flips {len(fd)} W->R {w} R->W {l}")
R.to_csv('seqlab/audit_run4.csv',index=False)
