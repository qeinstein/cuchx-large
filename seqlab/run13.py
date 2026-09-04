import os, sys, itertools, collections, pickle, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from fast import Block
R = pickle.load(open('seqlab/res11.pkl','rb')); EV=R['EV']; KS=R['KS']
cache = pickle.load(open('seqlab/cache.pkl','rb'))
tr, te, meta = load_all()
hau = tr[tr.source=='HAU'].copy(); hau['blk']=list(zip(hau.user,hau.aa,hau.bb))
clips_of = {b: sorted(g.path.unique(), key=lambda p:p.split('-')[-1]) for b,g in hau.groupby('blk')}
byblk = collections.defaultdict(list)
for k,v in cache.items(): byblk[v['blk']].append(k)
BL={}
for blk,ks in byblk.items():
    ks=sorted(ks); qo=[cache[k]['opts'] for k in ks]
    BL[blk]=Block(sorted(set().union(*[set(o) for o in qo])),qo)
fold={}
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q]=f

SRC={}
for nm,fn,off in [('dense2','dense2_logits.npz',0),('loc','loc_maps.npz',0),
                  ('locdepth','loc_depth_maps.npz',0),('locdino','loc_dino_maps.npz',0),
                  ('locfused','loc_fused_maps.npz',0)]:
    SRC[nm]=np.load(f'champ/{fn}')

def pairlo(M, acts, tau=1.0, conf=True):
    """M: C x T scores (log-prob-like). acts -> class rows."""
    ci=[OPT2CLS[a] for a in acts]
    sub=M[ci]
    q=np.exp((sub-sub.max(1,keepdims=True))/tau); q=q/(q.sum(1,keepdims=True)+1e-12)
    C=np.cumsum(q,axis=1); n=len(acts); raw=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j: raw[i,j]=(q[j][1:]*C[i][:-1]).sum()
    P=np.exp(sub); tot=np.exp(M).sum(0)+1e-12
    cf=np.array([float((np.exp(M[c])/tot).max()) for c in ci]) if conf else np.ones(n)
    Rr={}
    for i in range(n):
        for j in range(i+1,n):
            p=raw[i,j]/(raw[i,j]+raw[j,i]+1e-12); p=float(np.clip(p,1e-3,1-1e-3))
            lo=np.log(p/(1-p))*float(np.sqrt(cf[i]*cf[j]))
            Rr[(acts[i],acts[j])]=lo; Rr[(acts[j],acts[i])]=-lo
    return Rr

EXTRA=collections.defaultdict(dict)
for blk,ks in KS.items():
    U=BL[blk].U; Ua=[a for a in U if a in OPT2CLS]
    for nm,z in SRC.items():
        acc=collections.Counter(); ncl=0
        for p in clips_of.get(blk,[]):
            k=f'oof|{p}'
            if k not in z: continue
            M=z[k]
            if M.min()>=0 and M.max()<=1.0000001: M=np.log(np.clip(M,1e-9,1.0))
            ncl+=1
            for kk,v in pairlo(M,Ua).items(): acc[kk]+=v
        EXTRA[blk][nm]={k:v/max(ncl,1) for k,v in acc.items()} if ncl else {}

def run(w, extras=()):
    preds={}
    for blk,ks in KS.items():
        e=EV[blk]; pm=[(w[0],e['h05']),(w[1],e['h10']),(w[2],e['learn'])]
        for nm,wx in extras: pm.append((wx,EXTRA[blk].get(nm,{})))
        labs,_,_=BL[blk].best(pm,[(0.0,np.zeros(24)) for _ in ks])
        for k,l in zip(ks,labs): preds[k]=l
    return preds
def sc(p): return sum(p[k]==cache[k]['ans'] for k in cache)
def pf(p):
    r=collections.Counter();n=collections.Counter()
    for k,v in cache.items(): f=fold.get(k,-1);n[f]+=1;r[f]+=(p[k]==v['ans'])
    return {f:r[f] for f in sorted(n)}

W=(4.0,4.0,1.0)
base=run(W); print('current best (h05,h10,learn)=(4,4,1): %d/305  %s'%(sc(base),pf(base)))
print('\n=== add each extra source alone ===')
for nm in SRC:
    row=[]
    for wx in [0.5,1,2,4,8]:
        row.append(sc(run(W,[(nm,wx)])))
    print(f'  {nm:10s} ' + ' '.join(f'{w}:{v}' for w,v in zip([0.5,1,2,4,8],row)))
print('\n=== best pair of extras ===')
best=(None,-1)
for nm1 in SRC:
    for w1 in [1,2,4]:
        v=sc(run(W,[(nm1,w1)]))
        if v>best[1]: best=(((nm1,w1),),v)
for combo in itertools.combinations(SRC,2):
    for w1 in [1,2,4]:
        for w2 in [1,2,4]:
            v=sc(run(W,[(combo[0],w1),(combo[1],w2)]))
            if v>best[1]: best=(((combo[0],w1),(combo[1],w2)),v)
print('  best:',best)
p=run(W,list(best[0])); print('  per-fold',pf(p))
