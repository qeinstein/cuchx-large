import os, sys, itertools, collections, pickle, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from pairclf import clip_stats, feat, OPT2CLS as O2C
from fast import Block, PERMS, PIDX

tr, te, meta = load_all()
cache = pickle.load(open('seqlab/cache.pkl','rb'))
MODELS = pickle.load(open('seqlab/pairmodels.pkl','rb'))
byblk = collections.defaultdict(list)
for k,v in cache.items(): byblk[v['blk']].append(k)
hau = tr[tr.source=='HAU'].copy(); hau['blk']=list(zip(hau.user,hau.aa,hau.bb))
clips_of = {b: sorted(g.path.unique(), key=lambda p:p.split('-')[-1]) for b,g in hau.groupby('blk')}
fold={}; 
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q]=f
ufold={}
for f in range(5):
    for u in pd.read_csv(f'splits/fold_{f}_val.csv').subject_id.unique(): ufold[u]=f
ST={}
for p in hau.path.unique():
    lp=DC.logits('oof',p)
    if lp is not None: ST[p]=clip_stats(lp)

# ---- learned pairwise matrices per block (averaged over the block's clips)
t0=time.time(); LP={}; KS={}; BL={}
for blk,ks in byblk.items():
    ks=sorted(ks); KS[blk]=ks
    qopts=[cache[k]['opts'] for k in ks]
    U=sorted(set().union(*[set(o) for o in qopts])); BL[blk]=Block(U,qopts)
    Ua=[a for a in U if a in O2C]
    f=ufold.get(blk[0],0); M=MODELS[f]
    rows=[];key=[]
    for p in clips_of.get(blk,[]):
        if p not in ST: continue
        for i in range(len(Ua)):
            for j in range(len(Ua)):
                if i==j: continue
                rows.append(feat(ST[p],Ua[i],Ua[j],M['prlo'],M['onmu'])); key.append((Ua[i],Ua[j]))
    acc=collections.defaultdict(list)
    if rows:
        pp=M['clf'].predict_proba(pd.DataFrame(rows).reindex(columns=M['cols']).to_numpy(float))[:,1]
        for kk,v in zip(key,pp):
            v=float(np.clip(v,1e-3,1-1e-3)); acc[kk].append(np.log(v/(1-v)))
    # antisymmetrise
    out={}
    for (x,y) in set(acc):
        if (y,x) in acc:
            m=(np.mean(acc[(x,y)])-np.mean(acc[(y,x)]))/2
        else: m=np.mean(acc[(x,y)])
        out[(x,y)]=float(m)
    LP[blk]=out
print('learned pair matrices %.1fs'%(time.time()-t0))

# count-prior matrices (for blending)
PRm={}
for blk in byblk:
    f=ufold.get(blk[0],0); PRm[blk]=MODELS[f]['prlo']

def run(w_lp=1.0,w_pr=0.0,w_dp=0.0):
    preds={}
    for blk,ks in KS.items():
        labs,_,_=BL[blk].best([(w_lp,LP[blk]),(w_pr,PRm[blk])],
                              [(w_dp,cache[k]['dp']) for k in ks])
        for k,l in zip(ks,labs): preds[k]=l
    return preds
def sc(p): return sum(p[k]==cache[k]['ans'] for k in cache)
def per_fold(p):
    r=collections.Counter();n=collections.Counter()
    for k,v in cache.items(): f=fold.get(k,-1); n[f]+=1; r[f]+=(p[k]==v['ans'])
    return {f:(r[f],n[f]) for f in sorted(n)}

print('\nchampion centroid: %d/305'%sum(v['cen']==v['ans'] for v in cache.values()))
print('\n=== learned pairwise, joint decode ===')
for w in [1.0]:
    p=run(w,0,0); print(f'  w_lp={w} alone -> {sc(p)}/305  per-fold {per_fold(p)}')
print('\n=== blend learned pairwise + count prior + per-question DP ===')
best=(None,-1)
for w_pr in [0,0.25,0.5,1.0,2.0]:
    row=[]
    for w_dp in [0,0.25,0.5,1.0]:
        p=run(1.0,w_pr,w_dp); v=sc(p); row.append(v)
        if v>best[1]: best=((1.0,w_pr,w_dp),v)
    print(f'  w_pr={w_pr:<5}' + ''.join(f'{v:>6d}' for v in row) + '   (w_dp=0,.25,.5,1)')
print('\nbest', best)
p=run(*best[0]); print('per-fold', per_fold(p))
pickle.dump(dict(LP=LP,PRm=PRm,KS=KS),open('seqlab/lp10.pkl','wb'))
# flip audit
rows=[]
for k,v in cache.items():
    rows.append(dict(qa=k,fold=fold.get(k,-1),ans=v['ans'],base=v['cen'],new=p[k],
                     bc=v['cen']==v['ans'],nc=p[k]==v['ans']))
R=pd.DataFrame(rows); fl=R[R.base!=R.new]
wr=int(((~fl.bc)&fl.nc).sum()); rw=int((fl.bc&(~fl.nc)).sum())
print(f'\nFLIP AUDIT: flips {len(fl)}  W->R {wr}  R->W {rw}  W->W {len(fl)-wr-rw}  precision {wr/max(wr+rw,1):.3f}')
R.to_csv('seqlab/audit_run10.csv',index=False)
