"""Honest pair-regime validation: when a clip is dropped, its dense logits must not
contribute to the cross-clip evidence either."""
import os, sys, itertools, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from pairclf import clip_stats, feat, OPT2CLS as O2C
from fast import Block, PERMS

MOD = pickle.load(open('seqlab/res11.pkl','rb'))['MOD']
cache = pickle.load(open('seqlab/cache.pkl','rb'))
tr, te, meta = load_all()
hau = tr[tr.source=='HAU'].copy(); hau['blk']=list(zip(hau.user,hau.aa,hau.bb))
clips_of = {b: sorted(g.path.unique(), key=lambda p:p.split('-')[-1]) for b,g in hau.groupby('blk')}
q2p = dict(zip(tr.qa_id, tr.path))
fold={}
for f in range(5):
    for qq in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[qq]=f
ufold={}
for f in range(5):
    for u in pd.read_csv(f'splits/fold_{f}_val.csv').subject_id.unique(): ufold[u]=f
ST={p:clip_stats(DC.logits('oof',p)) for p in hau.path.unique() if DC.logits('oof',p) is not None}
byblk=collections.defaultdict(list)
for k,v in cache.items(): byblk[v['blk']].append(k)

def pairlo_hand(lp,acts,tau,conf=True):
    ci=[O2C[a] for a in acts]; sub=lp[ci]
    qq=np.exp((sub-sub.max(1,keepdims=True))/tau); qq=qq/(qq.sum(1,keepdims=True)+1e-12)
    C=np.cumsum(qq,axis=1); n=len(acts); raw=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j: raw[i,j]=(qq[j][1:]*C[i][:-1]).sum()
    P=np.exp(lp); tot=P.sum(0)+1e-12
    cf=np.array([float((P[c]/tot).max()) for c in ci]) if conf else np.ones(n)
    Rr={}
    for i in range(n):
        for j in range(i+1,n):
            p=raw[i,j]/(raw[i,j]+raw[j,i]+1e-12); p=float(np.clip(p,1e-3,1-1e-3))
            lo=np.log(p/(1-p))*float(np.sqrt(cf[i]*cf[j])); Rr[(acts[i],acts[j])]=lo; Rr[(acts[j],acts[i])]=-lo
    return Rr

W=(4.0,4.0,1.0)
def solve(blk, ks, use_clips):
    """Decode questions ks using dense evidence only from use_clips."""
    qo=[cache[k]['opts'] for k in ks]
    U=sorted(set().union(*[set(o) for o in qo])); Ua=[a for a in U if a in O2C]
    B=Block(U,qo); M=MOD[ufold.get(blk[0],0)]
    hand={}
    for tau in (0.5,1.0):
        acc=collections.Counter(); ncl=0
        for p in use_clips:
            lp=DC.logits('oof',p)
            if lp is None: continue
            ncl+=1
            for kk,v in pairlo_hand(lp,Ua,tau).items(): acc[kk]+=v
        hand[tau]={k:v/max(ncl,1) for k,v in acc.items()}
    rows=[];key=[]
    for p in use_clips:
        if p not in ST: continue
        st=ST[p]
        for a in range(len(Ua)):
            for c in range(len(Ua)):
                if a!=c: rows.append(feat(st,Ua[a],Ua[c],M['prlo'],M['onmu'])); key.append((Ua[a],Ua[c]))
    ag=collections.defaultdict(list)
    if rows:
        pp=M['clf'].predict_proba(pd.DataFrame(rows).reindex(columns=M['cols']).to_numpy(float))[:,1]
        for kk,v in zip(key,pp):
            v=float(np.clip(v,1e-3,1-1e-3)); ag[kk].append(np.log(v/(1-v)))
    learn={}
    for (x,y) in set(ag):
        mm=(np.mean(ag[(x,y)])-np.mean(ag[(y,x)]))/2 if (y,x) in ag else np.mean(ag[(x,y)])
        learn[(x,y)]=float(mm)
    t=B.score([(W[0],hand[0.5]),(W[1],hand[1.0]),(W[2],learn)],[(0.0,np.zeros(24)) for _ in ks])
    k0=int(np.argmax(t))
    return {k:PERMS[int(L[k0])] for k,L in zip(ks,B.L)}

rows=[]
for blk,ks in byblk.items():
    ks=sorted(ks)
    if len(ks)!=3: continue
    for drop in range(3):
        keep=[k for i,k in enumerate(ks) if i!=drop]
        kept_clips=[q2p[k] for k in keep]
        pr=solve(blk,keep,kept_clips)
        for k in keep:
            rows.append(dict(qa=k,fold=fold.get(k,-1),regime='pair_honest',ans=cache[k]['ans'],
                             champ=cache[k]['cen'],new=pr[k]))
A=pd.DataFrame(rows); A['cc']=A.champ==A.ans; A['nc']=A.new==A.ans
fl=A[A.champ!=A.new]
wr=int(((~fl.cc)&fl.nc).sum()); rw=int((fl.cc&(~fl.nc)).sum())
print(f'PAIR regime, evidence from the 2 kept clips only:')
print(f'  n={len(A)}  champion {A.cc.sum()} ({A.cc.mean():.4f})  S {A.nc.sum()} ({A.nc.mean():.4f})  delta {A.nc.mean()-A.cc.mean():+.4f}')
print(f'  flips {len(fl)}  W->R {wr}  R->W {rw}  W->W {len(fl)-wr-rw}  precision {wr/max(wr+rw,1):.3f}  net/100 {100*(wr-rw)/max(len(fl),1):.1f}')
for f in sorted(A.fold.unique()):
    d=A[A.fold==f]; fd=d[d.champ!=d.new]
    w=int(((~fd.cc)&fd.nc).sum()); l=int((fd.cc&(~fd.nc)).sum())
    print(f'    fold {f}: champ {d.cc.sum():4d}/{len(d):4d} S {d.nc.sum():4d} net {d.nc.sum()-d.cc.sum():+4d} flips {len(fd):3d} W->R {w:3d} R->W {l:3d} prec {w/max(w+l,1):.2f}')
A.to_csv('seqlab/audit_pair_honest.csv',index=False)
