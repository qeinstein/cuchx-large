"""Does the DINOv2-augmented dense model improve Mechanism S?
OOF frame accuracy 0.4962 -> 0.5047.  Test it as (a) a replacement and (b) an extra
evidence stream, on the same 305 OOF sequence questions.
"""
import os, sys, itertools, collections, pickle, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('CHAMP_LOGITS', 'dense_logits.npz')
from seqcore import *
from prior import load_sources, OrderPrior
from pairclf import clip_stats, feat, OPT2CLS as O2C
from fast import Block, PERMS
from sklearn.ensemble import HistGradientBoostingClassifier

tr, te, meta = load_all()
cache = pickle.load(open('seqlab/cache.pkl','rb'))
byblk = collections.defaultdict(list)
for k,v in cache.items(): byblk[v['blk']].append(k)
hau = tr[tr.source=='HAU'].copy(); hau['blk']=list(zip(hau.user,hau.aa,hau.bb))
clips_of={b:sorted(g.path.unique(),key=lambda p:p.split('-')[-1]) for b,g in hau.groupby('blk')}
fold={}
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q]=f
ufold={}
for f in range(5):
    for u in pd.read_csv(f'splits/fold_{f}_val.csv').subject_id.unique(): ufold[u]=f
Z = {'base': np.load('champ/dense_logits.npz'), 'dino': np.load('champ/dense_dino_logits.npz')}
S=pd.read_csv('seqlab/onsets.csv'); Ssrc,seg,qa=load_sources()

def lg(src,p): 
    k=f'oof|{p}'
    return Z[src][k] if k in Z[src] else None

def pairlo_hand(lp,acts,tau,conf=True):
    ci=[O2C[a] for a in acts]; sub=lp[ci]
    q=np.exp((sub-sub.max(1,keepdims=True))/tau); q=q/(q.sum(1,keepdims=True)+1e-12)
    C=np.cumsum(q,axis=1); n=len(acts); raw=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j: raw[i,j]=(q[j][1:]*C[i][:-1]).sum()
    P=np.exp(lp); tot=P.sum(0)+1e-12
    cf=np.array([float((P[c]/tot).max()) for c in ci]) if conf else np.ones(n)
    R={}
    for i in range(n):
        for j in range(i+1,n):
            p=raw[i,j]/(raw[i,j]+raw[j,i]+1e-12); p=float(np.clip(p,1e-3,1-1e-3))
            lo=np.log(p/(1-p))*float(np.sqrt(cf[i]*cf[j])); R[(acts[i],acts[j])]=lo; R[(acts[j],acts[i])]=-lo
    return R

def fold_prior(hold):
    keys=set(tuple(sorted(k)) for k in list(seg)+list(qa)); out={}
    for (x,y) in keys:
        nxy=sum(v for u,v in seg.get((x,y),{}).items() if u not in hold)+sum(v for u,v in qa.get((x,y),{}).items() if u not in hold)
        nyx=sum(v for u,v in seg.get((y,x),{}).items() if u not in hold)+sum(v for u,v in qa.get((y,x),{}).items() if u not in hold)
        n=nxy+nyx
        if n==0: continue
        p=(nxy+1)/(n+2); lo=float(np.log(p/(1-p))); out[(x,y)]=lo; out[(y,x)]=-lo
    return out

t0=time.time()
MOD={}
for src in ('base','dino'):
    ST={p:clip_stats(lg(src,p)) for p in hau.path.unique() if lg(src,p) is not None}
    segord=collections.defaultdict(list)
    for r in S.itertuples(): segord[(r.user,r.trial)].append((r.on,r.act))
    LAB=[]
    for (u,t),v in segord.items():
        p=f'HAU/{u}/{t}'
        if p not in ST: continue
        v=sorted(v)
        for i in range(len(v)):
            for j in range(len(v)):
                if i!=j and v[i][1]!=v[j][1] and v[i][0]!=v[j][0] and v[i][1] in O2C and v[j][1] in O2C:
                    LAB.append((u,p,v[i][1],v[j][1],1))
    sq=tr[(tr.source=='HAU')&(tr.category=='sequence')]
    for _,r in sq.iterrows():
        if r.path not in ST: continue
        o={L:r[L] for L in 'ABCD'}; ordr=[o[L] for L in str(r['answer'])]
        for i in range(4):
            for j in range(4):
                if i!=j and ordr[i] in O2C and ordr[j] in O2C: LAB.append((r.user,r.path,ordr[i],ordr[j],int(i<j)))
    per={}
    for f in range(5):
        hold=[u for u,ff in ufold.items() if ff==f]
        prlo=fold_prior(hold); onmu={a:float(g.on.mean()) for a,g in S[~S.user.isin(hold)].groupby('act')}
        it=[l for l in LAB if l[0] not in hold]
        X=pd.DataFrame([feat(ST[p],x,y,prlo,onmu) for (_,p,x,y,_) in it]); yv=np.array([l[4] for l in it])
        clf=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,max_depth=6,
                                           l2_regularization=1.0,random_state=0).fit(X.to_numpy(float),yv)
        per[f]=dict(clf=clf,cols=list(X.columns),prlo=prlo,onmu=onmu)
    MOD[src]=dict(per=per,ST=ST)
    print(f'  {src}: pairwise models fitted ({time.time()-t0:.0f}s)',flush=True)

BL={};KS={};EV=collections.defaultdict(dict)
for blk,ks in byblk.items():
    ks=sorted(ks); KS[blk]=ks
    qo=[cache[k]['opts'] for k in ks]; U=sorted(set().union(*[set(o) for o in qo])); BL[blk]=Block(U,qo)
    Ua=[a for a in U if a in O2C]; f=ufold.get(blk[0],0)
    for src in ('base','dino'):
        M=MOD[src]['per'][f]; ST=MOD[src]['ST']
        hand={}
        for tau in (0.5,1.0):
            acc=collections.Counter(); ncl=0
            for p in clips_of.get(blk,[]):
                lp=lg(src,p)
                if lp is None: continue
                ncl+=1
                for kk,v in pairlo_hand(lp,Ua,tau).items(): acc[kk]+=v
            hand[tau]={k:v/max(ncl,1) for k,v in acc.items()}
        rows=[];key=[]
        for p in clips_of.get(blk,[]):
            if p not in ST: continue
            for i in range(len(Ua)):
                for j in range(len(Ua)):
                    if i!=j: rows.append(feat(ST[p],Ua[i],Ua[j],M['prlo'],M['onmu'])); key.append((Ua[i],Ua[j]))
        ag=collections.defaultdict(list)
        if rows:
            pp=M['clf'].predict_proba(pd.DataFrame(rows).reindex(columns=M['cols']).to_numpy(float))[:,1]
            for kk,v in zip(key,pp):
                v=float(np.clip(v,1e-3,1-1e-3)); ag[kk].append(np.log(v/(1-v)))
        learn={}
        for (x,y) in set(ag):
            mm=(np.mean(ag[(x,y)])-np.mean(ag[(y,x)]))/2 if (y,x) in ag else np.mean(ag[(x,y)])
            learn[(x,y)]=float(mm)
        EV[blk][src]=dict(h05=hand[0.5],h10=hand[1.0],learn=learn)

def run(w):
    preds={}
    for blk,ks in KS.items():
        pm=[]
        for src in ('base','dino'):
            e=EV[blk][src]
            pm += [(w[src][0],e['h05']),(w[src][1],e['h10']),(w[src][2],e['learn'])]
        labs,_,_=BL[blk].best(pm,[(0.0,np.zeros(24)) for _ in ks])
        for k,l in zip(ks,labs): preds[k]=l
    return preds
def sc(p): return sum(p[k]==cache[k]['ans'] for k in cache)
def pf(p):
    r=collections.Counter();n=collections.Counter()
    for k,v in cache.items(): f=fold.get(k,-1);n[f]+=1;r[f]+=(p[k]==v['ans'])
    return {f:r[f] for f in sorted(n)}

Zw=(0,0,0); B=(4.,4.,1.)
print('\nchampion centroid: %d/305'%sum(v['cen']==v['ans'] for v in cache.values()))
for nm,w in [('base only (current mechanism S)',{'base':B,'dino':Zw}),
             ('dino only',{'base':Zw,'dino':B}),
             ('base + dino equal',{'base':B,'dino':B}),
             ('base + 0.5*dino',{'base':B,'dino':(2.,2.,.5)}),
             ('base + 0.25*dino',{'base':B,'dino':(1.,1.,.25)})]:
    p=run(w); print(f'  {nm:34s} {sc(p)}/305   per-fold {pf(p)}')
