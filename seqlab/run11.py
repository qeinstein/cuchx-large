import os, sys, itertools, collections, pickle, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from prior import load_sources, OrderPrior
from pairclf import clip_stats, feat, OPT2CLS as O2C
from fast import Block
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
ST={p:clip_stats(DC.logits('oof',p)) for p in hau.path.unique() if DC.logits('oof',p) is not None}
S=pd.read_csv('seqlab/onsets.csv'); Ssrc,seg,qa=load_sources()

# ---------- label sources
segord=collections.defaultdict(list)
for r in S.itertuples(): segord[(r.user,r.trial)].append((r.on,r.act))
LAB=[]
for (u,t),v in segord.items():
    p=f'HAU/{u}/{t}'
    if p not in ST: continue
    v=sorted(v)
    for i in range(len(v)):
        for j in range(len(v)):
            if i==j or v[i][1]==v[j][1] or v[i][0]==v[j][0]: continue
            if v[i][1] in O2C and v[j][1] in O2C: LAB.append((u,p,v[i][1],v[j][1],1,'seg'))
sq=tr[(tr.source=='HAU')&(tr.category=='sequence')]
for _,r in sq.iterrows():
    o={L:r[L] for L in 'ABCD'}; ordr=[o[L] for L in str(r['answer'])]
    if r.path not in ST: continue
    for i in range(4):
        for j in range(4):
            if i==j: continue
            x,y=ordr[i],ordr[j]
            if x in O2C and y in O2C: LAB.append((r.user,r.path,x,y,int(i<j),'qa'))
print('labels: seg %d  qa %d'%(sum(1 for l in LAB if l[5]=='seg'),sum(1 for l in LAB if l[5]=='qa')))

def fold_prior(hold, w_qa=1.0):
    keys=set(tuple(sorted(k)) for k in list(seg)+list(qa)); out={}
    for (x,y) in keys:
        nxy=sum(v for u,v in seg.get((x,y),{}).items() if u not in hold)+w_qa*sum(v for u,v in qa.get((x,y),{}).items() if u not in hold)
        nyx=sum(v for u,v in seg.get((y,x),{}).items() if u not in hold)+w_qa*sum(v for u,v in qa.get((y,x),{}).items() if u not in hold)
        n=nxy+nyx
        if n==0: continue
        p=(nxy+1)/(n+2); lo=float(np.log(p/(1-p))); out[(x,y)]=lo; out[(y,x)]=-lo
    return out

MOD={}
for f in range(5):
    hold=[u for u,ff in ufold.items() if ff==f]
    prlo=fold_prior(hold); onmu={a:float(g.on.mean()) for a,g in S[~S.user.isin(hold)].groupby('act')}
    it=[l for l in LAB if l[0] not in hold]
    X=pd.DataFrame([feat(ST[p],x,y,prlo,onmu) for (_,p,x,y,_,_) in it]); y=np.array([l[4] for l in it])
    clf=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,max_depth=6,
                                       l2_regularization=1.0,random_state=0).fit(X.to_numpy(float),y)
    MOD[f]=dict(clf=clf,cols=list(X.columns),prlo=prlo,onmu=onmu)
    print('  fold %d fitted on %d'%(f,len(it)))

# ---------- per-block evidence matrices
def pairlo_hand(lp,acts,tau=1.0,conf=True):
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

BL={};KS={};EV={}
for blk,ks in byblk.items():
    ks=sorted(ks);KS[blk]=ks
    qo=[cache[k]['opts'] for k in ks]; U=sorted(set().union(*[set(o) for o in qo])); BL[blk]=Block(U,qo)
    Ua=[a for a in U if a in O2C]; f=ufold.get(blk[0],0); M=MOD[f]
    hand={}
    for tau in [0.5,1.0]:
        acc=collections.Counter();ncl=0
        for p in clips_of.get(blk,[]):
            lp=DC.logits('oof',p)
            if lp is None: continue
            ncl+=1
            for kk,v in pairlo_hand(lp,Ua,tau,True).items(): acc[kk]+=v
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
        m=(np.mean(ag[(x,y)])-np.mean(ag[(y,x)]))/2 if (y,x) in ag else np.mean(ag[(x,y)])
        learn[(x,y)]=float(m)
    EV[blk]=dict(h05=hand[0.5],h10=hand[1.0],learn=learn,pr=M['prlo'])

def run(w_h05,w_h10,w_learn,w_pr,w_dp):
    preds={}
    for blk,ks in KS.items():
        e=EV[blk]
        labs,_,_=BL[blk].best([(w_h05,e['h05']),(w_h10,e['h10']),(w_learn,e['learn']),(w_pr,e['pr'])],
                              [(w_dp,cache[k]['dp']) for k in ks])
        for k,l in zip(ks,labs): preds[k]=l
    return preds
def byfold(p):
    r=collections.Counter();n=collections.Counter()
    for k,v in cache.items(): f=fold.get(k,-1);n[f]+=1;r[f]+=(p[k]==v['ans'])
    return r,n

GRID=[(a,b,c,d,e) for a in [0,4,8,16] for b in [0,4,8,16] for c in [0,1,2,4]
      for d in [0,0.5,1,2] for e in [0,0.5,1]]
GRID=[g for g in GRID if sum(1 for x in g if x)>=1]
print('\ngrid',len(GRID)); t0=time.time()
RES={}
for g in GRID:
    r,n=byfold(run(*g)); RES[g]=(sum(r.values()),{f:r[f] for f in sorted(n)},{f:n[f] for f in sorted(n)})
print('%.0fs'%(time.time()-t0))
top=sorted(RES.items(),key=lambda x:-x[1][0])
print('\ntop 10 pooled:')
for g,v in top[:10]: print(f'  h05={g[0]:<3} h10={g[1]:<3} learn={g[2]:<3} pr={g[3]:<4} dp={g[4]:<4} -> {v[0]}/305 {v[1]}')
print('\n=== NESTED CV ===')
ok=0;nn=0
for f in range(5):
    best,bs=None,-1
    for g,v in RES.items():
        o=v[0]-v[1][f]
        if o>bs: bs,best=o,g
    ok+=RES[best][1][f]; nn+=RES[best][2][f]
    print(f'  fold {f}: {best} -> {RES[best][1][f]}/{RES[best][2][f]}')
print(f'  NESTED TOTAL {ok}/{nn} = {ok/nn:.4f}  (champion 172/305 = 0.5639)')
pickle.dump(dict(RES=RES,EV=EV,KS=KS,MOD=MOD),open('seqlab/res11.pkl','wb'))
