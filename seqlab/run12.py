import os, sys, itertools, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from fast import Block
R=pickle.load(open('seqlab/res11.pkl','rb')); EV=R['EV']; KS=R['KS']; RES=R['RES']
cache=pickle.load(open('seqlab/cache.pkl','rb'))
tr,te,meta=load_all()
byblk=collections.defaultdict(list)
for k,v in cache.items(): byblk[v['blk']].append(k)
hau=tr[tr.source=='HAU'].copy(); hau['blk']=list(zip(hau.user,hau.aa,hau.bb))
BL={}
for blk,ks in byblk.items():
    ks=sorted(ks); qo=[cache[k]['opts'] for k in ks]
    BL[blk]=Block(sorted(set().union(*[set(o) for o in qo])),qo)
fold={}
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q]=f

CFG=(4,4,1,0,0)
print('=== robustness of the chosen config neighbourhood ===')
for g in [(4,4,1,0,0),(8,8,2,0,0),(4,8,1,0,0),(8,4,1,0,0),(4,4,2,0,0),(4,4,1,0,0.5),(4,4,1,0.5,0),(16,16,4,0,0)]:
    if g in RES: print(f'  {g} -> {RES[g][0]}/305  {RES[g][1]}')

def decode(cfg):
    w_h05,w_h10,w_learn,w_pr,w_dp=cfg
    preds={}; marg={}
    for blk,ks in KS.items():
        e=EV[blk]; B=BL[blk]
        pm=[(w_h05,e['h05']),(w_h10,e['h10']),(w_learn,e['learn']),(w_pr,e['pr'])]
        dpt=[(w_dp,cache[k]['dp']) for k in ks]
        t=B.score(pm,dpt); k0=int(np.argmax(t))
        labs=[PERMS[int(L[k0])] for L in B.L]
        # per-question margin: best score minus best score with a DIFFERENT label for that q
        for qi,(k,lab) in enumerate(zip(ks,labs)):
            other=t[B.L[qi]!=B.L[qi][k0]]
            marg[k]=float(t[k0]-other.max()) if len(other) else np.inf
            preds[k]=lab
        # block margin
    return preds,marg
preds,marg=decode(CFG)
rows=[]
for k,v in cache.items():
    rows.append(dict(qa=k,fold=fold.get(k,-1),blk=str(v['blk']),ans=v['ans'],base=v['cen'],
                     new=preds[k],bc=v['cen']==v['ans'],nc=preds[k]==v['ans'],margin=marg[k]))
A=pd.DataFrame(rows)
print(f'\nchosen config {CFG}: base {A.bc.sum()}  new {A.nc.sum()}  net {A.nc.sum()-A.bc.sum():+d}')
fl=A[A.base!=A.new]
wr=int(((~fl.bc)&fl.nc).sum()); rw=int((fl.bc&(~fl.nc)).sum())
print(f'FLIPS {len(fl)}  W->R {wr}  R->W {rw}  W->W {len(fl)-wr-rw}  precision {wr/max(wr+rw,1):.3f}  net/100 {100*(wr-rw)/len(fl):.1f}')
print('\nper fold:')
for f in sorted(A.fold.unique()):
    d=A[A.fold==f]; fd=d[d.base!=d.new]
    w=int(((~fd.bc)&fd.nc).sum()); l=int((fd.bc&(~fd.nc)).sum())
    print(f'  fold {f}: base {d.bc.sum():3d}/{len(d):3d} new {d.nc.sum():3d} net {d.nc.sum()-d.bc.sum():+3d}  flips {len(fd)} W->R {w} R->W {l} prec {w/max(w+l,1):.2f}')
print('\n=== margin gating of the flips ===')
q=fl.margin.quantile([0,.2,.4,.5,.6,.8]).tolist()
for thr in sorted(set([0.0]+ [round(x,3) for x in q])):
    d=fl[fl.margin>=thr]
    w=int(((~d.bc)&d.nc).sum()); l=int((d.bc&(~d.nc)).sum())
    kept=A.copy(); 
    print(f'  margin>={thr:8.3f}: flips {len(d):3d} W->R {w:3d} R->W {l:3d} prec {w/max(w+l,1):.3f} net {w-l:+3d}')
print('\n=== accuracy by margin decile (all questions, new decoder) ===')
A['dec']=pd.qcut(A.margin,5,labels=False,duplicates='drop')
print(A.groupby('dec').agg(n=('nc','size'),base=('bc','mean'),new=('nc','mean'),mmin=('margin','min')).round(3).to_string())
A.to_csv('seqlab/audit_final_seq.csv',index=False)
