"""MECHANISM Y: nested HARn children as direct order observations for the parent session.

Each HARn clip is nested in a parent HAU clip at a known global frame range, so it contributes
an observed (action, normalized onset) pair for that session -- and the action comes from the
clip's own `single` question, which the champion answers at 0.951, versus the dense model's
0.496 frame accuracy.  Because the session's action order is shared across its trials
(104/104 conflict-free), children of ANY clip in the block constrain the whole order.

Density is matched to the test set by using only HARn clips that the QA table references
(524 of 2927), which is what a test-visible view exposes: 23 nested children across the 15
test sequence blocks (~1.5/block).
"""
import os, sys, itertools, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from fast import Block, PERMS

R = pickle.load(open('seqlab/res11.pkl','rb')); EV=R['EV']; KS=R['KS']
cache = pickle.load(open('seqlab/cache.pkl','rb'))
tr, te, meta = load_all()
au = pd.read_csv('champ/audit_champ.csv').set_index('qa_id')
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

# nesting: train HARn clip -> parent HAU clip, from timestamps + global frame ranges
mi = meta.set_index('qa_path')
hl = [(r.qa_path, r.t0, r.t1, r.f0, r.f1) for r in meta[meta.kind=='train_hau'].itertuples()
      if np.isfinite(r.t0)]
nest = {}
for r in meta[meta.kind=='train_harn'].itertuples():
    if not np.isfinite(r.t0): continue
    c=[h for h in hl if h[1]<=r.t0+0.5 and h[2]>=r.t1-0.5 and h[3]<=r.f0 and h[4]>=r.f1]
    if len(c)==1: nest[r.qa_path]=c[0][0]
print('train HARn clips uniquely nested:', len(nest))

# HARn single options are phrased differently from the HAU action vocabulary
# ('washing their face' vs 'Washing face'), so map phrase -> HARn folder -> HAU option.
import json
VOC = json.load(open('champ/vocab.json'))['HARN2HAU']
p2f = {}
for r in tr[(tr.source=='HARn')&(tr.category=='single')].itertuples():
    L=[x for x in str(r.answer) if x in 'ABCD']
    if L: p2f.setdefault(str(getattr(r,L[0])).strip(), collections.Counter())[r.harn_action]+=1
p2f = {k: v.most_common(1)[0][0] for k,v in p2f.items()}
print('HARn phrase -> folder map:', len(p2f))

# QA-referenced HARn single questions only (that is what a test view exposes)
hs = tr[(tr.source=='HARn')&(tr.category=='single')]
print('QA-referenced HARn single questions:', len(hs))
obs = collections.defaultdict(list)      # parent HAU path -> [(norm_onset, action_text)]
nok=nbad=0
for r in hs.itertuples():
    par = nest.get(r.path)
    if par is None or r.qa_id not in au.index: continue
    pv = au.loc[r.qa_id,'pred']
    if pv not in 'ABCD': continue
    phrase = str(getattr(r, pv)).strip()
    folder = p2f.get(phrase)
    act = VOC.get(folder) if folder else None
    if act is None or act not in OPT2CLS: continue
    p = mi.loc[par]; c = mi.loc[r.path]
    span = p.f1 - p.f0
    if not np.isfinite(span) or span<=0: continue
    on = (c.f0 - p.f0)/span
    obs[par].append((float(on), act))
    nok += int(au.loc[r.qa_id,'correct']==1); nbad += int(au.loc[r.qa_id,'correct']==0)
print(f'observations: {sum(len(v) for v in obs.values())} over {len(obs)} parent clips '
      f'(child action correct {nok}/{nok+nbad} = {nok/max(nok+nbad,1):.3f})')
print('observations per sequence block:',
      collections.Counter(sum(len(obs.get(p,[])) for p in clips_of.get(b,[])) for b in KS))

def child_pairs(blk, w=1.0):
    """Pairwise log-odds from the children of every clip in the block."""
    pts=[]
    for p in clips_of.get(blk,[]):
        pts += obs.get(p,[])
    out=collections.Counter()
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i==j or pts[i][1]==pts[j][1]: continue
            if pts[i][0] < pts[j][0]: out[(pts[i][1],pts[j][1])] += 1
    R={}
    for (x,y) in list(out):
        n1=out[(x,y)]; n2=out.get((y,x),0)
        if n1+n2==0: continue
        pr=(n1+0.5)/(n1+n2+1.0)
        R[(x,y)]=w*float(np.log(pr/(1-pr)))
    return R

W=(4.0,4.0,1.0)
def run(wy):
    preds={}
    for blk,ks in KS.items():
        e=EV[blk]
        pm=[(W[0],e['h05']),(W[1],e['h10']),(W[2],e['learn'])]
        if wy: pm.append((wy, child_pairs(blk)))
        labs,_,_=BL[blk].best(pm,[(0.0,np.zeros(24)) for _ in ks])
        for k,l in zip(ks,labs): preds[k]=l
    return preds
def sc(p): return sum(p[k]==cache[k]['ans'] for k in cache)
def pf(p):
    r=collections.Counter();n=collections.Counter()
    for k,v in cache.items(): f=fold.get(k,-1);n[f]+=1;r[f]+=(p[k]==v['ans'])
    return {f:r[f] for f in sorted(n)}
print('\n=== MECHANISM S + Y (nested-child order observations) ===')
base=run(0.0); print(f'  w_y=0    {sc(base)}/305  per-fold {pf(base)}')
for wy in [1,2,4,8,16,32]:
    p=run(wy); print(f'  w_y={wy:<4} {sc(p)}/305  per-fold {pf(p)}')
    if wy==8:
        A=pd.DataFrame([dict(qa=k,ans=cache[k]['ans'],b=base[k],n=p[k]) for k in cache])
        A['bc']=A.b==A.ans; A['nc']=A.n==A.ans
        fl=A[A.b!=A.n]
        print(f'          vs S alone: flips {len(fl)} W->R {int(((~fl.bc)&fl.nc).sum())} '
              f'R->W {int((fl.bc&(~fl.nc)).sum())}')
