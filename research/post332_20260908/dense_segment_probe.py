"""Evaluate the cached dense temporal logits on exact HARn child intervals.

The cache is already subject-disjoint on training parents.  This probe is an action-level
sanity check: it maps each segment's dense action scores into the visible QA options and
reports absolute OOF accuracy plus test disagreements.  It is not allowed to override the
332 solver without a separate old/new flip audit.
"""
import os
import numpy as np,pandas as pd
from champ.core import load_all
import champ.dense as D, champ.harn as H
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,'research/post332_20260908')
def parents(meta,kind_hau,kind_harn):
 hau=meta[meta.kind==kind_hau]; out={}; prow={}
 for r in meta[meta.kind==kind_harn].itertuples():
  c=hau[(hau.user==r.user)&(hau.trial==r.trial)&(hau.f0<=r.f0)&(hau.f1>=r.f1)&(hau.t0<=r.t0+1)&(hau.t1>=r.t1-1)]
  if len(c)==1:out[r.qa_path]=c.iloc[0].qa_path; prow[r.qa_path]=c.iloc[0]
 return out,prow
def main():
 tr,te,meta=load_all(); z=np.load(os.path.join(ROOT,'champ/dense_logits_screen1_bucket.npz'),allow_pickle=True); sk=np.load(os.path.join(ROOT,'champ/skel_seq.npz'),allow_pickle=True)
 pa,pr=parents(meta,'train_hau','train_harn'); rows=[]
 for _,q in tr[(tr.source=='HARn')&(tr.category=='single')].iterrows():
  p=pa.get(q.path); k='oof|'+p if p else None
  if not k or k not in z.files:continue
  mr=meta[meta.qa_path==q.path].iloc[0]; pp=pr[q.path]; F=sk[pp.unit_dir+'|F']; m=(F>=mr.f0)&(F<=mr.f1)
  if not m.any():continue
  s=z[k][:D.BG,m].mean(1); vals={}
  for L in 'ABCD':
   if pd.notna(q[L]):
    a=H.S2A.get(str(q[L]).strip()); vals[L]=float(s[D.A2I[a]]) if a in D.A2I else -999.
  sv=sorted(vals.values(),reverse=True); pred=max(vals,key=vals.get)
  rows.append(dict(qa_id=q.qa_id,truth=str(q.answer),prediction=pred,correct=int(pred==q.answer),margin=sv[0]-sv[1],action=H.S2A.get(str(q[pred]).strip()),parent=p))
 oof=pd.DataFrame(rows); oof.to_csv(os.path.join(OUT,'dense_segment_oof.csv'),index=False)
 tm=pd.read_csv(os.path.join(ROOT,'test_qa.csv')); tm['lm']=tm.path.str.extract(r'(LM_test_\d+)')[0]; champ=pd.read_csv(os.path.join(ROOT,'submissions/submission_097076_332of342_CHAMPION.csv')).set_index('qa_id').prediction; out=[]
 # test dense cache is a full-data model; only rows with an exact cached clip are shown
 for lm,g in tm[tm.source=='HARn'].groupby('lm'):
  k='test|'+lm
  if k not in z.files:continue
  s=z[k][:D.BG].mean(1)
  for _,q in g[g.category=='single'].iterrows():
   vals={}
   for L in 'ABCD':
    if pd.notna(q[L]):
     a=H.S2A.get(str(q[L]).strip()); vals[L]=float(s[D.A2I[a]]) if a in D.A2I else -999.
   sv=sorted(vals.values(),reverse=True);pred=max(vals,key=vals.get)
   out.append(dict(qa_id=q.qa_id,champion=champ[q.qa_id],dense=pred,changed=int(pred!=champ[q.qa_id]),margin=sv[0]-sv[1],values=str(vals)))
 test=pd.DataFrame(out);test.to_csv(os.path.join(OUT,'dense_segment_test.csv'),index=False)
 print('OOF',len(oof),'accuracy',float(oof.correct.mean()))
 for th in [1,2,3,5]:
  x=oof[oof.margin>=th];print('OOF margin>=',th,len(x),float(x.correct.mean()) if len(x) else 0)
 print('TEST disagreements');print(test[test.changed==1].sort_values('margin',ascending=False).to_string(index=False))
if __name__=='__main__':main()
