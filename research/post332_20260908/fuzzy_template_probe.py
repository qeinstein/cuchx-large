"""Partial option-template transfer audit.

Exact option multisets were already tested.  This probe asks whether a near-template
(same source/category/question and >=3 of 4 option texts) carries any trustworthy
semantic answer signal.  It is label-safe: OOF donors exclude the target's user, and
test output is diagnostic only.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
TR=pd.read_csv(ROOT/'training_qa.csv',encoding='utf-8-sig'); TE=pd.read_csv(ROOT/'test_qa.csv')
if 'prediction' in TE: TE=TE.drop(columns='prediction')

def opts(r): return [str(r[L]).strip() for L in 'ABCD']
def sig(r): return (str(r.source),str(r.category),str(r.question).strip())
def sem(r):
 oo=opts(r); return tuple(oo[ord(x)-65] for x in str(r.answer) if x in 'ABCD')
def toletters(r,a):
 oo=opts(r)
 if not a or not all(x in oo for x in a): return None
 return ''.join('ABCD'[oo.index(x)] for x in a)
def user(r):
 p = str(r if isinstance(r,str) else r.path)
 x=p.split('/'); return next((q for q in x if q.startswith('user')),None)

def predict(r, donors, min_overlap):
 oo=set(opts(r)); seen=[]
 for d in donors.get(sig(r),[]):
  ov=len(oo & set(d['options']))
  if ov>=min_overlap: seen.append(d)
 if not seen: return None
 # only donors whose semantic answer is representable in target's options can vote.
 valid=[d for d in seen if toletters(r,d['answer']) is not None]
 if not valid: return None
 cnt=Counter(tuple(d['answer']) for d in valid); a,v=cnt.most_common(1)[0]
 return dict(pred=toletters(r,a),support=len(valid),raw_support=len(seen),purity=v/len(valid),answers=len(cnt),semantic='|'.join(a))

def main():
 rows=[]
 users=sorted(TR.path.map(user).dropna().unique(),key=lambda x:int(x[4:]))
 for held in users:
  donors=defaultdict(list)
  for _,d in TR[TR.path.map(user)!=held].iterrows(): donors[sig(d)].append({'options':opts(d),'answer':sem(d),'user':user(d)})
  for _,r in TR[TR.path.map(user)==held].iterrows():
   for k in (3,2):
    p=predict(r,donors,k)
    if p:
     rows.append(dict(split='oof',held=held,qa_id=r.qa_id,category=r.category,overlap=k,user=held,truth=str(r.answer),**p)); break
 test_rows=[]; donors=defaultdict(list)
 for _,d in TR.iterrows(): donors[sig(d)].append({'options':opts(d),'answer':sem(d),'user':user(d)})
 for _,r in TE.iterrows():
  for k in (3,2):
   p=predict(r,donors,k)
   if p:
    test_rows.append(dict(split='test',qa_id=r.qa_id,category=r.category,overlap=k,current='?',**p)); break
 out=pd.DataFrame(rows); out['correct']=out.pred==out.truth; out.to_csv(ROOT/'research/post332_20260908/fuzzy_template_oof.csv',index=False)
 td=pd.DataFrame(test_rows); td.to_csv(ROOT/'research/post332_20260908/fuzzy_template_test.csv',index=False)
 print('OOF',len(out));
 for k,g in out.groupby('overlap'):
  for purity in (1,.9,.8):
   z=g[g.purity>=purity]; print('overlap',k,'purity',purity,'n',len(z),'correct',int(z.correct.sum()),'acc',float(z.correct.mean()) if len(z) else None)
 print('category/high-purity'); print(out[out.purity>=.9].groupby('category').correct.agg(['count','sum','mean']).to_string())
 print('TEST',len(td)); print(td[td.purity>=.9].to_string(index=False))
if __name__=='__main__': main()
