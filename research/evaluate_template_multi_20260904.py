import os
from collections import Counter
import pandas as pd

ROOT=os.path.dirname(os.path.dirname(__file__))
TR=pd.read_csv(os.path.join(ROOT,'training_qa.csv'))
TR['user']=TR.path.str.extract(r'(user\d+)')
TR['aa']=TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[0]
TR['bb']=TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[1]
TR['blk']=list(zip(TR.user,TR.aa,TR.bb))
ACTC=['single','multi','combination','sequence']
ACT=TR[(TR.source=='HAU')&TR.category.isin(ACTC)]
M=TR[(TR.source=='HAU')&(TR.category=='multi')]

def letters(a): return [c for c in str(a) if c in 'ABCD']
def opts(r): return [str(r[c]).strip() for c in 'ABCD']
def pool(b):
 q=ACT[ACT.blk.map(lambda x:x==b)];out=set()
 for _,r in q.iterrows():
  for c in letters(r.answer):out.update(x.strip() for x in str(r[c]).split(','))
 return frozenset(out)
blocks={b:g.sort_values('path') for b,g in M.groupby('blk')}
pools={b:pool(b) for b in blocks}
ans={r.qa_id:set(str(r.answer)) for _,r in M.iterrows()}

def predict(b):
 src=[x for x in blocks if x[0]!=b[0] and pools[x]==pools[b]]
 if not src:return {},0
 # source multi answers are a per-trial set of actions; use exact option text votes.
 out={}
 for i,(_,r) in enumerate(blocks[b].iterrows()):
  vote=Counter()
  for x in src:
   sg=blocks[x]
   if i>=len(sg):continue
   sr=sg.iloc[i]
   for c in letters(sr.answer): vote[str(sr[c]).strip()]+=1
  if not vote:continue
  # only commit an action when at least half of the matching templates include it.
  chosen={a for a,n in vote.items() if n*2>=len(src)}
  oo=opts(r); p=''.join(c for c,a in zip('ABCD',oo) if a in chosen)
  if p:out[r.qa_id]=p
 return out,len(src)

rows=[]
for b in blocks:
 p,n=predict(b)
 for q,new in p.items():
  r=M[M.qa_id==q].iloc[0]; rows.append(dict(qa=q,blk=str(b),user=r.user,src=n,new=new,truth=''.join(sorted(ans[q]))))
R=pd.DataFrame(rows)
base=pd.read_csv(os.path.join(ROOT,'champ','audit_champ.csv')).set_index('qa_id')
R['base']=R.qa.map(base.pred);R['bc']=R.base==R.truth;R['nc']=R.new==R.truth;R['flip']=R.base!=R.new
R['wr']=(~R.bc)&R.nc;R['rw']=R.bc&~R.nc
R.to_csv(os.path.join(ROOT,'research','template_multi_audit.csv'),index=False)
print('rows',len(R),'base',int(R.bc.sum()),'new',int(R.nc.sum()),'flips',int(R.flip.sum()),'wr',int(R.wr.sum()),'rw',int(R.rw.sum()),'net',int(R.wr.sum()-R.rw.sum()))
print(R.groupby('src').agg(n=('qa','size'),base=('bc','sum'),new=('nc','sum'),wr=('wr','sum'),rw=('rw','sum')).to_string())
