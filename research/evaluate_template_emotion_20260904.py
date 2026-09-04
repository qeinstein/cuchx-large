import itertools
import os
from collections import Counter
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
TR = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
TR['user'] = TR.path.str.extract(r'(user\d+)')
TR['aa'] = TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[0]
TR['bb'] = TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[1]
TR['blk'] = list(zip(TR.user, TR.aa, TR.bb))
E = TR[(TR.source == 'HAU') & (TR.category == 'emotion')]
ACT = TR[(TR.source == 'HAU') & TR.category.isin(['single','multi','combination','sequence'])]

def letters(a): return [x for x in str(a) if x in 'ABCD']
def opts(r): return [str(r[x]).strip() for x in 'ABCD']
def group(x):
    x = x.lower()
    if any(k in x for k in ['slow','leisure','unhurried','calm','peace','relax','lazy','gentle','soft','quiet','comfort','sooth','patient','light']): return 'slow'
    if any(k in x for k in ['fast','quick','rapid','hast','hurried','swift','brisk','urgent','frantic','forceful','eager','impatient']): return 'fast'
    if any(k in x for k in ['care','cautious','meticul','precis','thorough','deliber','method','attent','dilig','earnest','neat','order','serious']): return 'care'
    if any(k in x for k in ['nerv','anx','tens','restless']): return 'nerv'
    return 'neutral'

def pool(b):
    q = ACT[ACT.blk.map(lambda x: x == b)]
    out=set()
    for _,r in q.iterrows():
        for x in letters(r.answer): out.update(z.strip() for z in str(r[x]).split(','))
    return frozenset(out)

blocks = {b:g.sort_values('path') for b,g in E.groupby('blk')}
pools = {b:pool(b) for b in blocks}
true = {}
true_letter = {}
for _,r in E.iterrows():
    a=letters(r.answer)[0]; true[r.qa_id]=group(r[a]); true_letter[r.qa_id]=a

def source_labels(b):
    g=blocks[b].sort_values('path')
    return [true[r.qa_id] for _,r in g.iterrows()]

def candidates(target):
    return [b for b in blocks if b[0] != target[0] and pools[b] == pools[target]]

def predict(target):
    tg=blocks[target].sort_values('path')
    src=candidates(target)
    if not src: return {}, 0, {}
    # For complete triples, use the modal manner group at each chronological trial.
    votes=[Counter(source_labels(b)[i] for b in src if len(source_labels(b)) >= 3)
           for i in range(3)]
    out={}; exact={}
    for i,(_,r) in enumerate(tg.iterrows()):
        og=opts(r); poss=[group(x) for x in og]
        v=votes[i] if i < 3 else Counter()
        if v:
            want=v.most_common(1)[0][0]
            ok=[j for j,g in enumerate(poss) if g==want]
            if ok: out[r.qa_id]='ABCD'[ok[0]]
            labels=Counter()
            for b in src:
                sl=source_labels(b)
                if i < len(sl):
                    # reconstruct the exact source label from the source answer
                    sr=blocks[b].sort_values('path').iloc[i]
                    aa=letters(sr.answer)[0]; labels[str(sr[aa]).strip()] += 1
            if labels:
                best=labels.most_common(1)[0][0]
                if best in [str(x).strip() for x in opts(r)]:
                    exact[r.qa_id]='ABCD'[[str(x).strip() for x in opts(r)].index(best)]
    return out,len(src),exact

rows=[]
for b in blocks:
    p,n,ex=predict(b)
    for q,new in p.items():
        r=E[E.qa_id==q].iloc[0]
        old='?' # filled from a saved OOF prediction below when available
        rows.append(dict(qa=q,blk=str(b),user=r.user,src=n,ans=true[q],new=new,
                         exact=ex.get(q),new_group=group(r[new])))
R=pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT,'research','template_emotion_transfer.csv'),index=False)
print('full-triple transferred rows',len(R),'correct group',sum(R.ans==R.new_group),
      'accuracy',sum(R.ans==R.new_group)/max(1,len(R)))
base = pd.read_csv(os.path.join(ROOT, 'champ', 'audit_champ.csv')).set_index('qa_id')
R['base'] = [base.loc[q, 'pred'] for q in R.qa]
R['bc'] = R.base == R.qa.map(true_letter)
R['nc'] = R.new == R.qa.map(true_letter)
print('decision audit vs base:', 'base',int(R.bc.sum()), 'new',int(R.nc.sum()),
      'flips',int((R.base!=R.new).sum()), 'W->R',int(((~R.bc)&R.nc).sum()),
      'R->W',int((R.bc&~R.nc).sum()), 'net',int(((~R.bc)&R.nc).sum()-(R.bc&~R.nc).sum()))
print('target blocks',len(blocks),'covered',sum(bool(candidates(b)) for b in blocks),
      'source count',Counter(len(candidates(b)) for b in blocks))
