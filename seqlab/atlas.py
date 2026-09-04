"""Error atlas over the 0.92105 champion's 278 OOF errors."""
import os, sys, itertools, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'champ'))
from core import load_all, opts, mgroup, GROUPS

tr, te, meta = load_all()
a = pd.read_csv('champ/audit_champ.csv')
J = tr.merge(a[['qa_id','fold','pred','margin','correct']], on='qa_id')
J['blk'] = list(zip(J.user, J.aa, J.bb))

E = J[J.correct == 0].copy()
print('=== ERROR ATLAS: %d champion OOF errors ===' % len(E))
print(E.groupby(['source','category']).size().to_string())

# ---------------------------------------------------------------- EMOTION
em = J[J.category=='emotion'].copy()
def optset(r): return set(opts(r))
byblk = {k: g for k, g in em.groupby('blk')}
rows=[]
for k, g in byblk.items():
    O=[optset(r) for _,r in g.iterrows()]
    I=set.intersection(*O) if len(O)>1 else set(O[0])
    truths={r['answer'] and r[r['answer']] for _,r in g.iterrows()}
    for _,r in g.iterrows():
        tru=r[r['answer']]; prd=r[r['pred']] if r['pred'] in 'ABCD' else None
        rows.append(dict(qa_id=r.qa_id, fold=r.fold, blk=str(k), c=int(r.cc), k=len(g),
            nI=len(I), tru=tru, prd=prd, correct=int(r.correct),
            tru_in_I=tru in I, prd_in_I=(prd in I) if prd else False,
            g_tru=mgroup(tru), g_prd=mgroup(prd) if prd else None,
            same_group=(mgroup(tru)==mgroup(prd)) if prd else False,
            swap=(prd in truths - {tru}) if prd else False, margin=r.margin))
EM=pd.DataFrame(rows)
print('\n--- EMOTION (%d Q, %d errors) ---'%(len(EM),(1-EM.correct).sum()))
er=EM[EM.correct==0]
print('block size k          :', dict(EM.groupby('k').correct.agg(['size','mean']).round(3).T))
print('|intersection| nI     :'); print(EM.groupby('nI').correct.agg(n='size',acc='mean').round(3).to_string())
print('\ntruth inside intersection: %.4f  (errors where truth NOT in I: %d/%d)'%(EM.tru_in_I.mean(),(~er.tru_in_I).sum(),len(er)))
print('errors that are within-block SWAPS      : %d/%d = %.3f'%(er.swap.sum(),len(er),er.swap.mean()))
print('errors where pred/truth SAME manner group: %d/%d = %.3f'%(er.same_group.sum(),len(er),er.same_group.mean()))
print('\nconfusion group(truth) -> group(pred) on errors:')
print(pd.crosstab(er.g_tru,er.g_prd).to_string())
print('\nacc by trial position c:'); print(EM.groupby('c').correct.agg(n='size',acc='mean').round(3).to_string())
print('\nacc by (c,k):'); print(EM.groupby(['k','c']).correct.agg(n='size',acc='mean').round(3).to_string())
EM.to_csv('seqlab/atlas_emotion.csv',index=False)

# ---------------------------------------------------------------- MULTI
mu=J[J.category=='multi'].copy()
mu['nans']=[len([c for c in str(x) if c in 'ABCD']) for x in mu['answer']]
mu['npred']=[len([c for c in str(x) if c in 'ABCD']) for x in mu['pred']]
print('\n--- MULTI (%d Q, %d errors) ---'%(len(mu),(1-mu.correct).sum()))
print(mu.groupby('nans').correct.agg(n='size',acc='mean').round(3).to_string())
me=mu[mu.correct==0]
print('pred size vs answer size on errors:'); print(pd.crosstab(me.nans,me.npred).to_string())
print('over/under prediction: over %d  under %d  same-size-wrong %d'%((me.npred>me.nans).sum(),(me.npred<me.nans).sum(),(me.npred==me.nans).sum()))

# ---------------------------------------------------------------- HARn
hn=J[J.source=='HARn'].copy()
print('\n--- HARn (%d Q, %d errors) ---'%(len(hn),(1-hn.correct).sum()))
print(hn.groupby('category').correct.agg(n='size',acc='mean').round(3).to_string())
print('\naccuracy by true HARn action (worst 14):')
t=hn.groupby('harn_action').correct.agg(n='size',acc='mean').sort_values('acc')
print(t[t.n>=3].head(14).to_string())
J.to_csv('seqlab/atlas_all.csv',index=False)
print('\nwrote seqlab/atlas_all.csv, atlas_emotion.csv')
