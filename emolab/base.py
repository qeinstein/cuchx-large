import os, sys, itertools, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'champ'))
from core import load_all, opts, gt_letters, mgroup, GROUPS, PHYS, block_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

tr, te, meta = load_all()
print('PHYS n=%d' % len(PHYS)); print(PHYS)
em = tr[tr.category=='emotion'].copy(); em['blk']=list(zip(em.user,em.aa,em.bb))
em['lab']=[str(r[gt_letters(r)[0]]).strip() for _,r in em.iterrows()]
em['grp']=[mgroup(l) for l in em.lab]
print('\ngroup distribution:', dict(em.grp.value_counts()))
print('manners per group:')
for g in GROUPS:
    print(' ', g, sorted(em[em.grp==g].lab.unique()))
ufold={}
for f in range(5):
    for u in pd.read_csv('splits/fold_%d_val.csv'%f).subject_id.unique(): ufold[u]=f
mfeat={r.qa_path:{c:getattr(r,c) for c in PHYS if hasattr(r,c)} for r in meta.itertuples()}

def build(extra=None):
    X=[];y=[];u=[];bk=[];pos=[]
    for b,g in em.groupby('blk'):
        g=g.sort_values('cc'); blk=[r.path for _,r in g.iterrows()]; k=len(blk)
        bf=block_features(blk,mfeat,k)
        for i,(_,r) in enumerate(g.iterrows()):
            d=dict(bf[i])
            if extra is not None: d.update(extra.get(r.path,{}))
            X.append(d); y.append(r.grp); u.append(r.user); bk.append(b); pos.append(i)
    return pd.DataFrame(X), np.array(y), np.array(u), bk, np.array(pos)

def evaluate(X,y,u,label):
    X=X.loc[:,X.notna().any()]
    cols=list(X.columns); Xv=X.to_numpy(float)
    oof=np.empty(len(y),object); P=np.zeros((len(y),len(GROUPS)))
    for f in range(5):
        m=np.array([ufold.get(uu,-1)==f for uu in u])
        clf=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,max_depth=5,
                                           l2_regularization=1.0,random_state=0)
        clf.fit(Xv[~m],y[~m]); oof[m]=clf.predict(Xv[m])
        pp=clf.predict_proba(Xv[m])
        for j,c in enumerate(clf.classes_): P[m,GROUPS.index(c)]=pp[:,j]
    acc=(oof==y).mean()
    print(f'  {label:44s} 5-way group acc {acc:.4f}  ({(oof==y).sum()}/{len(y)})  nfeat={len(cols)}')
    return oof,P,acc

if __name__=='__main__':
    X,y,u,bk,pos=build()
    print('\n=== baseline group classifier on existing feats.csv features ===')
    oof,P,acc=evaluate(X,y,u,'block_features (abs+z+rank) [champion set]')
    np.save('emolab/P_base.npy',P); 
    pd.DataFrame(dict(y=y,u=u,blk=[str(b) for b in bk],pos=pos,oof=oof)).to_csv('emolab/meta_base.csv',index=False)
    # ablations
    evaluate(X[[c for c in X.columns if c.startswith('a_')]],y,u,'absolute only')
    evaluate(X[[c for c in X.columns if c.startswith('z_') or c.startswith('r_')]],y,u,'within-block relative only')
    evaluate(X[['pos','k','posfrac']],y,u,'position only')
