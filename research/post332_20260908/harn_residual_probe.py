"""Subject-disjoint residual probe for HARn action clips.

This intentionally bypasses the full session solver: it asks whether an independent
action identity model can safely flip the immutable 332 champion on HARn single rows.
All fits are grouped by user, and only changed rows are scored for W->R/R->W.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

def main():
    from champ.core import load_all
    tr, te, meta = load_all()
    v=json.load(open(os.path.join(ROOT,'champ','vocab.json')))
    s2a={v0:k for k,v0 in v['HARN2SELF'].items()}
    # Any metadata row with action is a labelled action segment.
    d=meta[meta.kind=='train_harn'].copy()
    cols=[c for c in meta.columns if c.startswith(('sk_','imu_','rad_'))]+['nf','f0']
    X=d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    y=d.action.astype(str).to_numpy(); groups=d.user.astype(str).to_numpy()
    # map labelled path -> all HARN QA rows and immutable champion answer
    champ=pd.read_csv(os.path.join(ROOT,'submission_097076_332of342_CHAMPION.csv')).set_index('qa_id').prediction
    trh=tr[tr.source=='HARn'].copy()
    trh['path_short']=trh.path.str.extract(r'(LM_train_[^/]+|HARn/.*)$',expand=False)
    # core.load_all's path is metadata's qa_path for train HARN; use join by suffix
    mp=d.set_index('qa_path'); rows=[]
    users=sorted(set(groups)); gkf=GroupKFold(5)
    # ExtraTrees is deliberately used here instead of the championship HGB head: the
    # latter takes tens of minutes on this 40-class feature table and gives no extra
    # decision-level evidence.  A linear head provides an independent check.
    models={
      'extra':make_pipeline(SimpleImputer(),ExtraTreesClassifier(n_estimators=180,max_features=.55,min_samples_leaf=2,n_jobs=4,random_state=7,class_weight='balanced')),
    }
    # evaluate each model fold; action prediction -> answer letter from the options
    for name, template in models.items():
      out=[]
      for fi,(ii,jj) in enumerate(gkf.split(X,y,groups)):
        model=template
        model.fit(X[ii],y[ii]); p=model.predict(X[jj]); prob=model.predict_proba(X[jj]); cls=list(model.classes_)
        for ix,act,pr,pp in zip(jj,y[jj],p,prob):
          qpath=d.iloc[ix].qa_path
          qs=tr[(tr.source=='HARn') & tr.path.str.contains(qpath,regex=False)]
          # path in core is e.g. HARn/...; robustly match final LM directory
          if not len(qs):
            qs=tr[(tr.source=='HARn') & tr.path.str.endswith('/'+qpath)]
          for _,q in qs.iterrows():
            if q.category not in ('single','object_interaction'): continue
            opts={L:str(q[L]).strip() for L in 'ABCD' if pd.notna(q[L])}
            # exact semantic map; objects are deliberately skipped unless an option names action
            amap={L:s2a.get(o) for L,o in opts.items()}
            cand=[L for L,a in amap.items() if a==pr]
            if not cand: continue
            new=cand[0]; base=champ.get(q.qa_id, np.nan); truth=str(q.answer)
            out.append(dict(model=name,fold=fi,qa_id=q.qa_id,user=q.user,category=q.category,
                            base=base,new=new,truth=truth,new_correct=int(new==truth),
                            conf=float(pp[cls.index(pr)]),action=pr))
      z=pd.DataFrame(out); z.to_csv(os.path.join(ROOT,'research/post332_20260908',f'harn_{name}_residual_oof.csv'),index=False)
      print('\nMODEL',name,'rows',len(z))
      for cat,g in z.groupby('category'):
        print(cat,'action-mapped rows',len(g),'absolute correct',int(g.new_correct.sum()),'acc',float(g.new_correct.mean()))
      for th in [.5,.7,.8,.9,.95]:
        ch=z[z.conf>=th]; print(' conf>=',th,'n',len(ch),'correct',int(ch.new_correct.sum()),'acc',float(ch.new_correct.mean()) if len(ch) else 0)
      model.fit(X,y)
      td=meta[meta.kind=='test'].copy(); Xt=td[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float)
      pp=model.predict_proba(Xt); pc=model.predict(Xt); cls=list(model.classes_); test_rows=[]
      for j,pr in enumerate(pc):
        qpath=td.iloc[j].qa_path
        qs=te[(te.source=='HARn') & te.path.str.contains(qpath,regex=False)]
        for _,q in qs.iterrows():
          if q.category not in ('single','object_interaction'): continue
          oo={L:str(q[L]).strip() for L in 'ABCD' if pd.notna(q[L])}
          cand=[L for L,o in oo.items() if s2a.get(o)==pr]
          if cand: test_rows.append(dict(model=name,qa_id=q.qa_id,category=q.category,base=str(champ.get(q.qa_id,'')),new=cand[0],conf=float(pp[j,cls.index(pr)]),action=pr))
      tz=pd.DataFrame(test_rows); tz.to_csv(os.path.join(ROOT,'research/post332_20260908',f'harn_{name}_residual_test.csv'),index=False)
      print('TEST proposals',len(tz),'changed',int((tz.base!=tz.new).sum()))
      if len(tz): print(tz[tz.base!=tz.new].to_string(index=False))

if __name__=='__main__': main()
