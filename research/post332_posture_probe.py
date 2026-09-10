"""Subject-disjoint binary residual probe: sitting down versus keyboard typing.

Uses labeled HARn segments, compact temporal skeleton descriptors, fixed model settings.
No stacking, threshold search, or leaderboard-derived labels.
"""
from pathlib import Path
import csv,json,re,argparse
from collections import Counter
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/post332_20260908'
def descriptor(k):
    k=np.nan_to_num(k.astype(float));n=len(k);width=max(1,n//4)
    means=k.mean(0);std=k.std(0);delta=k[-width:].mean(0)-k[:width].mean(0)
    vel=np.abs(np.diff(k,axis=0)).mean(0) if n>1 else np.zeros_like(means)
    return np.concatenate([means.ravel(),std.ravel(),delta.ravel(),vel.ravel(),[np.log1p(n)]])
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--negative',choices=['sitting down','using a smartphone'],default='sitting down')
    args=parser.parse_args()
    stem='phone_typing' if args.negative=='using a smartphone' else 'posture'
    meta=list(csv.DictReader((ROOT/'champ/meta.csv').open()))
    vocab=json.loads((ROOT/'champ/vocab.json').read_text())['HARN2SELF']
    wanted={k:v for k,v in vocab.items() if v in (args.negative,'typing on the keyboard')}
    z=np.load(ROOT/'champ/skel_seq.npz');rows=[];X=[]
    for r in meta:
        key=r['unit_dir']+'|K'
        if r['kind']=='train_harn' and r['action'] in wanted and key in z:
            k=z[key]
            if len(k)<2:continue
            rows.append({'path':r['qa_path'],'user':r['user'],'label':wanted[r['action']],'nframes':len(k),'hip_z_delta':float(k[-1,0,2]-k[0,0,2])});X.append(descriptor(k))
    X=np.array(X);y=np.array([int(r['label']=='typing on the keyboard') for r in rows]);groups=np.array([r['user'] for r in rows])
    oof=np.zeros(len(rows));folds=[]
    for fold,(train,test) in enumerate(GroupKFold(n_splits=5).split(X,y,groups)):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,random_state=0))
        model.fit(X[train],y[train]);oof[test]=model.predict_proba(X[test])[:,1]
        folds.append({'fold':fold,'n':len(test),'correct':int(((oof[test]>=.5)==y[test]).sum()),'users':sorted(set(groups[test]))})
    for i,r in enumerate(rows):r.update(p_typing_oof=float(oof[i]),correct=bool((oof[i]>=.5)==y[i]))
    model=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,random_state=0));model.fit(X,y)
    tests=[]
    for r in meta:
        if r['qa_path'] in ('LM_test_0017','LM_test_0015','LM_test_0016','LM_test_0041','LM_test_0042','LM_test_0044') and r['unit_dir']+'|K' in z:
            k=z[r['unit_dir']+'|K'];tests.append({'clip':r['qa_path'],'p_typing_binary':float(model.predict_proba([descriptor(k)])[0,1]),'hip_z_delta':float(k[-1,0,2]-k[0,0,2]),'nframes':len(k)})
    result={'classes':wanted,'n':len(rows),'class_counts':dict(Counter(r['label'] for r in rows)),'oof_correct':int(((oof>=.5)==y).sum()),'folds':folds,'test':tests,'limitations':'Binary probability applies only to sitting-down versus typing. Standing clips are negative controls outside its class universe. OOF segment accuracy is not correction precision versus champion.'}
    result['limitations']=f'Binary probability applies only to {args.negative} versus typing. Other actions are outside its class universe. OOF segment accuracy is not correction precision versus champion.'
    with (OUT/(stem+'_oof.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/(stem+'_probe.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
