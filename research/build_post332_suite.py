"""Offline, evidence-gated candidates and signed-effect decoder; never submits."""
import argparse,csv,hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/post332_20260908'
PRIMARY=['test_0488','test_0477','test_0506','test_0519','test_0501']
NEW={'test_0483':'C','test_0526':'C'}
SHA='25e79e1dae1149bdad81d081d1fad3a94db4e1eb88df7f00e91276e6d5668d56'
def read(path):return list(csv.DictReader(path.open()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--observe',action='append',default=[],help='file_stem=integer_delta_vs_332');a=p.parse_args()
    base=ROOT/'submission_097076_332of342_CHAMPION.csv'
    assert hashlib.sha256(base.read_bytes()).hexdigest()==SHA
    rows=read(base);old={r['qa_id']:r['prediction'] for r in read(ROOT/'submission_096491_330of342_CHAMPION.csv')}
    current={r['qa_id']:r['prediction'] for r in rows};qa={r['qa_id']:r for r in read(ROOT/'test_qa.csv')}
    assert len(rows)==len(current)==682 and list(current)==list(qa)
    configs={'KEEP_SCORED_332':{}}
    for q,v in NEW.items():configs['EXPLORATORY_'+q]={q:v}
    configs['EXPLORATORY_pair']=NEW
    for q in PRIMARY:configs['CONDITIONAL_revert_'+q]={q:old[q]}
    # Tomorrow's four-probe adaptive plan: omit 0501, infer it from the exact +2 sum.
    for q in PRIMARY[:-1]:configs['ADAPTIVE_probe_revert_'+q]={q:old[q]}
    configs['ADAPTIVE_fifth_0526']={'test_0526':'C'}
    configs['ADAPTIVE_fifth_0526_plus_revert_0501']={'test_0526':'C','test_0501':old['test_0501']}
    manifest={}
    for name,changes in configs.items():
        assert 'test_0458' not in changes
        pred=dict(current,**changes)
        assert all(all(c in 'ABCD' and qa[q][c] for c in v) for q,v in pred.items())
        path=OUT/(name+'.csv')
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['qa_id','prediction']);w.writeheader();w.writerows({'qa_id':q,'prediction':v} for q,v in pred.items())
        diffs={}
        for label,baseline in [('332',current),('330',old)]:
            diff=[{'qa_id':q,'from':baseline[q],'to':v} for q,v in pred.items() if baseline[q]!=v]
            diffs[label]=diff
            with (OUT/(name+'.vs'+label+'.diff.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['qa_id','from','to']);w.writeheader();w.writerows(diff)
        manifest[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changes':changes,'diffs':diffs,'gate':'scored' if not changes else 'NOT approved for submission; requires additional evidence or informative score constraints'}
    (OUT/'submission_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ids=PRIMARY+list(NEW);states=[s for s in itertools.product((-1,0,1),repeat=7) if sum(s[:5])==2]
    def delta(name,s):
        changes=configs[name]
        return sum((-s[ids.index(q)] if q in PRIMARY else s[ids.index(q)]) for q in changes)
    observations={}
    for obs in a.observe:
        name,value=obs.rsplit('=',1);value=int(value);assert name in configs
        if name in observations:assert observations[name]==value,'Conflicting duplicate observations'
        observations[name]=value;states=[s for s in states if delta(name,s)==value]
    assert states,'Observations contradict the signed-effect assumptions'
    bounds={name:sorted({delta(name,s) for s in states}) for name in configs}
    best=max({'KEEP_SCORED_332':0,**observations},key=lambda k:observations.get(k,0))
    result={'delta_reference':332,'states':len(states),'observations':observations,'possible_deltas':bounds,'best_scored_configuration':best,'warning':'Never replace the scored champion with only individually identified winners. Bounds are algebraic, not probabilities. No unscored file is automatically recommended.'}
    probe_names=['ADAPTIVE_probe_revert_'+q for q in PRIMARY[:-1]]
    if all(n in observations for n in probe_names):
        # Reversion delta is -d. Total d over all five is exactly +2.
        sum_d_probed=-sum(observations[n] for n in probe_names)
        d_omitted=2-sum_d_probed
        result['adaptive_inference']={'probed':probe_names,'sum_d_probed':sum_d_probed,'forced_d_test_0501':d_omitted,'next_configuration':('ADAPTIVE_fifth_0526_plus_revert_0501' if d_omitted==-1 else 'ADAPTIVE_fifth_0526'),'reason':'If 0501 is the unique loss, combine its recovery with 0526; otherwise use the fifth slot on 0526.'}
    (OUT/'decoder_state.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
