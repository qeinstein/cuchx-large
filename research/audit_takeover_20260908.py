"""Read-only evidence audit; writes reproducible local research artifacts only."""
import csv, json, re
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/takeover_20260908'
OUT.mkdir(exist_ok=True)
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def actions(text):
    return {x.strip().lower() for x in text.split(',') if x.strip()}
def selected(r, pred):
    return set().union(*(actions(r.get(x,'')) for x in pred if x in 'ABCD'))
def groups(rows):
    d=defaultdict(list)
    for r in rows:
        d[r['path']].append(r)
    return d
tr,te=read('training_qa.csv'),read('test_qa.csv')
base={r['qa_id']:r['prediction'] for r in read('submission_096491_330of342_CHAMPION.csv')}
metrics=Counter(); examples=defaultdict(list)
for path,rows in groups(tr).items():
    act=[r for r in rows if r['category'] in ('single','multi','combination','sequence')]
    for r in act:
        others=set().union(*(selected(s,s['answer']) for s in act if s is not r))
        if r['category'] in ('sequence','combination'):
            own=selected(r,r['answer'])
            metrics[r['category']+'_clips']+=1
            if others-own:
                metrics[r['category']+'_incomplete']+=1
                examples[r['category']+'_incomplete'].append({'qa_id':r['qa_id'],'outside':sorted(others-own)})
        if r['category']=='single':
            neg=set().union(*(actions(r[L]) for L in 'ABCD' if L!=r['answer']))
            metrics['single_negative_checks']+=len(neg)
            metrics['single_negative_violations']+=len(neg&others)
        if r['category']=='combination':
            for L in 'ABCD':
                if L==r['answer']: continue
                neg=actions(r[L])-selected(r,r['answer'])
                metrics['combination_individual_negative_checks']+=len(neg)
                metrics['combination_individual_negative_violations']+=len(neg&others)
                if neg&others:
                    examples['combination_negative_violation'].append({'qa_id':r['qa_id'],'option':L,'present':sorted(neg&others)})

conflicts=[]; omissions=[]
for path,rows in groups(te).items():
    act=[r for r in rows if r['category'] in ('single','multi','combination','sequence')]
    for r in act:
        if r['category']!='multi': continue
        pred=base[r['qa_id']]; positive=defaultdict(list); negative=defaultdict(list)
        for s in act:
            if s is r: continue
            for a in selected(s,base[s['qa_id']]): positive[a].append(s['qa_id'])
            if s['category']=='single':
                for L in 'ABCD':
                    if L!=base[s['qa_id']]:
                        for a in actions(s[L]): negative[a].append(s['qa_id'])
        new=''
        for L in 'ABCD':
            a=r[L].lower().strip()
            include=L in pred
            if a in positive and not include:
                conflicts.append({'qa_id':r['qa_id'],'from':pred,'to':''.join(sorted(pred+L)),'kind':'positive_sibling_omission','action':a,'support':positive[a]})
            if a in negative and include:
                conflicts.append({'qa_id':r['qa_id'],'from':pred,'to':pred.replace(L,''),'kind':'single_distractor_included','action':a,'support':negative[a]})
        seq=set().union(*(selected(s,base[s['qa_id']]) for s in act if s['category']=='sequence'))
        if seq:
            outside=selected(r,pred)-seq
            if outside: omissions.append({'qa_id':r['qa_id'],'outside_sequence':sorted(outside)})

emotion=Counter(); emotion_opts=Counter()
for r in tr:
    if r['category']=='emotion':
        emotion[r[r['answer']]]+=1
        emotion_opts.update(r[L] for L in 'ABCD')
fallback=[]
margins=json.loads((ROOT/'champ/test_margins.json').read_text())
vlm=json.loads((ROOT/'vlm_predictions_cache.json').read_text())
for r in te:
    if r['source']=='HARn' and r['category']=='single' and margins.get(r['qa_id'])==0:
        fallback.append({'qa_id':r['qa_id'],'base':base[r['qa_id']],'options':{L:r[L] for L in 'ABCD'},'vlm':vlm.get(r['qa_id']),'path':r['path']})
result={'training_metrics':dict(metrics),'training_counterexamples':{k:v[:12] for k,v in examples.items()},'test_strict_conflicts':conflicts,'test_multi_outside_sequence':omissions,'zero_margin_harn':fallback,'emotion_unseen_answers':{k:v for k,v in emotion_opts.items() if not emotion[k]}}
(OUT/'structural_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
