"""Audit conjunction constraints, object vocabulary, and donor action evidence."""
import csv,json,re
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/takeover_20260908'
def read(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig')))
def aset(s):return {a.strip().lower() for a in s.split(',') if a.strip()}
tr,te=read('training_qa.csv'),read('test_qa.csv')
base={r['qa_id']:r['prediction'] for r in read('submission_096491_330of342_CHAMPION.csv')}
def grouped(rows):
    d=defaultdict(list)
    for r in rows:d[r['path']].append(r)
    return d
def clauses(rows,preds):
    pos=defaultdict(list);neg=defaultdict(list);excluded=[]
    for r in rows:
        c=r['category'];p=preds[r['qa_id']]
        if c not in ('single','multi','combination','sequence'):continue
        for L in p:
            for a in aset(r[L]):pos[a].append(r['qa_id'])
        if c in ('single','multi'):
            for L in 'ABCD':
                if L not in p:
                    for a in aset(r[L]):neg[a].append(r['qa_id'])
        if c=='combination':
            for L in 'ABCD':
                if L!=p:excluded.append((aset(r[L]),r['qa_id'],L))
    violations=[{'kind':'positive_negative','action':a,'positive':pos[a],'negative':neg[a]} for a in pos.keys()&neg.keys()]
    violations += [{'kind':'rejected_combination_fully_present','qa_id':q,'option':L,'actions':sorted(a)} for a,q,L in excluded if a<=pos.keys()]
    return violations
result={}
for name,rows,preds in [('train',tr,{r['qa_id']:r['answer'] for r in tr}),('test',te,base)]:
    v=[{'path':path,**v} for path,g in grouped(rows).items() for v in clauses(g,preds)]
    result[name+'_conjunction_violations']=v
obj=defaultdict(Counter)
for r in tr:
    if r['category']=='object_interaction':
        action=r['path'].split('/')[1]
        obj[action].update([r[r['answer']]])
voc=json.loads((ROOT/'champ/vocab.json').read_text())['HARN2SELF']
obj_constraints=[]
for path,rows in grouped(te).items():
    singles=[r for r in rows if r['category']=='single' and r['source']=='HARn']
    objects=[r for r in rows if r['category']=='object_interaction']
    for s in singles:
        for o in objects:
            universe={o[L] for L in 'ABCD'}
            compatible={L:sum(n for act,c in obj.items() if voc.get(act)==s[L] for item,n in c.items() if item in universe) for L in 'ABC' if s[L]}
            if sum(v>0 for v in compatible.values())==1:
                p=max(compatible,key=compatible.get)
                obj_constraints.append({'qa_id':s['qa_id'],'base':base[s['qa_id']],'candidate':p,'changed':p!=base[s['qa_id']],'object_qa':o['qa_id'],'compatibility_counts':compatible})
result['object_option_universe_constraints']=obj_constraints
loo=[]
for path,rows in grouped(tr).items():
    singles=[r for r in rows if r['category']=='single' and r['source']=='HARn']
    objects=[r for r in rows if r['category']=='object_interaction']
    user=re.search(r'user\d+',path).group() if re.search(r'user\d+',path) else None
    for s in singles:
        for o in objects:
            universe={o[L] for L in 'ABCD'}
            compatible=Counter()
            for donor in tr:
                if donor['category']!='object_interaction' or '/'+str(user)+'/' in donor['path']:continue
                action=donor['path'].split('/')[1]
                if donor[donor['answer']] not in universe:continue
                for L in 'ABC':
                    if voc.get(action)==s[L]:compatible[L]+=1
            if len(compatible)==1:
                p=next(iter(compatible))
                loo.append({'qa_id':s['qa_id'],'user':user,'prediction':p,'truth':s['answer'],'correct':p==s['answer'],'support':compatible[p]})
result['object_option_leave_user_out']={'n':len(loo),'correct':sum(r['correct'] for r in loo),'rows':loo}
# Inspect full labelled action unions in identified donor sessions, not just sequences.
targets={'test_0146':('user21','2-2'),'test_0150':('user21','3-2'),'test_0151':('user21','3-2')}
donors=[]
for q,(user,session) in targets.items():
    for path,rows in grouped(tr).items():
        if path.startswith('HAU/'+user+'/'+session+'-'):
            pool=set().union(*(aset(r[L]) for r in rows if r['category'] in ('single','multi','combination','sequence') for L in r['answer']))
            donors.append({'qa_id':q,'donor_path':path,'labelled_action_union':sorted(pool)})
result['donor_action_unions']=donors
(OUT/'constraints_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
