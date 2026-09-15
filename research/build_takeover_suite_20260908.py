"""Build local CSVs, exact diffs and hashes. No submission/network capability."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/takeover_20260908'
BASE=ROOT/'submissions/submission_096491_330of342_CHAMPION.csv'
BASE_SHA='e42cde96bafb108b8c16bcd97e5deb2af32c1657a089351ada8a72a95ed805cd'
ZERO={'test_0444':'B','test_0426':'B','test_0647':'DCBA','test_0206':'CD'}
PRIMARY={'test_0488':'C','test_0477':'B','test_0506':'C','test_0519':'C','test_0501':'B'}
def build(name,changes):
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
    rows=list(csv.DictReader(BASE.open()))
    qa={r['qa_id']:r for r in csv.DictReader((ROOT/'test_qa.csv').open())}
    assert len(rows)==682 and len({r['qa_id'] for r in rows})==682
    assert [r['qa_id'] for r in rows]==list(qa)
    assert 'test_0458' not in changes, 'Verified public regression is forbidden'
    assert set(changes)<=set(qa)
    diffs=[]
    for r in rows:
        qid=r['qa_id']
        if qid in changes:
            new=changes[qid];old=r['prediction'];q=qa[qid]
            assert old!=new and len(set(new))==len(new)
            assert all(L in 'ABCD' and q[L] for L in new)
            if q['category']=='sequence':assert set(new)=={L for L in 'ABCD' if q[L]}
            elif q['category']=='multi':assert new==''.join(sorted(new))
            else:assert len(new)==1
            diffs.append({'qa_id':qid,'from':old,'to':new})
            r['prediction']=new
    OUT.mkdir(exist_ok=True)
    path=OUT/(name+'.csv')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['qa_id','prediction']);w.writeheader();w.writerows(rows)
    with (OUT/(name+'.diff.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['qa_id','from','to']);w.writeheader();w.writerows(diffs)
    return {'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'diffs':diffs}
if __name__=='__main__':
    result=build('TODAY_RECOMMENDED',dict(ZERO,**PRIMARY))
    (OUT/'today_manifest.json').write_text(json.dumps({'base_sha256':BASE_SHA,'public_delta_bounds':[-5,5],'submission_performed':False,**result},indent=2)+'\n')
    manifest={}
    ids=list(PRIMARY)
    for mask in range(32):
        changes={q:PRIMARY[q] for i,q in enumerate(ids) if mask&(1<<i)}
        name='primary_subset_'+format(mask,'05b')
        manifest[name]=build(name,dict(ZERO,**changes))
    (OUT/'subset_manifest.json').write_text(json.dumps({'base_sha256':BASE_SHA,'bit_order_low_to_high':ids,'files':manifest},indent=2)+'\n')
    print(json.dumps(result,indent=2))
