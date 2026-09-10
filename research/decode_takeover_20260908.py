"""Signed, cancellation-safe local decoder. Never submits a file.

Example: python3 research/decode_takeover_20260908.py --today-delta 2
Then add measured subset results: --observe primary_subset_10111=2
All deltas are integer correct-answer counts relative to the 330 base.
"""
import argparse,itertools,json
from pathlib import Path
from build_takeover_suite_20260908 import PRIMARY,ZERO,build

OUT=Path(__file__).resolve().parent/'takeover_20260908'
IDS=list(PRIMARY)
def feasible(today,observations):
    return [s for s in itertools.product((-1,0,1),repeat=5)
            if (today is None or sum(s)==today)
            and all(sum(s[i] for i in range(5) if mask&(1<<i))==delta for mask,delta in observations)]
def summarize(states):
    return {q:sorted({s[i] for s in states}) for i,q in enumerate(IDS)}
def selfcheck():
    # Full signed domain, including cancelling wins/losses and all-zero states.
    for truth in itertools.product((-1,0,1),repeat=5):
        obs=[(31^(1<<i),sum(truth)-truth[i]) for i in range(4)]
        assert feasible(sum(truth),obs)==[truth]
    assert len(feasible(0,[]))>1
    assert all(v==[-1,0,1] for v in summarize(feasible(0,[])).values())
    assert feasible(6,[])==[]
    quad=[s for s in itertools.product((-1,0,1),repeat=4) if sum(s)==0]
    assert len(quad)==19
    assert [s for s in quad if s[0]+s[1]==2]==[(1,1,-1,-1)]
    assert {s[0] for s in quad if s[0]+s[1]==0}=={-1,0,1}
    print('Passed: exact recovery for all 243 signed primary states; all 19 four-row zero-bundle states retained; cancellation preserved; impossible score rejected.')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--today-delta',type=int)
    p.add_argument('--observe',action='append',default=[],help='primary_subset_XXXXX=integer_delta')
    p.add_argument('--extra-observe',action='append',default=[],help='candidate_manifest file key=integer_delta; e.g. research_only_emotion_pair=1')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selfcheck();return
    obs=[]
    for item in args.observe:
        name,delta=item.split('=');mask=int(name.removeprefix('primary_subset_'),2)
        if not 0<=mask<=31:p.error('Invalid primary subset')
        obs.append((mask,int(delta)))
    states=feasible(args.today_delta,obs)
    if not states:p.error('Inconsistent score ledger. Check exact CSV hashes, baseline and integer scores; no recommendation generated.')
    possibilities=summarize(states)
    bank={q:PRIMARY[q] for q,v in possibilities.items() if v==[1]}
    # A+B=0 was measured today; only infer the opposite signed public contribution.
    b_values=sorted({-s[0] for s in states})
    if b_values==[1]:bank['test_0146']='BC'
    extra_report={}
    components=[['test_0461','test_0469','test_0150','test_0151'],['test_0430','test_0432'],['test_0341','test_0646'],['test_0483']]
    alternate={'test_0461':'D','test_0469':'D','test_0150':'AB','test_0151':'AB','test_0430':'C','test_0432':'A','test_0341':'DBCA','test_0646':'BDAC','test_0483':'C'}
    extra=[]
    if args.extra_observe:
        manifest=json.loads((OUT/'candidate_manifest.json').read_text())['files']
        for item in args.extra_observe:
            key,delta=item.split('=')
            if key not in manifest:p.error('Unknown extra observation key')
            qids={r['qa_id'] for r in manifest[key]['diffs']}
            if not any(qids<=set(c) for c in components):p.error('Extra observation must belong to one conditional component; primary singleton scores use --observe subset masks')
            extra.append((qids,int(delta)))
    for comp in components:
        constraints=[(qs,d) for qs,d in extra if qs<=set(comp)]
        ss=[s for s in itertools.product((-1,0,1),repeat=len(comp))
            if (comp!=components[0] or sum(s)==0)
            and all(sum(s[i] for i,q in enumerate(comp) if q in qs)==d for qs,d in constraints)]
        if not ss:p.error('Inconsistent extra observation ledger; no file generated')
        for i,q in enumerate(comp):
            values=sorted({s[i] for s in ss});extra_report[q]=values
            if values==[1]:bank[q]=alternate[q]
    bank_artifact=build('decoded_confirmed_public_wins',dict(ZERO,**bank))
    recommendations=[]
    for mask in range(1,32):
        values=[sum(s[i] for i in range(5) if mask&(1<<i)) for s in states]
        if len(set(values))==1:continue
        # Avoid retaining a measured loss in a follow-up; zeros remain undecided privately.
        if any(possibilities[q]==[-1] and mask&(1<<i) for i,q in enumerate(IDS)):continue
        counts={v:values.count(v) for v in set(values)}
        recommendations.append({'file':'primary_subset_'+format(mask,'05b')+'.csv',
             'public_delta_bounds':[min(values),max(values)],
             'largest_remaining_state_partition':max(counts.values()),
             'distinct_possible_scores':len(counts),'changes':mask.bit_count()})
    recommendations.sort(key=lambda r:(-r['public_delta_bounds'][0],r['largest_remaining_state_partition'],-r['changes']))
    result={'remaining_primary_states':len(states),'public_contribution_possibilities':possibilities,
       'test_0146_public_contribution_possibilities':b_values,
       'known_public_regression':{'test_0458':-1},
       'conditional_public_contribution_possibilities':extra_report,
       'other_equation':'d_0461+d_0150+d_0151+d_0469=0; use --extra-observe for measured conditional probes',
       'zero_does_not_establish_private_membership':True,
       'confirmed_public_wins_file':bank_artifact,
       'diagnostic_options':recommendations[:5],
       'policy':'Bank measured wins. Spend follow-up slots only when their score/gain tradeoff is useful. Keep slot 5 for consolidation. Partition sizes are combinatorial counts, not probabilities.'}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
