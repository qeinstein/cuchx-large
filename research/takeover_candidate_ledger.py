"""Evidence-ranked research queue; conditional candidates are not submission approvals."""
import csv,json
from build_takeover_suite_20260908 import ROOT,OUT,BASE_SHA,build,PRIMARY
ROWS=[
 ('test_0488','C','promoted','fallback; object-option constraint','Zero cached margin. Paper-object option universe supports pages (33/33 eligible leave-user-out training questions for the general rule). Seated material handling visible. VLM C.','Clip/annotation mismatch; object vocabulary transfer; public A+B cancellation unresolved.'),
 ('test_0501','B','promoted_new','low-margin temporal error','New finding: frames show seated-to-standing transition; options explicitly include standing up. Champion mopping margin 0.604. Similar standing transition in test_0502 already answered B.','Segment boundaries or annotation inconsistency; visual interpretation is not a scored label.'),
 ('test_0477','B','promoted','zero-margin fallback','Bent posture, downward arms and implement-like motion across clip 0007 support mopping over jumping jacks/pages.','Implement detail is limited in depth; no reliable VLM support (cached D is invalid).'),
 ('test_0506','C','promoted','zero-margin fallback','Clip 0046 shows forward stepping, knee/body lowering and recovery; supports lunges over bowl wiping or jumping jacks.','Cropped legs; boundary fragment; no reliable VLM support (D is invalid).'),
 ('test_0519','C','promoted','zero-margin fallback','Clip 0062 shows seated hand-to-mouth movement; drinking is more plausible than walking/writing among options.','Small object not resolved; hand-to-mouth gesture ambiguity; cached D is invalid.'),
 ('test_0483','C','conditional_new','low-margin temporal disagreement','Clip 0017 stays seated throughout sampled frames; cached VLM chooses typing; current sitting-down has margin 3.07. Adjacent timestamp overlap with checking-time clip 0022.','Keyboard not clearly visible; transition may precede annotation crop. Do not promote without stronger evidence.'),
 ('test_0461','D','conditional','unresolved measured-zero bundle','Protocol suggests Seriously; public contribution remains part of four-row zero equation. Raw evidence against the two multi removals raises interest in emotion terms, conditionally.','Same protocol family already failed at 0458; no individual public measurement.'),
 ('test_0469','D','conditional','unresolved measured-zero bundle','Calmly is an existing protocol candidate; remains algebraically unresolved with 0461,0150,0151.','Slowly/Calmly visual distinction weak; same-user transfer is not identity proof.'),
 ('test_0430','C','conditional','joint emotion consistency','Existing block-35 Slowly duplication and protocol triad suggest Steadily.','Triplet mapping and manner interpretation are assumptions; no independent public gain.'),
 ('test_0432','A','conditional','joint emotion consistency','Existing block-35 protocol suggests Restlessly in place of Nervously.','Both agitation labels; physical scores do not identify exact adverb reliably.'),
 ('test_0341','DBCA','conditional','sequence donor disagreement','Cached recovered-pool audit gives coverage 6, margin 1, three donors; alternative paired with 0646.','Pool similarity only 0.8333; current sequence order may be correct; donor transfer is not exact clip supervision.'),
 ('test_0646','BDAC','conditional','sequence donor disagreement','Same donor-supported order alternative as 0341; useful as joint research hypothesis.','Correlated with 0341, not independent evidence or an extra discovery; no public measurement.'),
]
REJECTED=[
 ('test_0458','A','Measured public regression -1; prohibited.'),
 ('test_0146','BC','Sequence/combination exhaustiveness claim falsified; raw video includes sitting. Public effect still -d_0488.'),
 ('test_0150','AB','Raw frames show jumping-jack motion; cited donor also contains jumping jacks.'),
 ('test_0151','AB','Cited donor trials contain massaging; non-exhaustive sequence is not absence proof.'),
 ('test_0165','D','Measured public zero, but claimed action-set exhaustion is invalid; insufficient reason to change privately.'),
 ('test_0137','ACD','Wrong-block sibling argument. Target is singleton clip 0103; cached dense 0.00175 and DINO 0.00390 for stirring.'),
 ('test_0590','ABC','New high-score omission lead rejected for now: raw sink/washing activity, no clear peeling; OOF omitted-peeling examples 0/32 positives.'),
 ('test_0591','ABD','New pouring omission lead held out: sampled raw clip shows phone/headphones sequence without clear pouring; model confusion remains plausible.'),
 ('test_0158','ABC','New page omission lead held out: raw material handling unclear, 0/28 same-action omitted OOF examples positive.'),
 ('test_0436','A','Existing counter-evidence says champion already satisfies adjacent-pair assignment.'),
 ('test_0464','A','Overlapping physical distributions; mechanism shared with measured regression 0458.'),
 ('test_0517','C','Negative control: raw frames show putting on clothes; retain A despite zero margin.'),
]
def main():
    qa={r['qa_id']:r for r in csv.DictReader((ROOT/'test_qa.csv').open())}
    base={r['qa_id']:r['prediction'] for r in csv.DictReader((ROOT/'submission_096491_330of342_CHAMPION.csv').open())}
    ledger=[];manifest={}
    for rank,(q,to,status,route,evidence,failure) in enumerate(ROWS,1):
        row={'rank':rank,'qa_id':q,'from':base[q],'to':to,'category':qa[q]['category'],'status':status,'route':route,'evidence':evidence,'failure_mode':failure}
        ledger.append(row)
        name=('research_only_' if status=='conditional' else 'primary_singleton_')+q
        manifest[name]=build(name,{q:to})
    for name,changes in [('research_only_emotion_pair',{'test_0461':'D','test_0469':'D'}),('research_only_block35',{'test_0430':'C','test_0432':'A'}),('research_only_order_pair',{'test_0341':'DBCA','test_0646':'BDAC'})]:
        manifest[name]=build(name,changes)
    with (OUT/'candidate_ledger.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    (OUT/'candidate_manifest.json').write_text(json.dumps({'base_sha256':BASE_SHA,'promoted_count':5,'conditional_count':7,'conditional_files_are_not_recommended_for_automatic_use':True,'files':manifest},indent=2)+'\n')
    (OUT/'rejected_candidates.json').write_text(json.dumps([dict(qa_id=q,to=to,reason=why) for q,to,why in REJECTED],indent=2)+'\n')
    print('Built 12 ranked candidates: 5 promoted, 7 conditional; 3 conditional pair bundles. All hashes/diffs stored.')
if __name__=='__main__':main()
