"""Static/test-side audit of every explicit fallback in champ/pipeline.py.

The pipeline's final OOF ledger is not available in this checkout, so this file does not
invent OOF branch accuracies.  It records exact branch definitions and test-side signatures
that can be observed without labels; analogous OOF evidence is linked separately.
"""
import os,json
import pandas as pd
from champ.core import load_all

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,'research/post332_20260908')

def main():
 tr,te,meta=load_all(); tm=json.load(open(os.path.join(ROOT,'champ/test_margins.json')))
 te=te.copy(); te['lm']=te.path.str.extract(r'(LM_test_\d+)')[0]
 ids=set(meta[meta.kind=='test'].qa_path); te['meta_present']=te.lm.isin(ids)
 te['margin']=te.qa_id.map(tm); te['margin_zero']=te.margin.eq(0)
 te['likely_fallback']=((te.source=='HARn') & ((~te.meta_present)|te.margin_zero))
 te[['qa_id','source','category','lm','meta_present','margin','margin_zero','likely_fallback','A','B','C','D']].to_csv(os.path.join(OUT,'fallback_test_signatures.csv'),index=False)
 branches={
  'test_v8_fill': 'final.run_test fills solve(None) with submission_v8; label unavailable at test time',
  'sequence_no_evidence': 'no dense logits or unmapped option in solve sequence branch',
  'harn_no_evidence': 'HARn clip has sensor metadata but none of classifier/dino/aggregate/pool evidence',
  'harn_flat_scores': 'single score vector ties exactly after allowed-action filtering',
  'object_flat_prior': 'object candidate prior tuples are all equal',
  'never_predicted': 'question was not visited by any branch',
  'harn_filter_empty': 'allowed-action filter empty; solver explicitly retries unfiltered (not a default answer)',
 }
 counts={}
 for name in ['source','category','meta_present','margin_zero','likely_fallback']:
  counts[name]=te[name].value_counts(dropna=False).to_dict()
 counts['likely_rows']=te[te.likely_fallback][['qa_id','source','category','lm','margin']].to_dict('records')
 out={'branches':branches,'test_counts':counts,
      'known_oof_analogs':{
       'object_flat_prior_vs_exact_template':'exact_visible_object_fallback_oof_20260905.csv',
       'harn_weak_sensor_vs_exact_template':'harn_template_sensor_gate_oof_20260905.csv',
      }}
 json.dump(out,open(os.path.join(OUT,'fallback_static_audit.json'),'w'),indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
