"""Build non-submission test variants for exact-template disagreements.

These files are intentionally outside the adaptive manifest: the rows have not cleared
the evidence gate and must not silently enter tomorrow's five slots.
"""
import csv,hashlib,json,os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,'research/post332_20260908')
BASE=os.path.join(ROOT,'submissions/submission_097076_332of342_CHAMPION.csv')
VAR={'RESEARCH_template_0478_B.csv':{'test_0478':'B'},
     'RESEARCH_template_0507_C.csv':{'test_0507':'C'},
     'RESEARCH_template_0527_A.csv':{'test_0527':'A'}}
def main():
 rows=list(csv.DictReader(open(BASE))); out=[]
 for fn,chg in VAR.items():
  pred={r['qa_id']:r['prediction'] for r in rows}; pred.update(chg)
  p=os.path.join(OUT,fn)
  with open(p,'w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['qa_id','prediction']); w.writeheader(); w.writerows({'qa_id':q,'prediction':v} for q,v in pred.items())
  dp=os.path.join(OUT,fn.replace('.csv','.vs332.diff.csv'))
  with open(dp,'w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['qa_id','from','to']);w.writeheader()
   w.writerows({'qa_id':q,'from':next(r['prediction'] for r in rows if r['qa_id']==q),'to':v} for q,v in chg.items())
  out.append({'file':fn,'changes':chg,'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest(),'diff_vs_332':dp})
 json.dump(out,open(os.path.join(OUT,'research_variants_manifest.json'),'w'),indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
