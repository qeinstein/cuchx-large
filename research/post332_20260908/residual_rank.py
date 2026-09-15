"""Rank test rows for manual/active-learning review.

Scores are screening priors, not calibrated probabilities: no test labels exist.  The
only calibrated piece is the five-row signed-state uncertainty induced by the measured
aggregate +2 (uniform over the 30 algebraic states solely for probe ordering).
"""
import os,json
import pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,'research/post332_20260908')
def main():
 te=pd.read_csv(os.path.join(ROOT,'test_qa.csv')); sub=pd.read_csv(os.path.join(ROOT,'submissions/submission_097076_332of342_CHAMPION.csv')).set_index('qa_id').prediction
 tm=json.load(open(os.path.join(ROOT,'champ/test_margins.json'))); te['champion']=te.qa_id.map(sub); te['margin']=te.qa_id.map(tm)
 te['lm']=te.path.str.extract(r'(LM_test_\d+)')[0]
 # Evidence-based screening, intentionally coarse and explicitly labelled heuristic.
 te['p_wrong_screen']=0.0292; te['alternative']=''; te['rationale']='no surviving targeted residual evidence'; te['evidence_grade']='prior'
 for q in ['test_0488','test_0477','test_0506','test_0519','test_0501']:
  te.loc[te.qa_id==q,['p_wrong_screen','rationale','evidence_grade']]=[0.467,'measured five-row +2 leaves sign unresolved','algebraic-uncertainty']
 te.loc[te.qa_id=='test_0488','alternative']='A (probe only; direct label still unresolved)'
 te.loc[te.qa_id=='test_0477','alternative']='A (probe only; direct evidence favors B)'
 te.loc[te.qa_id=='test_0506','alternative']='A (probe only; direct evidence favors C)'
 te.loc[te.qa_id=='test_0519','alternative']='A (probe only; direct evidence favors C)'
 te.loc[te.qa_id=='test_0501','alternative']='A (probe only; direct evidence favors B)'
 for q,p,a,r in [('test_0483',.35,'C','desk/posture ambiguity; independent models disfavor flip'),('test_0526',.40,'C','phone/typing ambiguity; dense favors phone'),('test_0527',.25,'A','object template purity .60') ,('test_0478',.12,'B','single exact-template disagreement; ExtraTrees agrees C'),('test_0507',.08,'C','single exact-template disagreement; sensor/ExtraTrees agree A')]:
  te.loc[te.qa_id==q,['p_wrong_screen','alternative','rationale','evidence_grade']]=[p,a,r,'research-only']
 # Other no-metadata clips are a weakly elevated review cohort, not candidate flips.
 no_meta=~te.lm.isin(set(pd.read_csv(os.path.join(ROOT,'champ/meta.csv')).query("kind=='test'").qa_path))
 mask=no_meta & (te.p_wrong_screen<.1); te.loc[mask,['p_wrong_screen','rationale','evidence_grade']]=[.10,'no-metadata fallback cohort; raw sweep found incumbent-compatible action','weak-cohort']
 te=te.sort_values(['p_wrong_screen','qa_id'],ascending=[False,True]).reset_index(drop=True); te.insert(0,'rank',range(1,len(te)+1))
 te.to_csv(os.path.join(OUT,'residual_rank.csv'),index=False)
 print(te.head(30)[['rank','qa_id','category','champion','p_wrong_screen','alternative','rationale','evidence_grade']].to_string(index=False))
 print('wrote',len(te),'rows; scores are screening priors, not labels')
if __name__=='__main__': main()
