"""Hybrid submission: structural pipeline on HAU questions, v8 on HARn questions.
The split is chosen per (source, category) cell by pseudo-test OOF, not by guesswork."""
import os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
te = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
new = pd.read_csv(os.path.join(ROOT, 'submission_v12_structural.csv'))
v8 = pd.read_csv(os.path.join(ROOT, 'submission_v8.csv'))
m = te[['qa_id', 'source', 'category']].merge(new, on='qa_id').merge(
    v8, on='qa_id', suffixes=('_new', '_v8'))
# per-cell choice from pseudo-test OOF: v12 wins every cell except HARn object
# (114/133 = 0.8571 vs v8's 117/133 = 0.8797)
use_v8 = m.category == 'object_interaction'
m['prediction'] = m.prediction_new.where(~use_v8, m.prediction_v8)
sub = m[['qa_id', 'prediction']]
# ---- format validation
assert len(sub) == 682, len(sub)
assert sub.qa_id.is_unique and list(sub.qa_id) == list(te.qa_id)
assert sub.prediction.notna().all()
bad = []
for _, r in m.iterrows():
    p = str(r.prediction)
    if not p or any(c not in 'ABCD' for c in p) or len(set(p)) != len(p):
        bad.append((r.qa_id, p))
    if r.category in ('single', 'emotion', 'combination', 'object_interaction') and len(p) != 1:
        bad.append((r.qa_id, p, r.category))
    if r.category == 'sequence' and sorted(p) != list('ABCD'):
        bad.append((r.qa_id, p, r.category))
    if r.category == 'multi' and not (1 <= len(p) <= 4):
        bad.append((r.qa_id, p, r.category))
assert not bad, bad[:10]
out = os.path.join(ROOT, 'submission_v12.csv')
sub.to_csv(out, index=False)
print('rows', len(sub), '| format checks passed')
print('answer-length histogram by category:')
print(m.assign(L=m.prediction.map(len)).pivot_table(index='category', columns='L',
      values='qa_id', aggfunc='count', fill_value=0).to_string())
chg = (m.prediction != m.prediction_v8)
print('\nchanges vs v8: %d / 682' % chg.sum())
print(m.assign(c=chg).groupby(['source', 'category']).c.agg(['sum', 'count']).to_string())
print('\nwritten', out)
