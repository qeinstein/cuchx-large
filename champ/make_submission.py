"""Real test inference -> submission CSV.  Same solve() path as the pseudo-test."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, real_test_view
import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tr, te, meta = load_all()
P.caches(meta)
ctx = P.fit_all(tr, meta, hold_users=[], split_train='oof')   # fit on ALL subjects
vis = real_test_view(te)
pred, blocks, pool_of = P.solve(vis, ctx, 'test')
sub = pd.DataFrame({'qa_id': te.qa_id,
                    'prediction': [pred.get(q, 'A') for q in te.qa_id]})
assert len(sub) == 682 and sub.prediction.notna().all()
assert sub.prediction.map(lambda s: len(s) > 0 and all(c in 'ABCD' for c in s)).all()
out = os.path.join(ROOT, 'submission_v13_structural.csv')
sub.to_csv(out, index=False)
print('blocks:', len(blocks), 'sizes:', pd.Series([len(b) for b in blocks]).value_counts().to_dict())
print(sub.prediction.map(len).value_counts().to_dict())
v8 = pd.read_csv(os.path.join(ROOT, 'submission_v8.csv'))
m = sub.merge(v8, on='qa_id', suffixes=('_new', '_v8')).merge(
    te[['qa_id', 'category', 'source']], on='qa_id')
print('\nchanges vs v8 by category:')
print(m.assign(chg=m.prediction_new != m.prediction_v8).groupby('category').chg.agg(['sum', 'count']).to_string())
print('total changed: %d / 682' % (m.prediction_new != m.prediction_v8).sum())
print('written', out)
