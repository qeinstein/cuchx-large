"""Real test inference -> submission CSV.  Same solve() path as the pseudo-test.

This utility is intentionally explicit about missing evidence: ``solve`` returns
four values and may return ``None`` for a question.  Missing predictions are
filled from the immutable champion (or ``CHAMP_FALLBACK``), never from a hidden
constant such as ``A``.
"""
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
pred, blocks, pool_of, diag = P.solve(vis, ctx, 'test')
fallback_name = os.environ.get('CHAMP_FALLBACK',
                                'submission_097076_332of342_CHAMPION.csv')
fallback_path = os.path.join(ROOT, fallback_name)
if not os.path.exists(fallback_path):
    raise FileNotFoundError('CHAMP_FALLBACK does not exist: ' + fallback_path)
fallback = pd.read_csv(fallback_path).set_index('qa_id').prediction
if list(fallback.index) != list(te.qa_id):
    raise ValueError('fallback row order/key mismatch: ' + fallback_path)
sub = pd.DataFrame({'qa_id': te.qa_id,
                    'prediction': [pred.get(q) if pred.get(q) is not None else fallback.loc[q]
                                   for q in te.qa_id]})
assert len(sub) == 682 and sub.prediction.notna().all()
assert sub.prediction.map(lambda s: len(s) > 0 and all(c in 'ABCD' for c in s)).all()
out = os.path.join(ROOT, 'submission_v13_structural.csv')
sub.to_csv(out, index=False)
print('blocks:', len(blocks), 'sizes:', pd.Series([len(b) for b in blocks]).value_counts().to_dict())
print(sub.prediction.map(len).value_counts().to_dict())
base = pd.read_csv(fallback_path)
m = sub.merge(base, on='qa_id', suffixes=('_new', '_fallback')).merge(
    te[['qa_id', 'category', 'source']], on='qa_id')
print(f'\nchanges vs {fallback_name} by category:')
print(m.assign(chg=m.prediction_new != m.prediction_fallback).groupby('category').chg.agg(['sum', 'count']).to_string())
print('total changed: %d / 682' % (m.prediction_new != m.prediction_fallback).sum())
print('written', out)
