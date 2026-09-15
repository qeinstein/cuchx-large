"""Real-test inference for a named mechanism configuration, plus a diff against the champion.

Usage:  python champ/make_candidate.py <out_name>
Environment flags select the mechanisms (all default to champion behaviour):
    CHAMP_REPAIR=1        split inferred blocks that cannot be one session (champ/repair.py)
    CHAMP_W_SLOT=<w>      learned prior over which protocol slots a two-clip block contains
    CHAMP_EMO_DTW=1       DTW warping-path features in the pairwise manner discriminator
    CHAMP_LOGITS=<file>   which cached dense frame log-probs to read
Mechanism S (sequence joint order decoding) is layered separately by seqlab/apply_test.py,
because it is a selective override rather than a pipeline setting.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, real_test_view
import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAMP = os.path.join(ROOT, os.environ.get(
    'CHAMP_BASE', 'submissions/submission_097076_332of342_CHAMPION.csv'))


def main(name):
    tr, te, meta = load_all()
    P.caches(meta)
    print('flags: REPAIR=%s W_SLOT=%s EMO_DTW=%s LOGITS=%s' % (
        P.W_REPAIR, P.W_SLOT, os.environ.get('CHAMP_EMO_DTW', '0'),
        os.environ.get('CHAMP_LOGITS', 'dense_logits.npz')), flush=True)
    ctx = P.fit_all(tr, meta, hold_users=[], split_train='oof')   # fit on ALL subjects
    vis = real_test_view(te)
    pred, blocks, pool_of, diag = P.solve(vis, ctx, 'test')
    fallback = pd.read_csv(CHAMP).set_index('qa_id').prediction
    if list(fallback.index) != list(te.qa_id):
        raise ValueError('fallback row order/key mismatch: ' + CHAMP)
    sub = pd.DataFrame({'qa_id': te.qa_id,
                        'prediction': [pred.get(q) if pred.get(q) is not None
                                       else fallback.loc[q] for q in te.qa_id]})
    assert len(sub) == 682 and sub.prediction.notna().all()
    assert sub.prediction.map(lambda s: len(s) > 0 and all(c in 'ABCD' for c in s)).all()
    out = os.path.join(ROOT, f'{name}.csv')
    sub.to_csv(out, index=False)
    print('blocks:', len(blocks), 'sizes:',
          pd.Series([len(b) for b in blocks]).value_counts().to_dict(),
          'repaired:', diag.get('blocks_repaired', 0))
    ch = pd.read_csv(CHAMP)
    m = sub.merge(ch, on='qa_id', suffixes=('_new', '_ch')).merge(
        te[['qa_id', 'category', 'source']], on='qa_id')
    m['chg'] = m.prediction_new != m.prediction_ch
    print('\nchanges vs the 0.92105 champion, by category:')
    print(m.groupby(['source', 'category']).chg.agg(['sum', 'count']).to_string())
    print('total changed: %d / 682' % int(m.chg.sum()))
    m[m.chg][['qa_id', 'source', 'category', 'prediction_ch', 'prediction_new']].to_csv(
        os.path.join(ROOT, 'champ', f'diff_{name}.csv'), index=False)
    print('written', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'submission_candidate')
