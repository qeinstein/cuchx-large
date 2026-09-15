"""Assemble validated submission CSVs from ranked DINO disagreements.

Usage:
  python3 research/t2_dino/make_flips.py --guess research/t2_dino/test_dino_guesses.csv \
      --n 1 --tag DINO_TOP1 --out submissions/
Reads champ 334, applies top-N disagreements (single/multi/combination only,
margin-ranked), validates 682-row format + qa_id order, writes CSV + diff log.
"""
import argparse
import os

import pandas as pd

ROOT = '/home/fluxx/Workspace/cuchx-large'
BASE = os.path.join(ROOT, 'submissions', 'submission_097660_334of342_CHAMPION.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guess', required=True)
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--min-margin', type=float, default=0.0)
    ap.add_argument('--cats', default='single,multi,combination')
    ap.add_argument('--out', default=os.path.join(ROOT, 'submissions'))
    a = ap.parse_args()
    g = pd.read_csv(a.guess)
    g = g[(g.disagree == 1) & (g.margin >= a.min_margin)
          & g.cat.isin(a.cats.split(','))].sort_values('margin', ascending=False)
    picks = g.head(a.n)
    if len(picks) < a.n:
        print('WARN: only %d/%d flips available at margin>=%.3f'
              % (len(picks), a.n, a.min_margin), flush=True)
    base = pd.read_csv(BASE)
    te = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
    assert list(base.qa_id) == list(te.qa_id) and len(base) == 682, 'base mismatch'
    sub = base.set_index('qa_id').prediction.astype(str)
    for r in picks.itertuples():
        assert sub.loc[r.qa_id] == r.champ, 'champ drift on %s' % r.qa_id
        sub.loc[r.qa_id] = r.dino
    out = pd.DataFrame({'qa_id': base.qa_id,
                        'prediction': [sub.loc[q] for q in base.qa_id]})
    assert (out.qa_id == te.qa_id).all() and len(out) == 682
    assert out.prediction.notna().all() and (out.prediction.str.len() > 0).all()
    fn = os.path.join(a.out, 'CAND_%s_top%d.csv' % (a.tag, len(picks)))
    out.to_csv(fn, index=False)
    print('wrote %s' % fn, flush=True)
    print(picks[['qa_id', 'cat', 'champ', 'dino', 'margin']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
