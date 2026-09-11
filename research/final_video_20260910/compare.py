"""Head-to-head OOF comparison: specialist vs pipeline on identical rows.

Usage: compare.py <pipeline_csv> <specialist_csv> <label>
Both CSVs need qa_id,fold,category,correct. Joins on (qa_id, fold).
Reports exact accuracy, W->R/R->W/net overall + per category, per-user
and per-fold net gain, worst fold, unique sessions, paired bootstrap CI
for net gain (2000 resamples, fixed seed). Folds must align; aborts if
>5% of specialist rows lack a pipeline counterpart.
"""
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)


def main(pipe_path, spec_path, label):
    p = pd.read_csv(pipe_path)
    s = pd.read_csv(spec_path)
    m = s.merge(p[['qa_id', 'fold', 'correct']], on=['qa_id', 'fold'],
                how='left', suffixes=('', '_base'),
                indicator=True)
    miss = (m['_merge'] != 'both').mean()
    if miss > 0.05:
        raise SystemExit(f'fold alignment failed: {miss:.1%} unmatched')
    m = m[m['_merge'] == 'both'].copy()
    m['w2r'] = ((m['correct'] == 1) & (m['correct_base'] == 0)).astype(int)
    m['r2w'] = ((m['correct'] == 0) & (m['correct_base'] == 1)).astype(int)
    n = len(m)
    acc_s, acc_b = m['correct'].mean(), m['correct_base'].mean()
    W, L = int(m['w2r'].sum()), int(m['r2w'].sum())
    print(f'== {label} n={n} ==')
    print(f'specialist {m["correct"].sum()}/{n}={acc_s:.4f}  '
          f'pipeline {m["correct_base"].sum()}/{n}={acc_b:.4f}')
    print(f'W->R {W}  R->W {L}  net {W - L:+d} '
          f'({(W - L) / n:+.4f})')
    for c, g in m.groupby('category'):
        w, l = int(g['w2r'].sum()), int(g['r2w'].sum())
        print(f'  {c:18s} spec={g["correct"].mean():.4f} '
              f'base={g["correct_base"].mean():.4f} net={w - l:+d} '
              f'n={len(g)}')
    if 'user' in m.columns:
        ug = m.groupby('user').agg(w2r=('w2r', 'sum'), r2w=('r2w', 'sum'),
                                   n=('correct', 'size'))
        ug['net'] = ug['w2r'] - ug['r2w']
        print('per-user net:', ug['net'].to_dict())
        print(f'users positive: {(ug["net"] > 0).sum()}/{len(ug)}, '
              f'negative: {(ug["net"] < 0).sum()}/{len(ug)}')
    fg = m.groupby('fold').agg(w2r=('w2r', 'sum'), r2w=('r2w', 'sum'),
                               n=('correct', 'size'))
    fg['net'] = fg['w2r'] - fg['r2w']
    print('per-fold net:', fg['net'].to_dict(),
          'worst:', int(fg['net'].min()))
    d = (m['w2r'] - m['r2w']).to_numpy()
    boots = np.array([RNG.choice(d, size=n, replace=True).sum()
                      for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f'paired bootstrap 95% CI for net: [{lo:.0f}, {hi:.0f}]')
    sess = m['qa_id'].str.extract(r'(test_\d+|training_\d+)')[0]
    print(f'unique rows affected: {int(((m["w2r"] | m["r2w"]).sum()))}/{n}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
