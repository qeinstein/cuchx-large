"""Grouped-user 5-fold OOF driver for champ/pipeline.py (final video campaign).

Compares pipeline predictions vs truth on identical OOF rows; writes a
committed CSV consumed by the head-to-head comparison harness. Frozen
protocol: folds from champ.pseudotest.folds(users, 5, seed=7); thresholds
and weights are pipeline defaults (no per-fold tuning).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'champ'))

import pandas as pd  # noqa: E402

from core import load_all, make_pseudo  # noqa: E402
from pseudotest import folds  # noqa: E402
import pipeline as P  # noqa: E402


def main(tag='base', nfold=5, only_fold=None):
    tr, te, meta = load_all()
    P.caches(meta)
    users = sorted(tr.user.dropna().unique())
    rows = []
    for fi, hold in enumerate(folds(users, nfold)):
        if only_fold is not None and fi != only_fold:
            continue
        ctx = P.fit_all(tr, meta, hold)
        vis, key, aux = make_pseudo(tr, meta, hold, seed=100 + fi)
        pred, blocks, _, diag = P.solve(vis, ctx, 'oof')
        ans = dict(zip(key.qa_id, key.answer))
        for r in vis.itertuples():
            pv = pred.get(r.qa_id)
            rows.append(dict(qa_id=r.qa_id, fold=fi, source=r.source,
                             category=r.category, user=getattr(r, 'user', ''),
                             pred=pv, answer=ans[r.qa_id],
                             correct=int(pv == ans[r.qa_id])))
        print(f'  fold {fi} done ({len(hold)} users)', flush=True)
    d = pd.DataFrame(rows)
    suffix = f'{tag}_f{only_fold}' if only_fold is not None else tag
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       f'oof_pipeline_{suffix}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    print(f'wrote {out} n={len(d)}', flush=True)
    for c, g in d.groupby('category'):
        print(f'  {c:20s} {int(g.correct.sum()):5d}/{len(g):5d} = '
              f'{g.correct.mean():.4f}', flush=True)
    print(f'  {"TOTAL":20s} {int(d.correct.sum()):5d}/{len(d):5d} = '
          f'{d.correct.mean():.4f}', flush=True)


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'base'
    of = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(tag=tag, only_fold=of)
