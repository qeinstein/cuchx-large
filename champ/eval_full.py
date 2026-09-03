"""Full 5-fold pseudo-test evaluation, integer counts, ablations, wins/losses vs v7."""
import os, sys, json
import numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, make_pseudo
from pseudotest import folds, thin_to_pairs
import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(nfold=5, tag='full', pair_frac=0.0):
    tr, te, meta = load_all()
    P.caches(meta)
    users = sorted(tr.user.dropna().unique())
    rows = []
    for fi, hold in enumerate(folds(users, nfold)):
        ctx = P.fit_all(tr, meta, hold)
        tr_eval = thin_to_pairs(tr, hold, pair_frac, seed=fi) if pair_frac else tr
        vis, key, aux = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
        pred, blocks, _ = P.solve(vis, ctx, 'oof')
        ans = dict(zip(key.qa_id, key.answer))
        for r in vis.itertuples():
            pv = pred.get(r.qa_id, 'A')
            rows.append(dict(qa_id=r.qa_id, fold=fi, source=r.source, category=r.category,
                             pred=pv, answer=ans[r.qa_id], correct=int(pv == ans[r.qa_id])))
        print(f'  fold {fi} done ({len(hold)} users)', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, 'champ', f'oof_{tag}.csv'), index=False)
    print(f'\n  ===== PSEUDO-TEST OOF ({tag}) =====')
    d['blocksize'] = d.qa_id.map(dict(zip(d.qa_id, d.qa_id)))  # placeholder
    v7 = pd.read_csv(os.path.join(ROOT, 'oof_v7_final.csv'))[['qa_id', 'correct', 'category']]
    v7 = v7.rename(columns={'correct': 'c7'})
    m = d.merge(v7[['qa_id', 'c7']], on='qa_id', how='left')
    for c, g in m.groupby('category'):
        print(f'    {c:20s} {g.correct.sum():5d}/{len(g):5d} = {g.correct.mean():.4f}'
              f'   (v7 {int(g.c7.sum()):5d} = {g.c7.mean():.4f})')
    print(f'    {"TOTAL":20s} {m.correct.sum():5d}/{len(m):5d} = {m.correct.mean():.4f}'
          f'   (v7 {int(m.c7.sum()):5d} = {m.c7.mean():.4f})')
    w = ((m.correct == 1) & (m.c7 == 0)).sum(); l = ((m.correct == 0) & (m.c7 == 1)).sum()
    print(f'    wins vs v7 {w}   losses vs v7 {l}   net {w - l:+d}')
    print('\n    per-fold total:')
    for f, g in m.groupby('fold'):
        print(f'      fold {f}: {g.correct.sum():4d}/{len(g):4d} = {g.correct.mean():.4f}'
              f'  (v7 {int(g.c7.sum()):4d} = {g.c7.mean():.4f})')
    return m


if __name__ == '__main__':
    main(tag=sys.argv[1] if len(sys.argv) > 1 else 'full',
         pair_frac=float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
