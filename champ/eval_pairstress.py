"""Pseudo-test with the held-out sessions thinned to two trials, to match the real test mix.

The real test set is 34 three-trial blocks + 21 two-trial blocks (verified independently:
in clips 164-208 the emotion option-set intersection graph is a perfect matching of 21
adjacent pairs, with 0 contiguous triangles, against 31 in the three-trial zone 65-163).
45 of the 144 test HAU clips therefore sit in two-trial blocks, and every category that
depends on the session block or its action pool is solved there from 2/3 of the evidence.
This run measures that cost per category, and tags every question with its block size.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, make_pseudo
from pseudotest import folds, thin_to_pairs
import pipeline as P


def main(tag='pairstress', pair_frac=0.38, nfold=5, policy=None):
    tr, te, meta = load_all()
    P.caches(meta)
    users = sorted(tr.user.dropna().unique())
    rows = []
    print(f'config: pair_frac={pair_frac} policy={policy or "uniform"} '
          f'W_SLOT={P.W_SLOT} SLOT_MODEL={os.environ.get("CHAMP_SLOT_MODEL","gbm")} '
          f'ADJ_ONLY={os.environ.get("CHAMP_SLOT_ADJ_ONLY","0")} '
          f'REPAIR={P.W_REPAIR}', flush=True)
    for fi, hold in enumerate(folds(users, nfold)):
        ctx = P.fit_all(tr, meta, hold)
        tr_eval = thin_to_pairs(tr, hold, pair_frac, seed=fi, policy=policy) \
            if pair_frac else tr
        vis, key, aux = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
        pred, blocks, pool_of, diag = P.solve(vis, ctx, 'oof')
        bsz = {}
        for blk in blocks:
            for i in blk: bsz[i] = len(blk)
        ans = dict(zip(key.qa_id, key.answer))
        for r in vis.itertuples():
            pv = pred.get(r.qa_id)
            rows.append(dict(qa_id=r.qa_id, fold=fi, source=r.source, category=r.category,
                             pred=pv, answer=ans[r.qa_id], bsz=bsz.get(r.idx, np.nan),
                             correct=int(pv == ans[r.qa_id])))
        print(f'  fold {fi} done ({len(hold)} users, {len(blocks)} blocks, '
              f'sizes {dict(pd.Series([len(b) for b in blocks]).value_counts())})', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'oof_{tag}.csv'), index=False)
    print(f'\n===== {tag} (pair_frac={pair_frac}) =====')
    print(d.groupby(['source', 'category']).correct.agg(n='size', ok='sum', acc='mean').round(4).to_string())
    print(f'\nTOTAL {d.correct.sum()}/{len(d)} = {d.correct.mean():.4f}')
    print('\n--- by inferred block size ---')
    print(d.groupby(['bsz', 'category']).correct.agg(n='size', acc='mean').round(4).to_string())
    print('\n--- category accuracy, bsz=2 vs bsz=3 ---')
    p = d[d.bsz.isin([2, 3])].pivot_table(index='category', columns='bsz', values='correct',
                                          aggfunc=['size', 'mean'])
    print(p.round(4).to_string())
    return d


if __name__ == '__main__':
    main(tag=sys.argv[1] if len(sys.argv) > 1 else 'pairstress',
         pair_frac=float(sys.argv[2]) if len(sys.argv) > 2 else 0.38,
         policy=sys.argv[3] if len(sys.argv) > 3 else None)
