"""Pseudo-test that reproduces the real test's block-size mix INCLUDING orphan sessions,
so that the block-repair step can be validated subject-disjoint.

Real test after repair: 31 three-clip blocks, 23 two-clip blocks, 5 one-clip blocks
(59 sessions).  So pair_frac ~ 0.39 and orphan_frac ~ 0.085.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, make_pseudo, opts, fit_group_model, infer_blocks, ACTCAT
from pseudotest import folds, thin_to_pairs, thin_to_orphans
import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(tag, pair_frac=0.39, orph_frac=0.085, nfold=5):
    tr, te, meta = load_all()
    P.caches(meta)
    users = sorted(tr.user.dropna().unique())
    rows = []
    for fi, hold in enumerate(folds(users, nfold)):
        ctx = P.fit_all(tr, meta, hold)
        tr_eval = thin_to_orphans(tr, hold, orph_frac, seed=500 + fi)
        tr_eval = thin_to_pairs(tr_eval, hold, pair_frac, seed=fi)
        vis, key, aux = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
        pred, blocks, pool_of, diag = P.solve(vis, ctx, 'oof')
        # conformance of the block each question landed in, and whether it was a true session
        hau = vis[vis.source == 'HAU']
        eo = {r.idx: set(opts(r)) for _, r in hau[hau.category == 'emotion'].iterrows()}
        tb = aux['true_block']; pathof = dict(zip(vis.idx, vis.true_path))
        info = {}
        for blk in blocks:
            blk = [int(x) for x in blk]
            O = [eo[i] for i in blk if i in eo]
            I = set.intersection(*O) if len(O) > 1 else (O[0] if O else set())
            pure = len({tb.get(pathof[i]) for i in blk}) == 1
            truesz = None
            if pure:
                bi = list({tb.get(pathof[i]) for i in blk})[0]
                truesz = len(aux['blocks'][bi])
            for i in blk:
                info[i] = dict(bsz=len(blk), nI=len(I), conform=int(len(I) >= len(blk)),
                               pure=int(pure), truesz=truesz)
        ans = dict(zip(key.qa_id, key.answer))
        for r in vis.itertuples():
            pv = pred.get(r.qa_id)
            d = info.get(r.idx, {})
            rows.append(dict(qa_id=r.qa_id, fold=fi, source=r.source, category=r.category,
                             pred=pv, answer=ans[r.qa_id], correct=int(pv == ans[r.qa_id]),
                             **{k: d.get(k) for k in ('bsz', 'nI', 'conform', 'pure', 'truesz')}))
        print(f'  fold {fi}: {len(blocks)} blocks '
              f'{dict(pd.Series([len(b) for b in blocks]).value_counts())} '
              f'repaired={diag.get("blocks_repaired", 0)}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, 'champ', f'oof_{tag}.csv'), index=False)
    print(f'\n===== {tag} =====  TOTAL {d.correct.sum()}/{len(d)} = {d.correct.mean():.4f}')
    print(d.groupby('category').correct.agg(n='size', acc='mean').round(4).to_string())
    h = d[d.source == 'HAU']
    print('\n--- by block purity (block == exactly one true session) ---')
    print(h.groupby('pure').correct.agg(n='size', acc='mean').round(4).to_string())
    print('\n--- by inferred block size ---')
    print(h.groupby('bsz').correct.agg(n='size', acc='mean').round(4).to_string())
    print('\n--- category x purity ---')
    print(h.pivot_table(index='category', columns='pure', values='correct',
                        aggfunc=['size', 'mean']).round(4).to_string())
    return d


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'orphan',
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.39,
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.085)
