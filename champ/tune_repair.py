"""Offline tuning of the block repair: how much block purity does it recover?

Cheap - reconstructs the same orphan-injected pseudo-test blocks per fold (make_pseudo +
infer_blocks, no model fitting) and scores purity before and after repair for a range of the
singleton penalty.  Purity matters because, measured on the orphan-injected pseudo-test,
questions in impure blocks score 0.6117 against 0.9240 in pure ones.
"""
import os, sys, itertools, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, make_pseudo, fit_group_model, infer_blocks, opts
from pseudotest import folds, thin_to_pairs, thin_to_orphans
import repair as RP

tr, te, meta = load_all()
users = sorted(tr.user.dropna().unique())
gmod_all = RP.fit_gap_model(tr, meta)
FD = []
for fi, hold in enumerate(folds(users, 5)):
    trn = tr[~tr.user.isin(hold)]
    gm = fit_group_model(trn)
    gmod = RP.fit_gap_model(trn, meta)
    tr_eval = thin_to_orphans(tr, hold, 0.085, seed=500 + fi)
    tr_eval = thin_to_pairs(tr_eval, hold, 0.39, seed=fi)
    vis, key, aux = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
    blocks = infer_blocks(vis, gm)
    pathof = dict(zip(vis.idx, vis.true_path))
    t0 = {i: meta.set_index('qa_path').loc[pathof[i], 't0']
          for i in vis[vis.source == 'HAU'].idx.unique() if pathof[i] in set(meta.qa_path)}
    nq = vis[vis.source == 'HAU'].groupby('idx').size().to_dict()
    FD.append((vis, blocks, aux, pathof, t0, gm, gmod, nq))
    print(f'  fold {fi} prepared ({len(blocks)} blocks)', flush=True)

def purity(blocks, aux, pathof, nq):
    """questions in blocks that are exactly one true session / total, and exact-match count."""
    tb = aux['true_block']; tsz = [len(b) for b in aux['blocks']]
    q_pure = q_tot = 0; exact = 0
    for blk in blocks:
        blk = [int(x) for x in blk]
        ids = {tb.get(pathof[i]) for i in blk}
        n = sum(nq.get(i, 0) for i in blk)
        q_tot += n
        if len(ids) == 1:
            bi = ids.pop()
            if len(blk) == tsz[bi]:
                q_pure += n; exact += 1
    return q_pure, q_tot, exact, len(blocks)

print('\n=== block purity before / after repair, by singleton penalty ===')
print(f'  {"pen":>6s} {"blocks":>7s} {"exact":>6s} {"q_pure":>7s} {"q_tot":>6s} {"frac":>7s}')
base = [0, 0, 0, 0]
for vis, blocks, aux, pathof, t0, gm, gmod, nq in FD:
    p, t, e, nb = purity(blocks, aux, pathof, nq)
    base[0] += p; base[1] += t; base[2] += e; base[3] += nb
print(f'  {"OFF":>6s} {base[3]:7d} {base[2]:6d} {base[0]:7d} {base[1]:6d} {base[0]/base[1]:7.4f}')
for pen in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
    RP.SINGLETON_PEN = pen
    tot = [0, 0, 0, 0]; nrep = 0
    for vis, blocks, aux, pathof, t0, gm, gmod, nq in FD:
        nb2, log = RP.repair(blocks, vis, gm, gmod, t0)
        nrep += len(log)
        p, t, e, n = purity(nb2, aux, pathof, nq)
        tot[0] += p; tot[1] += t; tot[2] += e; tot[3] += n
    print(f'  {pen:6.1f} {tot[3]:7d} {tot[2]:6d} {tot[0]:7d} {tot[1]:6d} {tot[0]/tot[1]:7.4f}'
          f'   (repaired {nrep} blocks)')
