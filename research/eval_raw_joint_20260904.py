"""EXP-EMO-RAWJOINT-001: integrate raw temporal pair evidence in the joint solver.

The existing raw-motion probe measured a one-transposition override on top of a fixed
champion and was negative on the champion's disagreement slice.  This experiment tests
the narrower, stronger mechanism suggested by that result: use the same fold-fitted raw
pair head as an additive term while enumerating every valid block assignment.

This is HAU-only by design.  It reproduces the production HAU context (subject-disjoint
manner model, action-pool model, inferred blocks, and emopair term) while omitting the
unrelated HARn action classifier.  It never writes a submission or changes the default
pipeline because the callback is opt-in.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "champ"))

import emopair as EP  # noqa: E402
import pipeline as P  # noqa: E402
import pool as PL  # noqa: E402
from core import (  # noqa: E402
    fit_group_model,
    fit_manner,
    infer_blocks,
    load_all,
    make_pseudo,
    mgroup,
    solve_emotion,
)
from pseudotest import folds, thin_to_pairs  # noqa: E402
from emolab.probe_relative_raw import (  # noqa: E402
    build_descriptors,
    fit_pair_head,
    pair_vector,
)


OUT = os.path.join(ROOT, "research", "raw_joint_oof_20260904.csv")
# Keep the first pass bounded: a matched zero-weight control and a moderate integrated
# term are enough to falsify the mechanism before spending time on calibration sweeps.
WEIGHTS = [0.0, 0.2]


def _raw_callback(head, store):
    cache = {}

    def score(pa, pb, ma, mb, i, j, k):
        if ma == mb or mgroup(ma) == mgroup(mb):
            return 0.0
        key = (pa, pb, ma, mb, i, j, k)
        if key in cache:
            return cache[key]
        va, vb = store.vectors.get(pa), store.vectors.get(pb)
        if va is None or vb is None:
            return 0.0
        x = pair_vector(va, vb, ma, mb, i, j, k)
        p = np.clip(head.probability(x), 1e-5, 1.0 - 1e-5)
        out = float(np.log(p / (1.0 - p)))
        cache[key] = out
        return out
    return score


def fit_hau_context(tr, meta, hold, tpool, statcache):
    train = tr[~tr.user.isin(hold)]
    train_users = sorted(train.user.dropna().unique())
    PL.fit_cooc(train)
    blocks_by_user = P.training_blocks(tr, train_users, tpool)
    clfp, colsp = PL.fit_pool_model(
        train, meta, blocks_by_user, statcache, augment_subblocks=True)
    return train, fit_manner(train, meta), PL.make_scorer(clfp, colsp)


def block_pools(vis, blocks, scorer, statcache):
    out = {}
    for blk in blocks:
        _, bd = PL.solve_block(vis, blk, statcache, scorer, "oof")
        if bd:
            for b in blk:
                out[b] = set(bd["pool"])
    return out


def main():
    # The raw descriptor cache is label-free and is built once outside the fold loop.
    tr, te, meta = load_all()
    P.caches(meta)
    store = build_descriptors(meta)
    tpool = P.true_pool(tr)
    users = sorted(tr.user.dropna().unique())
    all_rows = []

    for fi, hold in enumerate(folds(users, 5)):
        train, mm, scorer = fit_hau_context(
            tr, meta, hold, tpool, P.statcache_view("oof"))
        mm["pm"] = EP.fit(train, meta, mm, pool_of=tpool)
        head = fit_pair_head(train, store, include_motion=True)
        raw = _raw_callback(head, store)

        evaluation = thin_to_pairs(tr, hold, 0.38, seed=fi, policy="ends")
        vis, key, _ = make_pseudo(evaluation, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, fit_group_model(train))
        pool_of = block_pools(vis, blocks, scorer, P.statcache_view("oof"))
        ans = dict(zip(key.qa_id, key.answer))
        eq = vis[vis.category.eq("emotion")]

        for w in WEIGHTS:
            pred, _ = solve_emotion(
                vis, blocks, mm, w_phys=1.0, w_pos=1.0, w_pair=1.0,
                pool_of={tuple(b): pool_of.get(b[0], set()) for b in blocks},
                raw_pair_score=raw if w else None, w_raw_pair=w)
            for r in eq.itertuples():
                p = pred.get(r.qa_id)
                all_rows.append(dict(fold=fi, weight=w, qa_id=r.qa_id,
                                     bsz=len(next((b for b in blocks if r.idx in b), [])),
                                     pred=p, answer=ans[r.qa_id], correct=int(p == ans[r.qa_id])))
        print(f"fold {fi} done: held={hold} blocks={len(blocks)}", flush=True)

    d = pd.DataFrame(all_rows)
    d.to_csv(OUT, index=False)
    summary = (d.groupby(["weight", "bsz"], dropna=False).correct
               .agg(n="size", correct="sum", acc="mean").reset_index())
    print(summary.to_string(index=False))
    print(d.groupby("weight").correct.agg(n="size", correct="sum", acc="mean")
          .to_string())


if __name__ == "__main__":
    main()
