"""Paired subject-disjoint audit of the latent full-session pool target.

The historical evaluator is intentionally general and refits every model for every
arm.  Here the two arms share all fitted components and the exact pseudo-test view;
only the pool-membership scorer is refit with the corrected sub-block target.
"""
from __future__ import annotations

import os
import sys
import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))

from core import load_all, make_pseudo, fit_group_model, infer_blocks  # noqa: E402
from pseudotest import folds, thin_to_pairs  # noqa: E402
import pipeline as P  # noqa: E402
import pool as PL  # noqa: E402


def report(out: pd.DataFrame, path: str) -> None:
    out.to_csv(path, index=False)
    print("\nTOTAL")
    print(out[["correct_base", "correct_corrected"]].sum().to_string())
    changed = out[out.changed == 1]
    print(f"\nCHANGED {len(changed)}/{len(out)}")
    print(changed.transition.value_counts().to_string())
    if len(changed):
        print("\nBY CATEGORY")
        print(pd.crosstab(changed.category, changed.transition).to_string())
        print("\nBY BLOCK SIZE")
        print(pd.crosstab(changed.bsz, changed.transition).to_string())
        print("\nDECISION LEDGER")
        print(changed[["qa_id", "fold", "category", "bsz", "answer", "pred_base",
                       "pred_corrected", "transition"]].to_string(index=False))
    print(f"\nwrote {path}")


def fit_pool(tr: pd.DataFrame, meta: pd.DataFrame, hold: list[str], full_target: bool):
    tpool = P.true_pool(tr)
    trn_users = [u for u in sorted(tr.user.dropna().unique()) if u not in hold]
    trn = tr[tr.user.isin(trn_users)]
    PL.fit_cooc(trn)
    os.environ["CHAMP_POOL_FULL_SUBBLOCKS"] = "1" if full_target else "0"
    clf, cols = PL.fit_pool_model(
        trn,
        meta,
        P.training_blocks(tr, trn_users, tpool),
        P.statcache_view("oof"),
    )
    return PL.make_scorer(clf, cols)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "combine":
        pattern = os.path.join(
            ROOT, "research", "pool_full_subblocks_paired_oof_20260905_fold*.csv"
        )
        paths = sorted(glob.glob(pattern))
        if len(paths) != 5:
            raise RuntimeError(f"expected five fold ledgers, found {len(paths)}")
        out = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        if sorted(out.fold.unique().tolist()) != list(range(5)):
            raise RuntimeError("fold ledger set is incomplete")
        path = os.path.join(
            ROOT, "research", "pool_full_subblocks_paired_oof_20260905.csv"
        )
        report(out, path)
        return

    tr, _, meta = load_all()
    P.caches(meta)
    users = sorted(tr.user.dropna().unique())
    rows: list[dict] = []

    split_folds = list(enumerate(folds(users, 5)))
    if len(sys.argv) > 1:
        requested = int(sys.argv[1])
        split_folds = [x for x in split_folds if x[0] == requested]
        if not split_folds:
            raise ValueError(f"fold must be 0..4, got {requested}")

    for fi, hold in split_folds:
        trn_users = [u for u in users if u not in hold]
        trn = tr[tr.user.isin(trn_users)]
        gm = fit_group_model(trn)
        scorer_base = fit_pool(tr, meta, hold, full_target=False)
        scorer_corrected = fit_pool(tr, meta, hold, full_target=True)

        tr_eval = thin_to_pairs(tr, hold, 0.38, seed=fi, policy="ends")
        vis, key, _ = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, gm)
        if P.W_REPAIR and blocks:
            import repair as RP
            gmod = RP.fit_gap_model(trn, meta)
            mrow = P._C["meta_idx"]
            mi = P._C["mi"]
            t0_of = {}
            hau = vis[vis.source == "HAU"]
            for r in hau.drop_duplicates("idx").itertuples():
                if r.true_path in mi:
                    t0_of[r.idx] = mrow.loc[r.true_path, "t0"]
            blocks, _ = RP.repair(blocks, vis, gm, gmod, t0_of)

        sc = P.statcache_view("oof")
        pred_base, pred_corrected = {}, {}
        for block in blocks:
            p0, _ = PL.solve_block(vis, block, sc, scorer_base, "oof")
            p1, _ = PL.solve_block(vis, block, sc, scorer_corrected, "oof")
            pred_base.update(p0)
            pred_corrected.update(p1)

        block_size = {i: len(block) for block in blocks for i in block}
        answer = dict(zip(key.qa_id, key.answer))
        action_vis = vis[(vis.source == "HAU") & vis.category.isin(PL.ACTCATS)]
        for r in action_vis.itertuples():
            b = pred_base.get(r.qa_id)
            c = pred_corrected.get(r.qa_id)
            y = answer[r.qa_id]
            rows.append(
                dict(
                    qa_id=r.qa_id,
                    fold=fi,
                    source=r.source,
                    category=r.category,
                    bsz=block_size.get(r.idx, np.nan),
                    answer=y,
                    pred_base=b,
                    pred_corrected=c,
                    correct_base=int(b == y),
                    correct_corrected=int(c == y),
                    changed=int(b != c),
                    transition=("W->R" if b != y and c == y else
                                "R->W" if b == y and c != y else
                                "W->W" if b != y and c != y else "R->R"),
                )
            )
        print(f"fold {fi} done: {len(hold)} held-out users, {len(blocks)} blocks", flush=True)

    out = pd.DataFrame(rows)
    suffix = f"_fold{split_folds[0][0]}" if len(sys.argv) > 1 else ""
    path = os.path.join(
        ROOT, "research", f"pool_full_subblocks_paired_oof_20260905{suffix}.csv"
    )
    report(out, path)


if __name__ == "__main__":
    main()
