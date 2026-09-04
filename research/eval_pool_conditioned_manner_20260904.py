"""Test action-pool identity as an additional manner-group calibration signal.

The production manner model uses motion descriptors plus only two coarse pool-context
scalars.  This probe adds a one-hot representation of the recovered action pool, while
keeping the same subject-disjoint pair-thinned protocol and the same constrained block
assignment.  OOF pool signatures come from the retained production atlas, not from answers.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "champ"))

import pipeline as P  # noqa: E402
import pool as PL  # noqa: E402
from core import (  # noqa: E402
    PHYS,
    block_features,
    fit_group_model,
    fit_manner,
    infer_blocks,
    load_all,
    make_pseudo,
    mgroup,
    opts,
)
from pseudotest import folds, thin_to_pairs  # noqa: E402


ACT = [
    "Brushing teeth", "Calling", "Checking body temperature", "Checking the time",
    "Combing hair", "Drinking", "Eating", "Folding clothes", "Getting dressed",
    "Grabbing utensils", "Jumping jacks", "Listening to the music with headphones",
    "Lunges", "Lying down", "Massaging oneself", "Mopping", "Peeling fruit",
    "Playing games", "Pouring", "Reading", "Running", "Sitting down", "Squats",
    "Standing up", "Stretching", "Taking a selfie", "Taking medicine", "Turning a page",
    "Typing on a keyboard", "Undressing", "Using a phone", "Walking", "Washing dishes",
    "Washing face", "Watching TV", "Wiping hands", "Wiping surface", "Writing", "Sweeping",
]
POOL_COLS = [f"pool_{a}" for a in ACT] + ["pool_n", "pool_loco"]
LOCO = {"Walking", "Running", "Squats", "Lunges", "Jumping jacks", "Stretching",
        "Sitting down", "Standing up", "Lying down", "Mopping", "Sweeping"}


def pool_features(pool):
    pool = set(pool or [])
    out = {f"pool_{a}": float(a in pool) for a in ACT}
    out["pool_n"] = float(len(pool))
    out["pool_loco"] = float(sum(a in LOCO for a in pool) / max(1, len(pool)))
    return out


def rows_for_block(paths, pool, mfeat, k):
    bf = block_features(paths, mfeat, k)
    pf = pool_features(pool)
    return [{**bf[i], **pf} for i in range(k)]


def fit_pool_head(train, mm, tpool):
    e = train[train.category == "emotion"]
    X, y = [], []
    for (u, a, b), g in e.groupby(["user", "aa", "bb"]):
        rs = list(g.sort_values("cc").itertuples(index=False))
        if len(rs) < 2:
            continue
        paths = [r.path for r in rs]
        pool = tpool[(u, a, b)]
        variants = [list(range(len(rs)))]
        if len(rs) >= 3:
            variants.extend(list(x) for x in itertools.combinations(range(len(rs)), 2))
        for vi in variants:
            vv = [paths[i] for i in vi]
            for i, r in zip(range(len(vi)), [rs[x] for x in vi]):
                # Recover the option text by its answer letter from the named tuple.
                lab = str(getattr(r, str(r.answer)[0])).strip()
                X.append(rows_for_block(vv, pool, mm["mfeat"], len(vv))[i])
                y.append(mgroup(lab))
    xd = pd.DataFrame(X)
    cols = list(xd.columns)
    clf = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.05, max_depth=4,
        l2_regularization=3.0, random_state=20260904)
    clf.fit(xd[cols].to_numpy(float), y)
    return clf, cols


def atlas_pools():
    p = os.path.join(ROOT, "research", "multi_pair_atlas_20260904", "block_atlas_oof_ends.csv")
    d = pd.read_csv(p)
    out = {}
    for r in d.itertuples():
        paths = tuple(json.loads(r.block_paths))
        pool = set(str(r.production_selected_pool).split("||")) if str(r.production_selected_pool) != "nan" else set()
        out[(int(r.fold), frozenset(paths))] = pool
    return out


def solve_group(vis, blocks, mm, clf, cols, path_pools):
    eq = vis[vis.category == "emotion"].set_index("idx")
    pathof = dict(zip(vis.idx, vis.true_path))
    out = {}
    for blk in blocks:
        blk = [b for b in blk if b in eq.index]
        if not blk:
            continue
        k = len(blk)
        rs = [eq.loc[b] for b in blk]
        own = [set(opts(r)) for r in rs]
        I = set.intersection(*own) if len(own) > 1 else set(own[0])
        cand = sorted(I) if len(I) >= k else sorted(set().union(*own))
        slots = len(cand) if k < len(cand) <= 4 else k
        paths = [pathof[b] for b in blk]
        pool = path_pools.get(frozenset(paths), set())
        X = pd.DataFrame(rows_for_block(paths, pool, mm["mfeat"], k)).reindex(columns=cols)
        Pp = clf.predict_proba(X.to_numpy(float))
        gcls = list(clf.classes_)

        def score(lab, positions):
            z = 0.0
            for i, m in enumerate(lab):
                if m not in own[i]:
                    return -1e9
                g = mgroup(m)
                pg = Pp[i, gcls.index(g)] if g in gcls else 1e-6
                z += np.log(max(pg, 1e-9))
                z += np.log(max(mm["pmg"].get(m, 1e-4), 1e-6))
            for s, m in enumerate(positions):
                z += np.log(max(mm["ppm"](m, s, slots), 1e-9))
            return z

        best = None
        for perm in itertools.permutations(cand, slots):
            if slots > k:
                for present in itertools.combinations(range(slots), k):
                    lab = [perm[s] for s in present]
                    z = score(lab, perm)
                    if best is None or z > best[0]:
                        best = (z, lab)
            else:
                z = score(list(perm), perm)
                if best is None or z > best[0]:
                    best = (z, list(perm))
        if best is None:
            continue
        for r, m in zip(rs, best[1]):
            oo = opts(r)
            if m in oo:
                out[r.qa_id] = "ABCD"[oo.index(m)]
    return out


def main():
    tr, te, meta = load_all()
    P.caches(meta)
    atlas = atlas_pools()
    tpool = P.true_pool(tr)
    users = sorted(tr.user.dropna().unique())
    result = []
    for fi, hold in enumerate(folds(users, 5)):
        train = tr[~tr.user.isin(hold)]
        mm = fit_manner(train, meta)
        clf, cols = fit_pool_head(train, mm, tpool)
        evaluation = thin_to_pairs(tr, hold, 0.38, seed=fi, policy="ends")
        vis, key, _ = make_pseudo(evaluation, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, fit_group_model(train))
        pp = {}
        for blk in blocks:
            paths = frozenset(vis[vis.idx.isin(blk)].true_path)
            pp[paths] = atlas.get((fi, paths), set())
        pred = solve_group(vis, blocks, mm, clf, cols, pp)
        ans = dict(zip(key.qa_id, key.answer))
        for r in vis[vis.category == "emotion"].itertuples():
            result.append(dict(fold=fi, bsz=len(next((b for b in blocks if r.idx in b), [])),
                               qa_id=r.qa_id, pred=pred.get(r.qa_id), answer=ans[r.qa_id],
                               correct=int(pred.get(r.qa_id) == ans[r.qa_id])))
        print("fold", fi, "done", flush=True)
    d = pd.DataFrame(result)
    out = os.path.join(ROOT, "research", "pool_conditioned_manner_oof_20260904.csv")
    d.to_csv(out, index=False)
    print(d.groupby(["bsz"]).correct.agg(n="size", correct="sum", acc="mean").to_string())
    print(d.correct.agg(n="size", correct="sum", acc="mean").to_string())


if __name__ == "__main__":
    main()
