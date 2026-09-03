"""5-fold subject-disjoint pseudo-test evaluation of the emotion solver.

For each fold the held-out users are re-shaped into a test-like problem: answers hidden,
user/trial ids stripped, clips renumbered.  The solver sees only the test-visible view.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_all, make_pseudo, fit_group_model, infer_blocks, fit_manner,
                  solve_emotion, opts, gt_letters, mgroup)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def folds(users, n=5, seed=7):
    u = sorted(users, key=lambda s: int(s[4:]))
    rng = np.random.default_rng(seed)
    u = list(rng.permutation(u))
    return [u[i::n] for i in range(n)]


def block_quality(vis, blocks, aux):
    """How well the inferred blocks match the true sessions (diagnostic only)."""
    pathof = dict(zip(vis.idx, vis.true_path))
    tb = aux['true_block']
    exact = 0
    tot = 0
    for blk in blocks:
        tot += 1
        ids = {tb.get(pathof[b]) for b in blk}
        true_sizes = [len(x) for x in aux['blocks']]
        if len(ids) == 1:
            bi = ids.pop()
            exact += (len(blk) == len(aux['blocks'][bi]))
    return exact, tot


def thin_to_pairs(tr, users, frac, seed=0):
    """Drop one trial from `frac` of the held-out users' sessions, so the pseudo-test block
    sizes match the real test set (which contains 21 two-trial sessions of 55)."""
    rng = np.random.default_rng(seed)
    sub = tr[tr.user.isin(users)]
    drop = []
    for (u, a, b), g in sub[sub.source == 'HAU'].groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique())
        if len(paths) >= 3 and rng.random() < frac:
            drop.append(paths[rng.integers(len(paths))])
    return tr[~tr.path.isin(drop)]


def run(w_phys=1.0, w_pos=1.0, verbose=True, nfold=5, pair_frac=0.0):
    tr, te, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    res = []
    bq = [0, 0]
    for fi, hold in enumerate(folds(users, nfold)):
        trn = tr[~tr.user.isin(hold)]
        tr_eval = thin_to_pairs(tr, hold, pair_frac, seed=fi) if pair_frac else tr
        vis, key, aux = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
        gm = fit_group_model(trn)
        mm = fit_manner(trn, meta)
        blocks = infer_blocks(vis, gm)
        e, t = block_quality(vis, blocks, aux)
        bq[0] += e; bq[1] += t
        pred, diag = solve_emotion(vis, blocks, mm, w_phys, w_pos)
        ans = dict(zip(key.qa_id, key.answer))
        eq = vis[vis.category == 'emotion']
        ok = sum(1 for q in eq.qa_id if pred.get(q) == ans[q])
        res.append(dict(fold=fi, users=len(hold), n=len(eq), correct=ok,
                        acc=ok / len(eq), blocks=len(blocks),
                        inter_used=diag.used_inter.mean() if len(diag) else np.nan))
        if verbose:
            print(f'  fold {fi} users={len(hold)} n={len(eq):4d} correct={ok:4d} '
                  f'acc={ok/len(eq):.4f} blocks={len(blocks)} inter_used={res[-1]["inter_used"]:.3f}')
    d = pd.DataFrame(res)
    tot_c, tot_n = d.correct.sum(), d.n.sum()
    print(f'  EMOTION pseudo-test: {tot_c}/{tot_n} = {tot_c/tot_n:.4f}   '
          f'block-recovery exact {bq[0]}/{bq[1]} = {bq[0]/max(1,bq[1]):.3f}')
    return d, tot_c, tot_n


if __name__ == '__main__':
    print('=== emotion: structure + physical evidence (all triples) ===')
    run(1.0, 1.0)
    print('=== ablation: position prior only (w_phys=0) ===')
    run(0.0, 1.0)
    print('=== ablation: physical evidence only (w_pos=0) ===')
    run(1.0, 0.0)
    print('=== stress test: 38%% of sessions thinned to pairs, matching the test mix ===')
    run(1.0, 1.0, pair_frac=0.38)
