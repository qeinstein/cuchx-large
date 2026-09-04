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


def thin_to_pairs(tr, users, frac, seed=0, policy=None):
    """Drop one trial from `frac` of the held-out users' sessions, so the pseudo-test block
    sizes match the real test set (which contains 21 two-trial sessions of 55).

    `policy` selects WHICH trial is withheld, which is a second protocol parameter that
    matters as much as `frac` and was previously left at 'uniform' by default:

      'uniform'  any of the three trials, equiprobably -> the surviving pair is
                 (0,1)/(0,2)/(1,2) each a third of the time.
      'ends'     only the first or the last trial, so the surviving pair is always ADJACENT.

    'ends' is the regime the real test is in.  Estimated by maximum likelihood from the 21
    real-test pair-block recording gaps against the training-fitted adjacent/skipped
    log-normal densities (slotlab/estimate_policy.py): pi = P(adjacent) = 1.000, parametric
    bootstrap 95% CI [0.601, 1.000], and a likelihood-ratio test rejects the pi = 2/3 implied
    by a flat prior at p = 0.0039.  The same estimator returns exactly 1.000 on the 34
    complete test triples, where pi = 1 is known, so it is not biased high by construction.
    """
    if policy is None:
        policy = os.environ.get('CHAMP_THIN_POLICY', 'uniform')
    rng = np.random.default_rng(seed)
    sub = tr[tr.user.isin(users)]
    drop = []
    for (u, a, b), g in sub[sub.source == 'HAU'].groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique())
        if len(paths) >= 3 and rng.random() < frac:
            if policy == 'ends':
                drop.append(paths[0] if rng.random() < 0.5 else paths[-1])
            elif policy == 'uniform':
                drop.append(paths[rng.integers(len(paths))])
            else:
                raise ValueError(f'unknown thinning policy {policy!r}')
    return tr[~tr.path.isin(drop)]


def thin_to_orphans(tr, users, frac, seed=0):
    """Drop two trials from `frac` of the held-out users' sessions, leaving a one-clip block.

    The real test set demonstrably contains such orphans: after repair it is 31 three-clip
    blocks, 23 two-clip blocks and 5 one-clip blocks.  Since infer_blocks has kmin=2 it can
    never emit a one-clip block, so an orphan is always glued onto a real session -- which is
    exactly the failure this injection reproduces so that the repair can be validated.
    """
    rng = np.random.default_rng(seed)
    sub = tr[tr.user.isin(users)]
    drop = []
    for (u, a, b), g in sub[sub.source == 'HAU'].groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique())
        if len(paths) >= 3 and rng.random() < frac:
            keep = paths[rng.integers(len(paths))]
            drop += [p for p in paths if p != keep]
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
