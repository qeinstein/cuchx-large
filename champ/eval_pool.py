"""Pseudo-test evaluation of the session action-pool solver (P2)."""
import os, sys, json, itertools
import numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_all, make_pseudo, fit_group_model, infer_blocks, opts)
from pseudotest import folds
import dense as D, decode as DC, pool as PL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGITS = os.environ.get('CHAMP_LOGITS', 'dense_logits.npz')


def build_statcache(paths, split):
    DC._LG = np.load(os.path.join(ROOT, 'champ', LOGITS))
    out = {}
    for p in paths:
        lp = DC.logits(split, p)
        if lp is not None:
            out[p] = DC.presence_stats(lp)
    return out


def true_pool(tr):
    pool = defaultdict(set)
    for _, r in tr[(tr.source == 'HAU') & tr.category.isin(PL.ACTCATS)].iterrows():
        for L in str(r['answer']):
            if L in 'ABCD':
                for p in str(r[L]).split(','):
                    pool[(r.user, r.aa, r.bb)].add(p.strip())
    return pool


def training_blocks(tr, users, tpool):
    """Sessions of the given (training) users, in a vis-like frame, with true pools."""
    out = {}
    h = tr[(tr.source == 'HAU') & tr.user.isin(users)]
    for (u, a, b), g in h.groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique())
        idx = {p: i for i, p in enumerate(paths)}
        v = g.copy()
        v['idx'] = v.path.map(idx)
        v['true_path'] = v.path
        out[(u, a, b)] = (v, list(range(len(paths))), tpool[(u, a, b)])
    return out


def run(nfold=5, verbose=True):
    tr, te, meta = load_all()
    tpool = true_pool(tr)
    users = sorted(tr.user.dropna().unique())
    sc_tr = build_statcache(tr[tr.source == 'HAU'].path.unique(), 'oof')
    tot = defaultdict(lambda: [0, 0])
    pooldiag = []
    for fi, hold in enumerate(folds(users, nfold)):
        trn_users = [u for u in users if u not in hold]
        tb = training_blocks(tr, trn_users, tpool)
        clf, cols = PL.fit_pool_model(tr, meta, tb, sc_tr)
        scorer = PL.make_scorer(clf, cols)
        vis, key, aux = make_pseudo(tr, meta, hold, seed=100 + fi)
        gm = fit_group_model(tr[~tr.user.isin(hold)])
        blocks = infer_blocks(vis, gm)
        ans = dict(zip(key.qa_id, key.answer))
        cat_of = dict(zip(vis.qa_id, vis.category))
        # true pool per inferred block, for diagnostics only
        pathof = dict(zip(vis.idx, vis.true_path))
        p2sess = {r.path: (r.user, r.aa, r.bb) for _, r in tr[tr.source == 'HAU'].iterrows()}
        for blk in blocks:
            pred, diag = PL.solve_block(vis, blk, sc_tr, scorer, 'oof')
            if diag:
                ks = {p2sess[pathof[b]] for b in blk}
                tp = set().union(*[tpool[k] for k in ks]) if ks else set()
                inter = set(diag['pool']) & tp
                pooldiag.append(dict(nsol=diag['nsol'], uni=diag['uni'], free=diag['free'],
                                     P=len(inter) / max(1, len(diag['pool'])),
                                     R=len(inter) / max(1, len(tp))))
            for q, L in pred.items():
                c = cat_of[q]
                tot[c][1] += 1
                tot[c][0] += int(L == ans[q])
        if verbose:
            print(f'  fold {fi} done', flush=True)
    pd_ = pd.DataFrame(pooldiag)
    print('\n  POOL solver (pseudo-test, blocks inferred from test-visible info):')
    for c in ['single', 'multi', 'combination']:
        k, n = tot[c]
        print(f'    {c:12s} {k:5d}/{n:5d} = {k/max(1,n):.4f}')
    print(f'\n  pool set precision {pd_.P.mean():.3f} recall {pd_.R.mean():.3f}; '
          f'mean #SAT pools {pd_.nsol.mean():.1f}; mean |universe| {pd_.uni.mean():.1f}; '
          f'mean #free {pd_.free.mean():.1f}')
    return tot


if __name__ == '__main__':
    run()
