"""Subject-disjoint accuracy of the slot-set model, in isolation.

Before wiring anything into solve_emotion: given the two released clips of a session, how well
can `present in {(0,1),(0,2),(1,2)}` be recovered from test-visible evidence alone?

Held-out sessions are cut to every ordered pair of their three trials, which is exactly the
inference-time situation (`pseudotest.thin_to_pairs` drops one trial; the survivors keep their
relative order, and block order is chronological inside a block -- verified 88/89 on the real
test).  Chance is 1/3.

Reported separately, because they feed different parts of solve_emotion's objective:
  * 3-way accuracy
  * adjacent vs middle-withheld (the binary that drives the SLOW over-prediction)
  * (0,1) vs (1,2) among truly-adjacent pairs (which end was withheld)
"""
import os, sys, itertools, json
from collections import Counter
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all, PHYS, block_features
from pseudotest import folds
import clockslot as CS
import slotprior as SPR

NCLS = [(0, 1), (0, 2), (1, 2)]


def sessions(meta):
    """Chronologically ordered 3-trial training sessions -> (user, [paths], [t0], [nf])."""
    h = meta[meta.kind == 'train_hau'].dropna(subset=['t0']).copy()
    t3 = h.trial.str.split('-', expand=True)
    h['a'], h['b'], h['c'] = t3[0], t3[1], t3[2].astype(int)
    out = []
    for (u, a, b), g in h.groupby(['user', 'a', 'b']):
        g = g.sort_values('t0')
        if len(g) != 3:
            continue
        if not (np.isfinite(g.t0).all() and np.isfinite(g.nf).all() and g.nf.min() > 0):
            continue
        out.append((u, list(g.qa_path), list(g.t0), list(g.nf)))
    return out


def main(nfold=5):
    tr, te, meta = load_all()
    mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)}
             for r in meta.itertuples()}
    sess = sessions(meta)
    users = sorted(tr.user.dropna().unique())
    print(f'3-trial training sessions with a clean clock: {len(sess)}')

    rows = []
    for fi, hold in enumerate(folds(users, nfold)):
        hold = set(hold)
        spc = CS.fit(tr, meta, hold)
        spg = SPR.fit_gbm(tr, meta, hold)
        for u, paths, t0, nf in sess:
            if u not in hold:
                continue
            for (i, j) in NCLS:
                pp = [paths[i], paths[j]]
                bf = block_features(pp, mfeat, 3)
                lc = CS.predict(spc, bf, 2, 3, pp)
                lg = SPR.predict_gbm(spg, bf, 2, 3, pp)
                rows.append(dict(
                    fold=fi, user=u, truth=(i, j),
                    clock=max(lc, key=lc.get) if lc else None,
                    gbm=max(lg, key=lg.get) if lg else None,
                    # margin of the clock call, for confidence strata
                    clock_margin=(sorted(lc.values())[-1] - sorted(lc.values())[-2]) if lc else np.nan,
                    gap=abs(t0[j] - t0[i]), lrat=np.log(nf[i] / nf[j])))
    d = pd.DataFrame(rows)
    d['adj_truth'] = d.truth != (0, 2)
    d['adj_clock'] = d.clock != (0, 2)
    d['adj_gbm'] = d.gbm != (0, 2)
    d.to_csv(os.path.join(ROOT, 'slotlab', 'slotmodel_oof.csv'), index=False)

    def line(name, ok, n):
        print(f'  {name:34s} {ok:5d}/{n:5d} = {ok/max(n,1):.4f}')

    print('\n===== 3-way slot-set recovery (subject-disjoint, chance 0.3333) =====')
    line('flat prior (chance)', len(d) // 3, len(d))
    line('slotprior GBM', int((d.gbm == d.truth).sum()), len(d))
    line('clockslot generative', int((d.clock == d.truth).sum()), len(d))

    print('\n===== binary: adjacent vs middle-withheld =====')
    line('GBM', int((d.adj_gbm == d.adj_truth).sum()), len(d))
    line('clockslot', int((d.adj_clock == d.adj_truth).sum()), len(d))
    for nm, col in (('GBM', 'adj_gbm'), ('clockslot', 'adj_clock')):
        tp = int(((d[col]) & (d.adj_truth)).sum()); fp = int(((d[col]) & (~d.adj_truth)).sum())
        fn = int(((~d[col]) & (d.adj_truth)).sum()); tn = int(((~d[col]) & (~d.adj_truth)).sum())
        bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        print(f'    {nm:10s} recall(adj)={tp/max(tp+fn,1):.3f} '
              f'recall(skip)={tn/max(tn+fp,1):.3f}  balanced={bal:.3f}')

    print('\n===== which end was withheld, among TRULY adjacent pairs =====')
    a = d[d.adj_truth]
    line('clockslot', int((a.clock == a.truth).sum()), len(a))
    line('GBM', int((a.gbm == a.truth).sum()), len(a))
    print('    (chance within {(0,1),(1,2)} is 0.5; clockslot has no signal here by design)')

    print('\n===== per fold, clockslot 3-way =====')
    for f, g in d.groupby('fold'):
        line(f'fold {f}', int((g.clock == g.truth).sum()), len(g))

    print('\n===== clockslot confusion (rows truth, cols predicted) =====')
    print(pd.crosstab(d.truth.astype(str), d.clock.astype(str)).to_string())

    print('\n===== clockslot 3-way accuracy by margin decile =====')
    d['dec'] = pd.qcut(d.clock_margin, 5, labels=False, duplicates='drop')
    for q, g in d.groupby('dec'):
        line(f'margin q{int(q)} (>= {g.clock_margin.min():.2f})',
             int((g.clock == g.truth).sum()), len(g))

    json.dump(dict(n=len(d),
                   clock_3way=float((d.clock == d.truth).mean()),
                   gbm_3way=float((d.gbm == d.truth).mean()),
                   clock_adj_bal=float(0.5 * (
                       ((d.adj_clock) & (d.adj_truth)).sum() / max(d.adj_truth.sum(), 1)
                       + ((~d.adj_clock) & (~d.adj_truth)).sum() / max((~d.adj_truth).sum(), 1))),
                   clock_end_acc=float((a.clock == a.truth).mean())),
              open(os.path.join(ROOT, 'slotlab', 'slotmodel_summary.json'), 'w'), indent=2)
    print('\nwrote slotlab/slotmodel_oof.csv and slotlab/slotmodel_summary.json')


if __name__ == '__main__':
    main()
