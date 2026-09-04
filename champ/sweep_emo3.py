"""Emotion fusion weights and the slot prior, swept SEPARATELY for two-clip and three-clip
blocks, on a pseudo-test whose block-size mix matches the real test (23 pairs + 5 orphans
of 59 sessions).

Rationale: the champion uses one global (w_phys, w_pos, w_pair).  The pair-stress run shows
two-clip blocks are a different regime -- 0.7296 vs 0.8966 -- because they carry an extra
latent (which two of the three protocol slots are present) that the three-clip case does not.
There is no reason the same fusion weights should be optimal in both.
"""
import os, sys, itertools, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_all, make_pseudo, fit_group_model, infer_blocks, fit_manner,
                  solve_emotion, opts)
import emopair as EP, slotprior as SPR
from pseudotest import folds, thin_to_pairs, thin_to_orphans

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(pair_frac=0.39, orph_frac=0.085):
    tr, te, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    FD = []
    for fi, hold in enumerate(folds(users, 5)):
        trn = tr[~tr.user.isin(hold)]
        mm = fit_manner(trn, meta)
        mm['pm'] = EP.fit(trn, meta, mm, pool_of=None)
        gm = fit_group_model(trn)
        sp = SPR.fit(tr, meta, hold)
        tr_eval = thin_to_orphans(tr, hold, orph_frac, seed=500 + fi)
        tr_eval = thin_to_pairs(tr_eval, hold, pair_frac, seed=fi)
        vis, key, aux = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, gm)
        FD.append((vis, dict(zip(key.qa_id, key.answer)), blocks, mm, sp))
        print(f'  fold {fi} fitted; blocks {dict(pd.Series([len(b) for b in blocks]).value_counts())}',
              flush=True)

    def score(w3, w2, wslot=0.0):
        res = {2: [0, 0], 3: [0, 0]}
        for vis, ans, blocks, mm, sp in FD:
            b2 = [b for b in blocks if len(b) == 2]
            b3 = [b for b in blocks if len(b) != 2]
            pred = {}
            if b3:
                p, _ = solve_emotion(vis, b3, mm, *w3); pred.update(p)
            if b2:
                p, _ = solve_emotion(vis, b2, mm, *w2, slotp=sp, w_slot=wslot); pred.update(p)
            eq = vis[vis.category == 'emotion']
            bs = {i: len(b) for b in blocks for i in b}
            for r in eq.itertuples():
                k = 2 if bs.get(r.idx) == 2 else 3
                res[k][1] += 1
                res[k][0] += int(pred.get(r.qa_id) == ans[r.qa_id])
        tot = sum(v[0] for v in res.values()); n = sum(v[1] for v in res.values())
        return tot, n, res

    base = (1.0, 1.0, 1.0)
    t, n, r = score(base, base, 0.0)
    print(f'\n  CHAMPION weights everywhere: {t}/{n} = {t/n:.4f}   '
          f'k=2 {r[2][0]}/{r[2][1]}={r[2][0]/max(r[2][1],1):.4f}  '
          f'k=3 {r[3][0]}/{r[3][1]}={r[3][0]/max(r[3][1],1):.4f}')

    print('\n  --- sweep k=2 weights (k>=3 held at the champion values) ---')
    print(f'  {"w_phys":>7s}{"w_pos":>7s}{"w_pair":>7s}{"w_slot":>7s}  k=2 acc      total')
    rows = []
    for wp in (0.25, 0.5, 1.0, 2.0):
        for wo in (0.25, 0.5, 1.0, 2.0):
            for wr in (0.5, 1.0, 2.0, 4.0):
                for ws in (0.0, 0.5, 1.0, 2.0):
                    t2, n2, r2 = score(base, (wp, wo, wr), ws)
                    rows.append(dict(wp=wp, wo=wo, wr=wr, ws=ws,
                                     k2=r2[2][0], k2n=r2[2][1], tot=t2, n=n2))
    R = pd.DataFrame(rows).sort_values('k2', ascending=False)
    for r_ in R.head(20).itertuples():
        print(f'  {r_.wp:7.2f}{r_.wo:7.2f}{r_.wr:7.2f}{r_.ws:7.2f}  '
              f'{r_.k2}/{r_.k2n}={r_.k2/r_.k2n:.4f}   {r_.tot}/{r_.n}={r_.tot/r_.n:.4f}')
    R.to_csv(os.path.join(ROOT, 'champ', 'sweep_emo3.csv'), index=False)
    print('\n  slot-prior marginal at champion k=2 weights:')
    for ws in (0.0, 0.5, 1.0, 2.0, 4.0):
        t2, n2, r2 = score(base, base, ws)
        print(f'    w_slot={ws:4.1f}  k=2 {r2[2][0]}/{r2[2][1]}={r2[2][0]/r2[2][1]:.4f}  total {t2}/{n2}={t2/n2:.4f}')


if __name__ == '__main__':
    main()
