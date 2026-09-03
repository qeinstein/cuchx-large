"""Fast emotion-only sweep over the fusion weights, on the shared inference path
(make_pseudo -> infer_blocks -> solve_emotion).  No answers are used at inference."""
import os, sys, itertools
import numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_all, make_pseudo, fit_group_model, infer_blocks, fit_manner,
                  solve_emotion)
import emopair as EP
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    tr, te, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    # pre-fit everything once per fold, then sweep weights over the cached models
    folds_data = []
    for fi, hold in enumerate(folds(users, 5)):
        trn = tr[~tr.user.isin(hold)]
        mm = fit_manner(trn, meta)
        mm['pm'] = EP.fit(trn, meta, mm, pool_of=None)
        gm = fit_group_model(trn)
        vis, key, aux = make_pseudo(tr, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, gm)
        folds_data.append((vis, dict(zip(key.qa_id, key.answer)), blocks, mm))
        print(f'  fold {fi} models fitted', flush=True)

    grid = [(wp, wo, wr) for wp in (1.0, 0.5, 0.0) for wo in (1.0,) for wr in
            (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)]
    best = None
    print(f'\n  {"w_phys":>7s} {"w_pos":>6s} {"w_pair":>7s}   correct/809      acc')
    for wp, wo, wr in grid:
        tot = c = 0
        per = []
        for vis, ans, blocks, mm in folds_data:
            pred, _ = solve_emotion(vis, blocks, mm, w_phys=wp, w_pos=wo, w_pair=wr)
            eq = vis[vis.category == 'emotion']
            k = sum(1 for q in eq.qa_id if pred.get(q) == ans[q])
            c += k; tot += len(eq); per.append(k)
        print(f'  {wp:7.1f} {wo:6.1f} {wr:7.1f}   {c:5d}/{tot:<5d}   {c/tot:.4f}   '
              f'folds {per}')
        if best is None or c > best[0]:
            best = (c, wp, wo, wr, per)
    print(f'\n  BEST: {best[0]}/809 = {best[0]/809:.4f} at w_phys={best[1]} '
          f'w_pos={best[2]} w_pair={best[3]}')
    print('  checkpoint 731/809 = 0.9036 | gate 755/809 = 0.9333 | target 770/809 = 0.9518')


if __name__ == '__main__':
    main()
