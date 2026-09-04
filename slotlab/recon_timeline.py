"""Is the real test set's clip index globally ordered by the recording clock?

If it is, then for a two-clip block the gaps to the NEIGHBOURING blocks identify which trial
was withheld, because a missing trial leaves a hole in an otherwise dense timeline:

    withheld trial 1 (present slots 1,2) -> abnormally large gap BEFORE the block
    withheld trial 3 (present slots 0,1) -> abnormally large gap AFTER  the block
    withheld trial 2 (present slots 0,2) -> abnormally large gap INSIDE the block

The three hypotheses are therefore separated by three purely test-visible clock statistics,
calibrated on the 34 test triples where nothing is missing.
"""
import os, sys, itertools
from collections import Counter
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all, real_test_view, fit_group_model, infer_blocks, opts


def sec(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78, flush=True)


def main():
    tr, te, meta = load_all()
    mrow = meta.set_index('qa_path')
    vis = real_test_view(te)
    gm = fit_group_model(tr)
    blocks = infer_blocks(vis, gm)
    pathof = dict(zip(vis.idx, vis.true_path))

    hau_idx = sorted(vis[vis.source == 'HAU'].idx.unique())
    t0 = {}
    nf = {}
    for i in hau_idx:
        p = pathof[i]
        if p in mrow.index:
            t0[i] = float(mrow.loc[p, 't0']); nf[i] = float(mrow.loc[p, 'nf'])

    sec('1. is HAU clip index monotone in the recording clock?')
    # NB keep index alignment: build (idx_a, idx_b, delta) over consecutive INDICES that
    # both have a clock, rather than diffing a NaN-filtered array (which desynchronises
    # the labels from the values).
    dl = [(hau_idx[i], hau_idx[i + 1], t0[hau_idx[i + 1]] - t0[hau_idx[i]])
          for i in range(len(hau_idx) - 1)
          if hau_idx[i] in t0 and hau_idx[i + 1] in t0]
    d = np.array([v for _, _, v in dl])
    print(f'HAU clips {hau_idx[0]}..{hau_idx[-1]}  n={len(hau_idx)}  '
          f'with clock={sum(1 for i in hau_idx if i in t0)}')
    print(f'consecutive-index clock deltas: n={len(d)}  '
          f'negative={int((d < 0).sum())}  zero={int((d == 0).sum())}')
    print(f'  deltas percentiles: {np.round(np.percentile(d, [0,5,25,50,75,95,100]),1)}')

    # which of those consecutive index pairs lie INSIDE an inferred block, and which
    # straddle a block boundary?  Only the inside ones bear on the position prior.
    blk_of = {int(i): bi for bi, b in enumerate(blocks) for i in b}
    inside = [(a, b, v) for a, b, v in dl if blk_of.get(int(a)) == blk_of.get(int(b))]
    across = [(a, b, v) for a, b, v in dl if blk_of.get(int(a)) != blk_of.get(int(b))]
    vi = np.array([v for _, _, v in inside]); va = np.array([v for _, _, v in across])
    print(f'\n  INSIDE  an inferred block: n={len(vi)}  negative={int((vi<0).sum())}  '
          f'median={np.median(vi):.1f}  p5={np.percentile(vi,5):.1f} '
          f'p95={np.percentile(vi,95):.1f}')
    print(f'  ACROSS  a block boundary : n={len(va)}  negative={int((va<0).sum())}  '
          f'median={np.median(va):.1f}')
    print('\n  within-block index pairs that go BACKWARDS in time:')
    for a, b, v in inside:
        if v < 0:
            print(f'     {a} -> {b}   delta = {v:.1f}s   (block {blocks[blk_of[int(a)]]})')
    print('  within-block index pairs with |delta| > 400s:')
    for a, b, v in inside:
        if abs(v) > 400:
            print(f'     {a} -> {b}   delta = {v:.1f}s   (block {blocks[blk_of[int(a)]]})')

    sec('1b. in TRAINING, is trial index (cc) the same order as the recording clock?')
    h = meta[meta.kind == 'train_hau'].dropna(subset=['t0']).copy()
    t3 = h.trial.str.split('-', expand=True)
    h['a'], h['b'], h['c'] = t3[0], t3[1], t3[2].astype(int)
    nmono = ntot = 0
    for (u, a, b), g in h.groupby(['user', 'a', 'b']):
        g = g.sort_values('c')
        if len(g) < 2:
            continue
        ntot += 1
        nmono += int(all(x < y for x, y in zip(g.t0, list(g.t0)[1:])))
    print(f'  sessions whose cc order equals clock order: {nmono}/{ntot} '
          f'= {nmono/max(ntot,1):.4f}')

    sec('2. calibrate on the 34 test triples: within-block and between-block gaps')
    trip = [b for b in blocks if len(b) == 3]
    pair = [b for b in blocks if len(b) == 2]
    within = []
    for b in trip:
        if all(i in t0 for i in b):
            within += [t0[b[1]] - t0[b[0]], t0[b[2]] - t0[b[1]]]
    within = np.array(within)
    print(f'within-triple adjacent gaps: n={len(within)} '
          f'median={np.median(within):.1f} '
          f'p5={np.percentile(within,5):.1f} p95={np.percentile(within,95):.1f}')

    # between-block gaps measured ONLY between two consecutive triples (both complete),
    # so no withheld trial contaminates the estimate
    order = sorted(blocks, key=lambda b: min(b))
    btw_tt = []
    for x, y in zip(order, order[1:]):
        if len(x) == 3 and len(y) == 3 and x[-1] in t0 and y[0] in t0:
            g = t0[y[0]] - t0[x[-1]]
            if 0 < g < 1e4:
                btw_tt.append(g)
    btw_tt = np.array(btw_tt)
    print(f'between consecutive COMPLETE triples: n={len(btw_tt)} '
          f'median={np.median(btw_tt):.1f} '
          f'p5={np.percentile(btw_tt,5):.1f} p95={np.percentile(btw_tt,95):.1f}')
    print(f'\n=> a withheld trial should inflate a between-block gap by roughly one '
          f'trial period (~{np.median(within):.0f}s)')

    sec('3. per-pair-block clock evidence')
    rows = []
    for bi, b in enumerate(order):
        if len(b) != 2:
            continue
        j = order.index(b)
        prev = order[j - 1] if j > 0 else None
        nxt = order[j + 1] if j + 1 < len(order) else None
        gb = (t0[b[0]] - t0[prev[-1]]) if (prev and prev[-1] in t0 and b[0] in t0) else np.nan
        ga = (t0[nxt[0]] - t0[b[1]]) if (nxt and nxt[0] in t0 and b[1] in t0) else np.nan
        gi = t0[b[1]] - t0[b[0]] if all(i in t0 for i in b) else np.nan
        rows.append(dict(blk=str([int(x) for x in b]), i0=int(b[0]), i1=int(b[1]),
                         prev=str([int(x) for x in prev]) if prev else '',
                         nxt=str([int(x) for x in nxt]) if nxt else '',
                         gap_before=gb, gap_inside=gi, gap_after=ga,
                         nf0=nf.get(b[0], np.nan), nf1=nf.get(b[1], np.nan)))
    P = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(P.round(1).to_string(index=False))
    P.to_csv(os.path.join(ROOT, 'slotlab', 'test_pair_clock.csv'), index=False)

    sec('4. naive hypothesis call from the clock alone')
    mw = np.median(within)
    hi = np.percentile(btw_tt, 95) if len(btw_tt) else np.inf
    print(f'thresholds: within-block hole if gap_inside > {np.percentile(within,95):.1f}s ; '
          f'neighbour hole if gap > {hi:.1f}s')
    call = []
    for r in P.itertuples():
        if np.isfinite(r.gap_inside) and r.gap_inside > np.percentile(within, 95):
            c = '(0,2) middle withheld'
        elif np.isfinite(r.gap_before) and r.gap_before > hi and not (
                np.isfinite(r.gap_after) and r.gap_after > hi):
            c = '(1,2) first withheld'
        elif np.isfinite(r.gap_after) and r.gap_after > hi and not (
                np.isfinite(r.gap_before) and r.gap_before > hi):
            c = '(0,1) last withheld'
        else:
            c = 'ambiguous'
        call.append(c)
    P['call'] = call
    print(P[['blk', 'gap_before', 'gap_inside', 'gap_after', 'call']].round(1).to_string(index=False))
    print('\ncall distribution:', Counter(call))
    P.to_csv(os.path.join(ROOT, 'slotlab', 'test_pair_clock.csv'), index=False)
    print('\nwrote slotlab/test_pair_clock.csv')


if __name__ == '__main__':
    main()
