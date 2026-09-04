"""Reconnaissance for the two-clip protocol-slot latent.

The densest remaining error pool against the 0.93859 champion is emotion in two-clip session
blocks: 0.7296 vs 0.8966 in three-clip blocks (champ/log_pairstress.txt), i.e. ~6 of the ~21
remaining public errors sit in ~22 public questions.

The candidate manner set is right there (|I|=3 for 20 of 21 test pairs); what is unknown is
WHICH two of the three protocol slots (trial 1 / 2 / 3) the released clips occupy.
`core.solve_emotion` maximises over that latent with a flat prior.

This script measures, with no labels used at inference-shaped time:
  1. how deterministically the protocol maps trial index -> manner group (train)
  2. how well the clip DURATION orders trials within a session (train, and test triples)
  3. how well the recording-clock GAP separates "adjacent trials" from "one trial skipped"
  4. the same two statistics on the 34 real-test triples, so the pair blocks can be
     calibrated in the test domain rather than the training domain
  5. what the resulting slot-set posterior looks like on the 21 real-test pair blocks
"""
import os, sys, json, itertools
from collections import Counter, defaultdict
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import (load_all, real_test_view, fit_group_model, infer_blocks, mgroup, opts,
                  gt_letters, GROUPS)

OUT = os.path.join(ROOT, 'slotlab')


def sec(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78, flush=True)


def main():
    tr, te, meta = load_all()
    mrow = meta.set_index('qa_path')

    # ------------------------------------------------------------------ 1. protocol map
    sec('1. trial index -> manner group (training HAU emotion)')
    e = tr[tr.category == 'emotion'].copy()
    e['lab'] = [str(r[gt_letters(r)[0]]).strip() for _, r in e.iterrows()]
    e['grp'] = e.lab.map(mgroup)
    e['c'] = e.cc.astype(int)
    ct = pd.crosstab(e.c, e.grp)
    print(ct.to_string())
    print('\nrow-normalised P(group | trial index):')
    print((ct.T / ct.sum(1)).T.round(3).to_string())

    # within a session, is the group->slot map a clean bijection?
    ok = tot = 0
    perm_cnt = Counter()
    for (u, a, b), g in e.groupby(['user', 'aa', 'bb']):
        if len(g) != 3:
            continue
        g = g.sort_values('c')
        gs = tuple(g.grp)
        tot += 1
        perm_cnt[gs] += 1
        ok += len(set(gs)) == 3
    print(f'\nsessions whose 3 manner groups are all distinct: {ok}/{tot} = {ok/max(tot,1):.3f}')
    print('most common (slot1,slot2,slot3) group signatures:')
    for k, v in perm_cnt.most_common(10):
        print(f'   {v:4d}  {k}')

    # ------------------------------------------------------------------ 2. duration ordering
    sec('2. does clip duration order the trials?  (nf = frame count)')
    h = meta[meta.kind == 'train_hau'].dropna(subset=['t0']).copy()
    t3 = h.trial.str.split('-', expand=True)
    h['a'], h['b'], h['c'] = t3[0], t3[1], t3[2].astype(int)
    print('mean/median nf by trial index:')
    print(h.groupby('c').nf.agg(['size', 'mean', 'median', 'std']).round(1).to_string())

    # within-session ranking accuracy of nf
    npair = nok = 0
    trip_perm = Counter()
    for (u, a, b), g in h.groupby(['user', 'a', 'b']):
        g = g.sort_values('c')
        if len(g) != 3:
            continue
        nf = list(g.nf)
        for i, j in itertools.combinations(range(3), 2):
            npair += 1
            nok += int(nf[i] > nf[j])          # expect longer for the earlier trial
        trip_perm[tuple(np.argsort(np.argsort([-x for x in nf])))] += 1
    print(f'\nwithin-session pairs where the EARLIER trial is longer: {nok}/{npair} '
          f'= {nok/max(npair,1):.3f}')
    print('rank of nf (descending) per slot, signature counts (0,1,2)=perfect:')
    for k, v in trip_perm.most_common(6):
        print(f'   {v:4d}  {k}')

    # ------------------------------------------------------------------ 3. gap separation
    sec('3. recording-clock gap: adjacent trials vs one trial skipped')
    adj, skip = [], []
    for (u, a, b), g in h.groupby(['user', 'a', 'b']):
        g = g.sort_values('c')
        if len(g) != 3:
            continue
        t = list(g.t0)
        adj += [t[1] - t[0], t[2] - t[1]]
        skip.append(t[2] - t[0])
    adj = np.array([x for x in adj if 0 < x < 1e5])
    skip = np.array([x for x in skip if 0 < x < 1e5])
    for nm, v in (('adjacent (slots i,i+1)', adj), ('skipped  (slots 0,2)', skip)):
        print(f'  {nm}: n={len(v):4d} median={np.median(v):8.1f} '
              f'p10={np.percentile(v,10):8.1f} p90={np.percentile(v,90):8.1f}')
    # optimal single threshold
    allv = np.concatenate([adj, skip])
    lab = np.concatenate([np.zeros(len(adj)), np.ones(len(skip))])
    ths = np.unique(np.percentile(allv, np.linspace(1, 99, 200)))
    best = max(((np.mean((allv > t) == lab), t) for t in ths))
    print(f'  best single-threshold accuracy separating the two: {best[0]:.3f} at gap={best[1]:.1f}s')
    # and the balanced version (what matters, since priors differ at inference)
    ba = max((((np.mean(allv[lab == 0] <= t) + np.mean(allv[lab == 1] > t)) / 2, t) for t in ths))
    print(f'  best BALANCED accuracy: {ba[0]:.3f} at gap={ba[1]:.1f}s')

    # ------------------------------------------------------------------ 4. test blocks
    sec('4. real-test block structure and its test-domain statistics')
    gm = fit_group_model(tr)
    vis = real_test_view(te)
    blocks = infer_blocks(vis, gm)
    print('inferred sizes:', Counter(len(b) for b in blocks))
    hau = vis[vis.source == 'HAU']
    eq = hau[hau.category == 'emotion'].set_index('idx')
    pathof = dict(zip(vis.idx, vis.true_path))

    def clipstat(i):
        p = pathof[i]
        if p not in mrow.index:
            return dict(nf=np.nan, t0=np.nan)
        r = mrow.loc[p]
        return dict(nf=float(r.nf), t0=float(r.t0))

    # test-domain triples: duration ordering + gap distribution
    tadj, tnok, tnpair = [], 0, 0
    trip_rows = []
    for blk in [b for b in blocks if len(b) == 3]:
        st = [clipstat(i) for i in blk]
        nf = [s['nf'] for s in st]; t0 = [s['t0'] for s in st]
        if not all(np.isfinite(nf)) or not all(np.isfinite(t0)):
            continue
        for i, j in itertools.combinations(range(3), 2):
            tnpair += 1; tnok += int(nf[i] > nf[j])
        tadj += [t0[1] - t0[0], t0[2] - t0[1]]
        trip_rows.append(dict(blk=str(blk), nf=nf, g1=t0[1] - t0[0], g2=t0[2] - t0[1]))
    tadj = np.array([x for x in tadj if 0 < x < 1e5])
    print(f'\ntest triples: earlier-trial-longer on {tnok}/{tnpair} = {tnok/max(tnpair,1):.3f}'
          f'   (train: {nok/max(npair,1):.3f})')
    print(f'test triples adjacent gap: n={len(tadj)} median={np.median(tadj):.1f} '
          f'p10={np.percentile(tadj,10):.1f} p90={np.percentile(tadj,90):.1f}'
          f'   (train median {np.median(adj):.1f})')
    print(f'test triples nf: median={np.median([x for r in trip_rows for x in r["nf"]]):.1f}'
          f'   (train median {h.nf.median():.1f})')

    # ------------------------------------------------------------------ 5. test pair blocks
    sec('5. the 21 real-test two-clip blocks')
    rows = []
    for blk in [b for b in blocks if len(b) == 2]:
        st = [clipstat(i) for i in blk]
        O = [set(opts(eq.loc[i])) for i in blk if i in eq.index]
        I = set.intersection(*O) if len(O) > 1 else (O[0] if O else set())
        cand = sorted(I)
        grp = [mgroup(m) for m in cand]
        gap = st[1]['t0'] - st[0]['t0']
        rows.append(dict(blk=str(blk), i0=blk[0], i1=blk[1],
                         nf0=st[0]['nf'], nf1=st[1]['nf'],
                         gap=gap, nI=len(I),
                         cand='|'.join(cand), groups='|'.join(grp),
                         ngroup=len(set(grp)),
                         nf_desc=int(st[0]['nf'] > st[1]['nf'])))
    P = pd.DataFrame(rows)
    print(P.to_string(index=False))
    P.to_csv(os.path.join(OUT, 'test_pair_blocks.csv'), index=False)
    print(f'\npairs with |I|=3: {(P.nI==3).sum()}/{len(P)}')
    print(f'pairs whose 3 candidates have 3 distinct groups: {(P.ngroup==3).sum()}/{len(P)}')
    print('group-multiset signatures:')
    print(P.groups.value_counts().to_string())
    print(f'\ngap: median={P.gap.median():.1f}  '
          f'vs test-triple adjacent median {np.median(tadj):.1f}')
    print('gap deciles:', np.round(np.percentile(P.gap.dropna(), [0, 25, 50, 75, 100]), 1))
    thr = ba[1]
    print(f'\napplying the train-fitted balanced gap threshold {thr:.1f}s:')
    print(f'  pairs classified "one trial skipped" (slots 0,2): {(P.gap > thr).sum()}/{len(P)}')
    print(f'  pairs classified "adjacent slots"               : {(P.gap <= thr).sum()}/{len(P)}')
    json.dump(dict(train_nf_order=nok / max(npair, 1), test_nf_order=tnok / max(tnpair, 1),
                   gap_thr=float(thr), gap_bal_acc=float(ba[0]),
                   train_adj_median=float(np.median(adj)), train_skip_median=float(np.median(skip)),
                   test_adj_median=float(np.median(tadj)),
                   n_pairs=int(len(P)), n_pairs_I3=int((P.nI == 3).sum()),
                   n_pairs_3groups=int((P.ngroup == 3).sum())),
              open(os.path.join(OUT, 'recon_summary.json'), 'w'), indent=2)
    print('\nwrote slotlab/test_pair_blocks.csv and slotlab/recon_summary.json')


if __name__ == '__main__':
    main()
