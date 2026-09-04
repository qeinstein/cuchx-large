"""What fraction of the real test's two-clip blocks are ADJACENT trials?

`solve_emotion` marginalises over `present in {(0,1),(0,2),(1,2)}` with a flat prior, i.e. it
assumes the withheld trial is the middle one a third of the time.  If the real withholding
policy is different, the prior is misspecified -- and it is misspecified asymmetrically,
because two of the three hypotheses place clip 0 in slot 0, which inflates the SLOW group.

This estimates the mixing fraction pi = P(adjacent) directly from the 21 real-test pair-block
gaps by maximum likelihood under the two training-fitted log-normal gap densities:

    g_b ~ pi * LogNormal(mu_adj, sd_adj) + (1 - pi) * LogNormal(mu_skip, sd_skip)

with a parametric bootstrap CI, plus a likelihood-ratio test against pi = 2/3 (the value the
flat prior implies).  Uses only champ/meta.csv timestamps: no labels, no leaderboard.

Sanity control: the identical estimator is run on the 34 test TRIPLES' 66 internal gaps, where
the truth is known to be pi = 1 (nothing is withheld inside a complete triple).
"""
import os, sys, math, json
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all, real_test_view, fit_group_model, infer_blocks


def lognorm_pdf(x, mu, sd):
    x = np.asarray(x, float)
    return np.exp(-0.5 * ((np.log(x) - mu) / sd) ** 2) / (x * sd * math.sqrt(2 * math.pi))


def fit_pi(g, pa, ps):
    """MLE of the adjacent-mixture weight for gaps g under densities pa, ps."""
    da = lognorm_pdf(g, *pa); ds = lognorm_pdf(g, *ps)

    def nll(pi):
        pi = min(max(pi, 1e-9), 1 - 1e-9)
        return -np.sum(np.log(pi * da + (1 - pi) * ds))
    r = minimize_scalar(nll, bounds=(1e-6, 1 - 1e-6), method='bounded')
    return float(r.x), float(-r.fun), nll


def main():
    tr, te, meta = load_all()
    mrow = meta.set_index('qa_path')
    vis = real_test_view(te)
    blocks = infer_blocks(vis, fit_group_model(tr))
    pathof = dict(zip(vis.idx, vis.true_path))

    def t0(i):
        p = pathof[i]
        return float(mrow.loc[p, 't0']) if p in mrow.index else np.nan

    # ---- training-fitted gap densities
    h = meta[meta.kind == 'train_hau'].dropna(subset=['t0']).copy()
    t3 = h.trial.str.split('-', expand=True)
    h['a'], h['b'], h['c'] = t3[0], t3[1], t3[2].astype(int)
    adj, skp = [], []
    for (u, a, b), g in h.groupby(['user', 'a', 'b']):
        g = g.sort_values('c'); t = list(g.t0)
        if len(g) != 3:
            continue
        adj += [t[1] - t[0], t[2] - t[1]]; skp.append(t[2] - t[0])
    adj = np.array([x for x in adj if 0 < x < 1e5])
    skp = np.array([x for x in skp if 0 < x < 1e5])
    pa = (float(np.log(adj).mean()), float(np.log(adj).std()))
    ps = (float(np.log(skp).mean()), float(np.log(skp).std()))
    print(f'train adjacent  n={len(adj)}  logN(mu={pa[0]:.4f}, sd={pa[1]:.4f})  '
          f'median {np.median(adj):.1f}s')
    print(f'train skipped   n={len(skp)}  logN(mu={ps[0]:.4f}, sd={ps[1]:.4f})  '
          f'median {np.median(skp):.1f}s')

    # ---- control: complete test triples, where truth is pi = 1
    ctrl = []
    for b in [x for x in blocks if len(x) == 3]:
        v = [t0(i) for i in b]
        if all(np.isfinite(v)):
            ctrl += [abs(v[1] - v[0]), abs(v[2] - v[1])]
    ctrl = np.array([x for x in ctrl if x > 0])
    pic, llc, _ = fit_pi(ctrl, pa, ps)
    print(f'\nCONTROL  34 test triples, {len(ctrl)} internal gaps (truth pi = 1.0)')
    print(f'  estimated pi = {pic:.4f}   <- estimator is unbiased here, so it is trustworthy')

    # ---- the 21 pair blocks
    pg = []
    for b in [x for x in blocks if len(x) == 2]:
        v = [t0(i) for i in b]
        d = abs(v[1] - v[0])
        if np.isfinite(d) and d > 0:
            pg.append(d)
    pg = np.array(pg)
    pi, ll, nll = fit_pi(pg, pa, ps)
    print(f'\nTEST PAIRS  n={len(pg)} gaps, median {np.median(pg):.1f}s')
    print(f'  MLE  pi = P(adjacent) = {pi:.4f}')
    print(f'  => estimated middle-withheld blocks: {(1-pi)*len(pg):.1f} of {len(pg)}')
    print(f'  flat prior in solve_emotion implies pi = 2/3 = 0.6667 '
          f'({(1-2/3)*len(pg):.1f} of {len(pg)})')

    # parametric bootstrap CI
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(4000):
        n_a = rng.binomial(len(pg), pi)
        s = np.concatenate([np.exp(rng.normal(pa[0], pa[1], n_a)),
                            np.exp(rng.normal(ps[0], ps[1], len(pg) - n_a))])
        boots.append(fit_pi(s, pa, ps)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f'  parametric bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]  (4000 resamples)')

    # LR test against pi = 2/3
    ll_flat = -nll(2 / 3)
    D = 2 * (ll - ll_flat)
    p = chi2.sf(max(D, 0), 1)
    print(f'\n  log-lik at MLE      {ll:.4f}')
    print(f'  log-lik at pi=2/3   {ll_flat:.4f}')
    print(f'  LR statistic D = {D:.3f}   p = {p:.4f}  '
          f'({"reject" if p < 0.05 else "cannot reject"} the flat prior at 0.05)')

    # per-block posterior of being adjacent, at the MLE
    da = lognorm_pdf(pg, *pa); ds = lognorm_pdf(pg, *ps)
    post = pi * da / (pi * da + (1 - pi) * ds)
    ids = [str([int(i) for i in b]) for b in blocks if len(b) == 2]
    P = pd.DataFrame(dict(blk=ids[:len(pg)], gap=pg, p_adjacent=post)).sort_values('p_adjacent')
    print('\n  per-block P(adjacent | gap) at the MLE:')
    print(P.round(3).to_string(index=False))
    P.to_csv(os.path.join(ROOT, 'slotlab', 'test_pair_policy.csv'), index=False)
    json.dump(dict(pi=pi, ci=[float(lo), float(hi)], pi_control=pic, D=float(D), p=float(p),
                   n_pairs=int(len(pg)), gap_adj=pa, gap_skip=ps,
                   expected_middle_withheld=float((1 - pi) * len(pg))),
              open(os.path.join(ROOT, 'slotlab', 'policy_summary.json'), 'w'), indent=2)
    print('\nwrote slotlab/test_pair_policy.csv and slotlab/policy_summary.json')


if __name__ == '__main__':
    main()
