"""Which END of the session was withheld?  Content-normalised absolute level.

Measured bottleneck (slotlab/log_ends_*.txt, pair-thinned, 'ends' policy, n=190 two-clip
emotion questions):

    baseline flat prior      0.7053
    adjacency-only prior     0.7316   (+5 OOF)
    ORACLE slot set          0.8526   (+28 OOF)

So 23 of the 28 available questions need the *identity* of the withheld end, not merely the
fact that the pair is adjacent.  The recording clock cannot supply it -- (0,1) and (1,2) are
both one trial period apart, and the clock model scores 0.5077 on that binary, i.e. chance.
The duration RATIO cannot either: consecutive-trial ratios are 1.21 and 1.29 against 1.55 for
a skipped middle, so the ratio identifies adjacency and nothing more.

Only the ABSOLUTE level can distinguish them -- trial durations and speeds are monotone in
trial index (frame counts 222 / 178 / 140) -- but absolute level is dominated by *what the
person is doing*, which is why raw absolute features have never worked here.  The exploitable
asymmetry is that a session's action content is KNOWN: all trials of a session run the same
script, and `pool.solve_block` recovers that action pool from test-visible evidence at high
accuracy.  So content can be regressed out:

    residual = observed level - E[level | session action pool]

and the residual mean of the two released clips should sit high when they are trials (1,2) and
low when they are trials (2,3).

This is a different claim from the eight falsified manner probes.  Those asked "can motion
predict the manner label"; this asks "can content-normalised absolute level identify the
protocol slot", against a bottleneck that has now been quantified rather than assumed.
Reported against the two relevant references: chance (0.5) and slotprior's GBM (0.5904).

Uses the TRUE action pool, which is the optimistic case; if the signal is absent here it is
absent with a recovered pool too, so this is a screen, not a shippable estimate.
"""
import os, sys, json, itertools
from collections import defaultdict
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all, PHYS
from pseudotest import folds
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

ACT = ['single', 'multi', 'combination', 'sequence']


def true_pool(tr):
    pool = defaultdict(set)
    for r in tr[(tr.source == 'HAU') & tr.category.isin(ACT)].itertuples():
        for L in str(r.answer):
            if L in 'ABCD':
                for p in str(getattr(r, L)).split(','):
                    pool[(r.user, r.aa, r.bb)].add(p.strip())
    return pool


def main():
    tr, te, meta = load_all()
    pool = true_pool(tr)
    vocab = sorted({a for s in pool.values() for a in s})
    vi = {a: i for i, a in enumerate(vocab)}
    print(f'action vocabulary: {len(vocab)}')

    # ---- per-session, per-slot feature matrix
    h = meta[meta.kind == 'train_hau'].dropna(subset=['t0']).copy()
    t3 = h.trial.str.split('-', expand=True)
    h['a'], h['b'], h['c'] = t3[0], t3[1], t3[2].astype(int)
    feats = [c for c in PHYS if c in h.columns]
    fe = pd.read_csv(os.path.join(ROOT, 'champ', 'feats.csv'))
    h = h.merge(fe, on='unit_dir', how='left', suffixes=('', '_f'))
    feats = [c for c in PHYS if c in h.columns]
    print(f'physical features available: {len(feats)}')

    sess = []
    for (u, a, b), g in h.groupby(['user', 'a', 'b']):
        g = g.sort_values('c')
        if len(g) != 3 or (u, a, b) not in pool:
            continue
        X = g[feats].to_numpy(float)
        nf = g.nf.to_numpy(float)
        if not np.isfinite(nf).all() or nf.min() <= 0:
            continue
        pv = np.zeros(len(vocab))
        for act in pool[(u, a, b)]:
            if act in vi:
                pv[vi[act]] = 1.0
        sess.append(dict(u=u, key=(u, a, b), X=X, lnf=np.log(nf), pool=pv))
    print(f'usable 3-trial sessions: {len(sess)}')

    # level vector per clip: log duration + log1p of each physical feature
    def levels(s):
        L = [s['lnf']]
        for j in range(len(feats)):
            v = s['X'][:, j]
            v = np.where(np.isfinite(v), v, np.nan)
            L.append(v)
        return np.vstack(L).T                      # (3, 1+F)

    users = sorted({s['u'] for s in sess})
    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        hold = set(hold)
        trn = [s for s in sess if s['u'] not in hold]
        tst = [s for s in sess if s['u'] in hold]
        if not tst:
            continue
        # ---- content model: E[session-mean level | action pool], fitted on training users
        Ptr = np.array([s['pool'] for s in trn])
        Ltr = np.array([np.nanmean(levels(s), 0) for s in trn])
        keep = np.isfinite(Ltr).all(0)
        Ltr = Ltr[:, keep]
        mu, sd = np.nanmean(Ltr, 0), np.nanstd(Ltr, 0) + 1e-9
        ridge = Ridge(alpha=10.0).fit(Ptr, (Ltr - mu) / sd)

        # ---- training examples for the which-end binary
        def example(s, slots):
            Lv = levels(s)[:, keep]
            obs = np.nanmean(Lv[list(slots)], 0)
            exp = ridge.predict(s['pool'][None])[0] * sd + mu
            return (obs - exp) / sd

        Xb, yb = [], []
        for s in trn:
            Xb.append(example(s, (0, 1))); yb.append(0)
            Xb.append(example(s, (1, 2))); yb.append(1)
        Xb = np.nan_to_num(np.array(Xb)); yb = np.array(yb)
        sc = StandardScaler().fit(Xb)
        clf = LogisticRegression(C=0.05, max_iter=4000).fit(sc.transform(Xb), yb)

        for s in tst:
            for slots, lab in (((0, 1), 0), ((1, 2), 1)):
                x = sc.transform(np.nan_to_num(example(s, slots))[None])
                p = clf.predict_proba(x)[0, 1]
                rows.append(dict(fold=fi, u=s['u'], truth=lab, p=p,
                                 pred=int(p > 0.5), correct=int((p > 0.5) == lab)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, 'slotlab', 'whichend_oof.csv'), index=False)

    print(f'\n===== which-end binary, subject-disjoint (n={len(d)}) =====')
    print(f'  chance                        0.5000')
    print(f'  slotprior GBM (measured)      0.5904')
    print(f'  content-normalised level      {d.correct.mean():.4f}   '
          f'({d.correct.sum()}/{len(d)})')
    print('\n  per fold: ' + '  '.join(
        f'f{f}={g.correct.mean():.3f}' for f, g in d.groupby('fold')))
    print('\n  by confidence stratum (|p-0.5|):')
    d['conf'] = (d.p - 0.5).abs()
    d['q'] = pd.qcut(d.conf, 4, labels=False, duplicates='drop')
    for q, g in d.groupby('q'):
        print(f'    q{int(q)} (|p-.5| >= {g.conf.min():.3f})  '
              f'{g.correct.sum():4d}/{len(g):4d} = {g.correct.mean():.4f}')
    # label-shuffle control
    rng = np.random.default_rng(0)
    sh = [np.mean(rng.permutation(d.truth.values) == d.pred.values) for _ in range(200)]
    print(f'\n  label-shuffle control: {np.mean(sh):.4f} +- {np.std(sh):.4f}'
          f'  (must sit at 0.5 if the estimator is clean)')
    json.dump(dict(n=len(d), acc=float(d.correct.mean()),
                   per_fold={int(f): float(g.correct.mean()) for f, g in d.groupby('fold')},
                   shuffle=float(np.mean(sh))),
              open(os.path.join(ROOT, 'slotlab', 'whichend_summary.json'), 'w'), indent=2)
    print('\nwrote slotlab/whichend_oof.csv and slotlab/whichend_summary.json')


if __name__ == '__main__':
    main()
