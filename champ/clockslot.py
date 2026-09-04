"""Generative prior over WHICH protocol slots a two-clip session block contains.

`solve_emotion` treats a two-clip block whose emotion-option intersection has size 3 as a
three-slot session with one trial withheld, and maximises over the latent
`present in {(0,1), (0,2), (1,2)}`.  With `CHAMP_W_SLOT=0` that latent carries a flat prior,
which is misspecified in a specific, measurable way: two of the three hypotheses put clip 0 in
slot 0, so the flat prior over-predicts the slot-0 (SLOW) manner group by ~5.6pp, and emotion
accuracy in two-clip blocks is 0.7296 against 0.8966 in three-clip blocks.

This module supplies the prior from the recording clock instead, as an explicit generative
likelihood rather than a learned high-dimensional map.  `slotprior.py`'s existing model puts
~1100 correlated features on ~800 samples and recovers only +5 OOF; the signal it has to find
is two-dimensional and is written down directly here.

Evidence, all test-visible (champ/meta.csv, derived from raw modality files only):

  gap        |t0[1] - t0[0]|.  Consecutive trials of a session start a median 99.0s apart
             (train, p5-p95 63-317); with the middle trial withheld the two released clips are
             two trial periods apart, median 203.2s (p5-p95 130-557).  Best balanced
             single-threshold separation 0.834.  Test triples confirm the scale in-domain
             (adjacent median 88.4s, 0/66 negative).

  log ratio  log(nf[0]/nf[1]).  Trial durations fall monotonically with trial index
             (mean frame counts 222 / 178 / 140), so a skipped middle trial roughly squares
             the ratio.  Scale-free, hence robust to the large between-session and
             between-action variation in absolute duration.

  log nf     absolute durations.  Weak on its own because absolute duration is dominated by
             action identity and subject, so it is available but weighted 0 by default.

Nothing here reads a label at inference time; the log-normal parameters come from a `fit` over
training subjects only.  Fitted per fold, exactly like every other `fit_*` in this pipeline.
"""
import os, re, sys, math, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NCLS = [(0, 1), (0, 2), (1, 2)]

W_GAP = float(os.environ.get('CHAMP_SLOT_W_GAP', 1.0))
W_RAT = float(os.environ.get('CHAMP_SLOT_W_RAT', 1.0))
W_ABS = float(os.environ.get('CHAMP_SLOT_W_ABS', 0.0))
# base rate over which trial was withheld.  'flat' is uninformative and is the default: the
# real test's withholding policy is a property of the test set, so fitting a base rate on
# training data (which has no withheld trials at all) would be inventing information.
PRIOR = os.environ.get('CHAMP_SLOT_PRIOR', 'flat')
# Tie the two ADJACENT hypotheses to a common likelihood.  The gap and duration-ratio
# statistics are near-identical for (0,1) and (1,2) -- trial periods and the monotone duration
# decay look the same at either end of the session -- so fitting them separately lets sampling
# noise in those two fits inject a preference the evidence does not support.  Measured: with
# untied fits the model calls (0,1) on 126 of 260 truly-(1,2) pairs.  Tied, the module
# contributes only the adjacency information it genuinely has and leaves which-end to
# solve_emotion's physical evidence.
TIE_ADJ = os.environ.get('CHAMP_SLOT_TIE_ADJ', '1') == '1'
# Hard structural constraint: forbid the middle-withheld hypothesis outright.  Justified for
# the real test by the pi = 1.000 estimate above; keep it off when validating under a
# 'uniform' thinning policy, where it is deliberately wrong a third of the time.
ADJ_ONLY = os.environ.get('CHAMP_SLOT_ADJ_ONLY', '0') == '1'
NEG = -30.0
# CEILING DIAGNOSTIC ONLY -- reads the trial index out of the pseudo-test's `true_path` to
# supply the exact slot set.  This measures the most any slot-set model could ever be worth,
# so that effort here can be justified or abandoned on evidence.  It CANNOT be used on the
# real test: `real_test_view` sets true_path to the bare clip id ('LM_test_0164'), which
# carries no trial index, so ORACLE silently returns None there.  Never set this flag when
# producing a submission.
ORACLE = os.environ.get('CHAMP_SLOT_ORACLE', '0') == '1'
_TRIAL = re.compile(r'/(\d+)-(\d+)-(\d+)$')


# Which-END term.  The clock is chance-level on (0,1) vs (1,2) (measured 0.5077), and that
# binary carries 23 of the 28 questions the oracle slot set is worth, so it is the only part
# of this latent that matters.  Absolute level is the only statistic that can separate them --
# durations and speeds are monotone in trial index -- but absolute level is dominated by what
# the person is doing.  A session's action content is known (all its trials run the same
# script, and pool.solve_block recovers the pool from test-visible evidence), so content is
# regressed out and the residual level classifies which end survived.
# Measured subject-disjoint: 0.6404 against 0.5904 for slotprior's GBM and 0.5000 chance,
# 5/5 folds above chance, label-shuffle control 0.4992, top-quartile confidence 0.7692.
W_END = float(os.environ.get('CHAMP_SLOT_W_END', 1.0))
ACT = ['single', 'multi', 'combination', 'sequence']


def _levels(bf, nf, cols):
    """Per-clip absolute level vector, built from the SAME block_features output the solver
    already has in hand, so fit and inference cannot drift apart."""
    out = []
    for i, d in enumerate(bf):
        v = [math.log(nf[i]) if (np.isfinite(nf[i]) and nf[i] > 0) else np.nan]
        v += [d.get(f'a_{c}', np.nan) for c in cols]
        out.append(v)
    return np.array(out, float)


def _fit_whichend(tr, meta, hold, mfeat, cols):
    from core import block_features
    from collections import defaultdict as _dd
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pool = _dd(set)
    for r in tr[(tr.source == 'HAU') & tr.category.isin(ACT)].itertuples():
        for L in str(r.answer):
            if L in 'ABCD':
                for p in str(getattr(r, L)).split(','):
                    pool[(r.user, r.aa, r.bb)].add(p.strip())
    vocab = sorted({a for s in pool.values() for a in s})
    vi = {a: i for i, a in enumerate(vocab)}

    h = meta[meta.kind == 'train_hau'].dropna(subset=['t0']).copy()
    t3 = h.trial.str.split('-', expand=True)
    h['a'], h['b'], h['c'] = t3[0], t3[1], t3[2].astype(int)
    L, PV = [], []
    for (u, a, b), g in h.groupby(['user', 'a', 'b']):
        if u in hold or (u, a, b) not in pool:
            continue
        g = g.sort_values('c')
        if len(g) != 3:
            continue
        nf = g.nf.to_numpy(float)
        if not np.isfinite(nf).all() or nf.min() <= 0:
            continue
        bf = block_features(list(g.qa_path), mfeat, 3)
        lv = _levels(bf, nf, cols)
        if not np.isfinite(lv).all():
            continue
        pv = np.zeros(len(vocab))
        for act in pool[(u, a, b)]:
            if act in vi:
                pv[vi[act]] = 1.0
        L.append(lv); PV.append(pv)
    if len(L) < 40:
        return None
    L = np.array(L); PV = np.array(PV)
    M = L.mean(1)
    mu, sd = M.mean(0), M.std(0) + 1e-9
    ridge = Ridge(alpha=10.0).fit(PV, (M - mu) / sd)
    Xb, yb = [], []
    for i in range(len(L)):
        exp = ridge.predict(PV[i][None])[0] * sd + mu
        for slots, lab in (((0, 1), 0), ((1, 2), 1)):
            Xb.append((L[i][list(slots)].mean(0) - exp) / sd); yb.append(lab)
    Xb = np.nan_to_num(np.array(Xb))
    sc = StandardScaler().fit(Xb)
    clf = LogisticRegression(C=0.05, max_iter=4000).fit(sc.transform(Xb), np.array(yb))
    return dict(vocab=vocab, vi=vi, ridge=ridge, sc=sc, clf=clf, mu=mu, sd=sd, cols=cols,
                n=len(L))


def _p_late(sp, bf, nf, pool):
    """P(present == (1,2)) i.e. the FIRST trial was withheld.  None when unavailable."""
    we = sp.get('we')
    if we is None or not pool:
        return None
    lv = _levels(bf, nf, we['cols'])
    if not np.isfinite(lv).all():
        return None
    pv = np.zeros(len(we['vocab']))
    for a in pool:
        j = we['vi'].get(a)
        if j is not None:
            pv[j] = 1.0
    if pv.sum() == 0:
        return None
    exp = we['ridge'].predict(pv[None])[0] * we['sd'] + we['mu']
    x = np.nan_to_num(((lv.mean(0) - exp) / we['sd'])[None])
    return float(we['clf'].predict_proba(we['sc'].transform(x))[0, 1])


def _norm(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 5:
        return None
    return float(v.mean()), float(v.std() + 1e-6)


def _lp(x, par):
    """Normal log density, or 0.0 when the statistic or the fit is unavailable."""
    if par is None or not np.isfinite(x):
        return 0.0
    mu, sd = par
    z = (x - mu) / sd
    return -0.5 * z * z - math.log(sd)


def fit(tr, meta, hold_users=()):
    """Log-normal gap / log-ratio / log-duration densities per slot-pair hypothesis."""
    h = meta[meta.kind == 'train_hau'].dropna(subset=['t0']).copy()
    t3 = h.trial.str.split('-', expand=True)
    h['u'], h['a'], h['b'], h['c'] = h.user, t3[0], t3[1], t3[2].astype(int)
    hold = set(hold_users)
    gap = {k: [] for k in NCLS}
    rat = {k: [] for k in NCLS}
    lnf = {0: [], 1: [], 2: []}
    for (u, a, b), g in h.groupby(['u', 'a', 'b']):
        if u in hold:
            continue
        g = g.sort_values('c')
        if len(g) != 3:
            continue
        t = list(g.t0); n = list(g.nf)
        if not all(np.isfinite(t)) or not all(np.isfinite(n)) or min(n) <= 0:
            continue
        for s in range(3):
            lnf[s].append(math.log(n[s]))
        for (i, j) in NCLS:
            d = abs(t[j] - t[i])
            if 0 < d < 1e5:
                gap[(i, j)].append(math.log(d))
            rat[(i, j)].append(math.log(n[i] / n[j]))
    from core import PHYS
    mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)}
             for r in meta.itertuples()}
    cols = [c for c in PHYS if any(c in v for v in mfeat.values())]
    we = _fit_whichend(tr, meta, hold, mfeat, cols) if W_END else None
    return dict(gap={k: _norm(v) for k, v in gap.items()},
                rat={k: _norm(v) for k, v in rat.items()},
                lnf={k: _norm(v) for k, v in lnf.items()},
                aux={r.qa_path: (r.t0, r.nf) for r in meta.itertuples()},
                we=we, n={k: len(v) for k, v in gap.items()})


def logpost(sp, paths, bf=None, pool=None):
    """-> {present_tuple: log posterior}.  `paths` are the two clips' meta keys, in clip order."""
    if sp is None or paths is None or len(paths) != 2:
        return None
    if ORACLE:
        m = [_TRIAL.search(str(p)) for p in paths]
        if not all(m):
            return None                      # real test: no trial index available
        slots = tuple(sorted(int(x.group(3)) - 1 for x in m))
        if slots not in NCLS:
            return None
        return {k: (0.0 if k == slots else NEG) for k in NCLS}
    aux = sp['aux']
    t = [aux.get(p, (np.nan, np.nan))[0] for p in paths]
    n = [aux.get(p, (np.nan, np.nan))[1] for p in paths]
    g = abs(t[1] - t[0]) if all(np.isfinite(t)) else np.nan
    lg = math.log(g) if (np.isfinite(g) and g > 0) else np.nan
    lr = (math.log(n[0] / n[1]) if all(np.isfinite(n)) and min(n) > 0 else np.nan)
    def ev(k):
        s = (W_GAP * _lp(lg, sp['gap'].get(k))
             + W_RAT * _lp(lr, sp['rat'].get(k)))
        if W_ABS:
            for pos, slot in enumerate(k):
                v = math.log(n[pos]) if (np.isfinite(n[pos]) and n[pos] > 0) else np.nan
                s += W_ABS * _lp(v, sp['lnf'].get(slot))
        return s

    if TIE_ADJ:
        # pool the two adjacent hypotheses under one likelihood, then split it evenly
        adj = math.log(0.5 * (math.exp(ev((0, 1))) + math.exp(ev((1, 2))))) \
            if max(ev((0, 1)), ev((1, 2))) > -700 else max(ev((0, 1)), ev((1, 2)))
        ll = {(0, 1): adj + math.log(0.5), (1, 2): adj + math.log(0.5), (0, 2): ev((0, 2))}
    else:
        ll = {k: ev(k) for k in NCLS}
    if ADJ_ONLY:
        ll[(0, 2)] = NEG
    # which-end term: splits the adjacent mass instead of leaving it at log(0.5) each
    if W_END and bf is not None:
        p = _p_late(sp, bf, n, pool)
        if p is not None:
            p = min(max(p, 1e-4), 1 - 1e-4)
            ll[(0, 1)] += W_END * (math.log(1 - p) - math.log(0.5))
            ll[(1, 2)] += W_END * (math.log(p) - math.log(0.5))
    m = max(ll.values())
    z = sum(math.exp(v - m) for v in ll.values())
    return {k: (v - m) - math.log(z) for k, v in ll.items()}


def predict(sp, bf, k, slots, paths=None, pool=None):
    """Interface-compatible with slotprior.predict: only defined for k=2, slots=3."""
    if sp is None or k != 2 or slots != 3:
        return None
    return logpost(sp, paths, bf=bf, pool=pool)
