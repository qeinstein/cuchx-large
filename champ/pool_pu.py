"""Positive/unknown action-pool model.

The original pool model treated the union of answers in a session as a complete
ground-truth pool.  That is too strong: the aligned HARn segments are high
precision but incomplete, while a QA option that was not selected is unknown,
not evidence that the action was absent.

This module keeps the question-constraint decoder in :mod:`pool`, but learns
pool-membership scores from two noisy channels:

* aligned HARn segment presence -- strong positive evidence;
* selected HAU QA actions -- weaker positive evidence for segment misses;
* candidates absent from both -- weak negative evidence only.

All features are test-visible.  The segment/answer targets are used only in the
fold's training subjects by the evaluator.
"""
import itertools
import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool as PL

ACTCATS = PL.ACTCATS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']


def _session_key(user, aa, bb):
    return (str(user), str(aa), str(bb))


def segment_pools(meta, users=None):
    """Return high-precision, possibly incomplete action pools from HARn units."""
    out = defaultdict(set)
    m = meta[meta.kind.eq('train_harn')]
    if users is not None:
        users = set(users)
        m = m[m.user.astype(str).isin(users)]
    for r in m.itertuples():
        if not np.isfinite(float(r.f0)):
            continue
        trial = str(r.trial).split('-')
        if len(trial) < 2 or pd.isna(r.user) or pd.isna(r.action):
            continue
        raw = str(r.action).strip()
        action = VOCAB.get(raw)
        if action is not None:
            out[_session_key(r.user, trial[0], trial[1])].add(action)
    return out


def selected_pools(tr):
    """Union only selected action options, never all distractor options."""
    out = defaultdict(set)
    q = tr[(tr.source == 'HAU') & tr.category.isin(ACTCATS)]
    for r in q.itertuples():
        key = _session_key(r.user, r.aa, r.bb)
        for letter in str(r.answer):
            if letter not in 'ABCD':
                continue
            for action in str(getattr(r, letter)).split(','):
                action = action.strip()
                if action:
                    out[key].add(action)
    return out


def fit_segment_cooc(meta, users=None):
    """Fit action co-occurrence from segment pools, with smoothing."""
    pools = segment_pools(meta, users)
    n = max(1, len(pools))
    marg = Counter()
    joint = defaultdict(Counter)
    for actions in pools.values():
        for a in actions:
            marg[a] += 1
            for b in actions:
                if a != b:
                    joint[a][b] += 1
    return {
        'p': {a: (c + 0.5) / (n + 1.0) for a, c in marg.items()},
        'joint': {a: {b: (c + 0.5) / (marg[a] + 1.0)
                      for b, c in d.items()} for a, d in joint.items()},
        'default': 0.5 / (n + 1.0),
    }


def set_cooc(cooc):
    """Install fold-local co-occurrence priors used by pool.candidate_evidence."""
    PL._COOC = cooc


def evidence(vis, blk, statcache):
    """Candidate features, extending the historical pool features with coverage terms."""
    uni, rows, qs, forced = PL.candidate_evidence(vis, blk, 'oof', statcache)
    if not qs:
        return uni, rows, qs, forced

    # A candidate can occur multiple times in one combination option.  The
    # additional counters below deliberately separate option occurrence from
    # clip/question coverage, which is more stable under a withheld trial.
    q_by_action = defaultdict(list)
    clip_by_action = defaultdict(set)
    cat_clip = defaultdict(lambda: defaultdict(set))
    for r in qs:
        present = set(a for op in PL.qopts(r) for a in op)
        for a in present:
            q_by_action[a].append(r)
            clip_by_action[a].add(r.idx)
            cat_clip[a][r.category].add(r.idx)
    nq = len(qs)
    nclips = len(set(r.idx for r in qs))
    cats = ('single', 'multi', 'combination', 'sequence')
    for a in uni:
        qn = len(q_by_action[a])
        f = rows[a]
        f.update(
            n_q_with_action=qn,
            q_action_frac=qn / max(1, nq),
            n_clips_with_action=len(clip_by_action[a]),
            clip_action_frac=len(clip_by_action[a]) / max(1, nclips),
            log_napp=float(np.log1p(f['napp'])),
            log_uni=float(np.log1p(f['uni'])),
            # These are all derived from visible option text and clip layout.
            first_pos=min((r.idx for r in q_by_action[a]), default=0),
            last_pos=max((r.idx for r in q_by_action[a]), default=0),
        )
        for cat in cats:
            f[f'q_{cat}_frac'] = len(cat_clip[a][cat]) / max(1, nclips)
        # The model should see sequence visibility as strong evidence, but the
        # decoder still handles sequence options as a semantic hard constraint.
        f['sequence_visibility'] = float(len(cat_clip[a]['sequence']))
    return uni, rows, qs, forced


def _variants(blk):
    """Full block plus the pair/sub-block regimes present in the real test."""
    out = [(list(blk), 'full')]
    if len(blk) >= 3:
        ks = (2, len(blk) - 1) if len(blk) > 3 else (2,)
        for k in ks:
            for sub in itertools.combinations(blk, k):
                out.append((list(sub), f'sub{k}'))
    return out


def fit_model(blocks_by_user, latent, weak, statcache):
    """Fit a fold-local noisy-channel pool classifier.

    ``latent`` comes from aligned HARn segments and ``weak`` from selected HAU
    answers.  Weights encode the asymmetry: segment positives are trusted,
    weak QA positives are useful but noisy, and negatives are mostly unknown.
    """
    X, y, sw = [], [], []
    seg_weight = float(os.environ.get('CHAMP_POOL_PU_SEG_WEIGHT', '1.00'))
    qa_weight = float(os.environ.get('CHAMP_POOL_PU_QA_WEIGHT', '0.22'))
    neg_weight = float(os.environ.get('CHAMP_POOL_PU_NEG_WEIGHT', '0.16'))
    for key, info in blocks_by_user.items():
        vis, blk, _ = info
        seg = set(latent.get(key, ()))
        qa = set(weak.get(key, ()))
        has_seg = key in latent
        for sub, _kind in _variants(blk):
            uni, rows, qs, _forced = evidence(vis, sub, statcache)
            for action in uni:
                if action in seg:
                    label, weight = 1, seg_weight
                elif action in qa:
                    # Segment files miss some action units; selected QA actions
                    # are therefore soft positives, not hard contradictions.
                    label, weight = 1, qa_weight if has_seg else qa_weight * 0.45
                else:
                    # Absence from both channels is only weak negative evidence.
                    label, weight = 0, neg_weight if has_seg else neg_weight * 0.4
                X.append(rows[action]); y.append(label); sw.append(weight)

    Xd = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
    cols = [c for c in Xd.columns if np.isfinite(
        pd.to_numeric(Xd[c], errors='coerce').to_numpy(float)).any()]
    if not cols or len(set(y)) < 2:
        raise RuntimeError('insufficient noisy-channel training examples')

    from sklearn.ensemble import HistGradientBoostingClassifier
    max_iter = int(os.environ.get('CHAMP_POOL_PU_MAX_ITER', '350'))
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.055, max_depth=4,
        min_samples_leaf=12, l2_regularization=2.0, random_state=0)
    clf.fit(Xd[cols].to_numpy(float), np.asarray(y), sample_weight=np.asarray(sw))
    return clf, cols


def make_scorer(clf, cols, clip=6.0):
    def scorer(rows, uni):
        Xd = pd.DataFrame([rows[a] for a in uni]).replace(
            [np.inf, -np.inf], np.nan).reindex(columns=cols)
        p = np.clip(clf.predict_proba(Xd.to_numpy(float))[:, 1], 1e-4, 1 - 1e-4)
        return {a: float(np.clip(np.log(p[i] / (1 - p[i])), -clip, clip))
                for i, a in enumerate(uni)}
    return scorer


def _milp_component(atoms, cq, forced, lo):
    """Exact large-component solver using binary atom/option variables.

    For each option ``o`` introduce z_o = AND(x_a for a in o).  The question
    constraints are then linear: exactly one z for single/combination and at
    least one z for multi.  This avoids the historical 2**22 dense matrix
    allocation while preserving its objective and semantics.
    """
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix

    idx = {a: i for i, a in enumerate(atoms)}
    options = []
    qopts_fixed = []
    for cat, ops in cq:
        vals = []
        for op in ops:
            free = []
            valid = True
            for a in op:
                if a in idx:
                    free.append(a)
                elif a not in forced:
                    valid = False
            if valid:
                options.append((cat, free))
                vals.append(len(options) - 1)
        qopts_fixed.append((cat, vals))

    n = len(atoms)
    nv = n + len(options)
    if nv == n:
        return {a for a in atoms if lo.get(a, 0.0) > 0}, False
    c = np.zeros(nv, float)
    c[:n] = [-2.0 * lo.get(a, 0.0) for a in atoms]
    rows = []
    lbs = []
    ubs = []
    # z_o <= x_a and z_o >= sum(x_a) - |o| + 1
    for oi, (_cat, free) in enumerate(options):
        z = n + oi
        for a in free:
            row = {}
            row[z] = 1.0; row[idx[a]] = -1.0
            rows.append(row); lbs.append(-np.inf); ubs.append(0.0)
        if free:
            row = {z: -1.0}
            for a in free:
                row[idx[a]] = row.get(idx[a], 0.0) + 1.0
            rows.append(row); lbs.append(-np.inf); ubs.append(len(free) - 1.0)
    # The fixed options are already true because their non-free atoms are in
    # `forced`; count them as constants in the question constraint.
    for cat, vals in qopts_fixed:
        const = sum(1 for oi in vals if not options[oi][1])
        use = [oi for oi in vals if options[oi][1]]
        row = {n + oi: 1.0 for oi in use}
        if cat in ('single', 'combination'):
            rhs = 1.0 - const
            rows.append(row); lbs.append(rhs); ubs.append(rhs)
        elif cat == 'multi':
            rows.append(row); lbs.append(1.0 - const); ubs.append(np.inf)
    if not rows:
        return {a for a in atoms if lo.get(a, 0.0) > 0}, False
    A = lil_matrix((len(rows), nv), dtype=float)
    for ri, row in enumerate(rows):
        for ci, val in row.items():
            A[ri, ci] = val
    res = milp(c=c, integrality=np.ones(nv), bounds=Bounds(0, 1),
               constraints=LinearConstraint(A.tocsr(), np.asarray(lbs), np.asarray(ubs)),
               options={'time_limit': 1.0})
    if not res.success or res.x is None:
        return None, False
    return {atoms[i] for i in range(n) if res.x[i] > 0.5}, True


def solve_block(vis, blk, statcache, scorer):
    """Decode with the existing exact question constraints and PU scores."""
    uni, rows, qs, forced = evidence(vis, blk, statcache)
    if not qs:
        return {}, None
    lo = scorer(rows, uni)
    bias = PL.POOL_BIAS.get(str(len(blk)), 0.0)
    if bias:
        lo = {a: v + bias for a, v in lo.items()}
    free, comps, _fixed_q = PL._components(qs, uni, forced)
    best = set(forced)
    allsat = True
    for atoms, cq in comps:
        # Exhaustive enumeration is faster and simpler for the normal small
        # components.  MILP is exact and much cheaper for the pathological
        # 17+ atom components that previously dominated the audit runtime.
        if len(atoms) <= 16:
            sel, sat = PL._enum_component(atoms, cq, forced, lo)
        else:
            sel, sat = _milp_component(atoms, cq, forced, lo)
        if not sat or sel is None:
            allsat = False
            sel = {a for a in atoms if lo.get(a, -9) > 0}
        best |= sel
    out = {}
    for r in qs:
        ops = PL.qopts(r)
        hit = [i for i, op in enumerate(ops) if all(a in best for a in op)]
        if r.category in ('single', 'combination'):
            if len(hit) != 1:
                hit = [max(range(4), key=lambda i: sum(lo.get(a, 0.0) for a in ops[i]))]
            out[r.qa_id] = 'ABCD'[hit[0]]
        elif r.category == 'multi':
            if not hit:
                hit = [max(range(4), key=lambda i: sum(lo.get(a, 0.0) for a in ops[i]))]
            out[r.qa_id] = ''.join('ABCD'[i] for i in sorted(hit))
    return out, dict(pool=sorted(best), nsol=int(allsat), uni=len(uni),
                     free=len(free), ncomp=len(comps),
                     maxcomp=max((len(a) for a, _ in comps), default=0))
