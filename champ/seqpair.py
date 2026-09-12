"""ATTACK 1 - pairwise temporal precedence model for the sequence questions.

Error diagnosis that motivates this: of the 132 held-out sequence errors, 83 (62.9%) are a
single adjacent transposition, and 0 involve an action occurring more than once.  The gross
ordering is already right; individual pairwise comparisons are what fail.  So model the
comparison directly instead of relying on argmax frame classification.

For a clip and an ordered pair of candidate actions (Ai, Aj) we learn
    P(Ai starts before Aj | dense temporal evidence, action identities, duration priors)
and then pick the permutation of the four candidates maximising the sum of pairwise
log-likelihoods (a 24-way enumeration, so cycles are resolved by maximum likelihood).

Training labels come from two legitimate sources, both subject-disjoint at fit time:
  * HARn->HAU aligned segment onsets: for every session and every pair of segmented
    actions, which one started first (~4.8k pairs)
  * the sequence questions' own answers: the order of all four candidates (~1.8k pairs),
    which also covers actions that were never segmented
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D
from core import load_all, opts
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
OPT2CLS = {V[a]: D.A2I[a] for a in D.ACTIONS}
CLS2OPT = {v: k for k, v in OPT2CLS.items()}
_LG = None
_LOC = None
_PC = {}


_LOCS = {}
LOC_SOURCES = [s for s in os.environ.get(
    'CHAMP_LOC_SRC', 'loc_maps,loc_fused_maps').split(',') if s]


def loc_map(split, p, name='loc_maps'):
    """Per-action activation map from a candidate-conditioned localizer."""
    if name not in _LOCS:
        f = os.path.join(ROOT, 'champ', f'{name}.npz')
        _LOCS[name] = np.load(f) if os.path.exists(f) else {}
    Z = _LOCS[name]
    k = f'{split}|{p}'
    return Z[k] if (hasattr(Z, 'files') and k in Z.files) else None


def logits(split, p):
    global _LG
    if _LG is None:
        _LG = np.load(os.path.join(ROOT, 'champ',
                                   os.environ.get('CHAMP_LOGITS', 'dense_logits.npz')))
    k = f'{split}|{p}'
    return _LG[k] if k in _LG else None


def clip_profiles(split, p, smooth=5):
    """Per-class normalised temporal distribution over the clip, plus summary positions."""
    key = (split, p)
    if key in _PC:
        return _PC[key]
    lp = logits(split, p)
    if lp is None:
        _PC[key] = None
        return None
    P = np.exp(lp[:D.BG])                      # 40 x T, action classes only
    T = P.shape[1]
    if T > smooth:
        k = np.ones(smooth) / smooth
        P = np.stack([np.convolve(P[c], k, mode='same') for c in range(P.shape[0])])
    mass = P.sum(1, keepdims=True) + 1e-9
    Q = P / mass                               # temporal distribution per action
    t = np.arange(T) / max(1, T - 1)
    cum = np.cumsum(Q, axis=1)
    locs = {}
    for nm in LOC_SOURCES:
        S = loc_map(split, p, nm)
        if S is None or S.shape[1] != T:
            continue
        Ql = S / (S.sum(1, keepdims=True) + 1e-9)
        locs[nm] = dict(Q=Ql, cum=np.cumsum(Ql, axis=1),
                        centroid=(Ql * t).sum(1), mx=S.max(1),
                        onset=np.array([np.argmax(S[c] >= 0.5 * max(S[c].max(), 1e-6))
                                        / max(1, T - 1) for c in range(S.shape[0])]))
    loc = locs.get(LOC_SOURCES[0]) if LOC_SOURCES else None
    prof = dict(T=T, Q=Q, cum=cum, t=t, loc=loc, locs=locs,
                centroid=(Q * t).sum(1),
                peak=P.argmax(1) / max(1, T - 1),
                mx=P.max(1), mean=P.mean(1),
                mass=mass.ravel() / T,
                # first time the smoothed score reaches half its own maximum
                onset=np.array([np.argmax(P[c] >= 0.5 * P[c].max()) / max(1, T - 1)
                                for c in range(P.shape[0])]),
                offset=np.array([(T - 1 - np.argmax(P[c][::-1] >= 0.5 * P[c].max()))
                                 / max(1, T - 1) for c in range(P.shape[0])]))
    _PC[key] = prof
    return prof


def pair_feats(prof, ci, cj, dpri, apri):
    """Features for the ordered pair (class ci, class cj).  All test-visible."""
    Q, cum, t = prof['Q'], prof['cum'], prof['t']
    qi, qj = Q[ci], Q[cj]
    # P(t_i < t_j) under independent draws from the two temporal distributions
    p_before = float((qi * np.concatenate([[0.0], cum[cj][:-1]])).sum())
    p_after = float((qj * np.concatenate([[0.0], cum[ci][:-1]])).sum())
    tot = p_before + p_after + 1e-9
    f = dict(
        p_before=p_before, p_after=p_after, p_before_n=p_before / tot,
        d_centroid=prof['centroid'][ci] - prof['centroid'][cj],
        d_peak=prof['peak'][ci] - prof['peak'][cj],
        d_onset=prof['onset'][ci] - prof['onset'][cj],
        d_offset=prof['offset'][ci] - prof['offset'][cj],
        cent_i=prof['centroid'][ci], cent_j=prof['centroid'][cj],
        onset_i=prof['onset'][ci], onset_j=prof['onset'][cj],
        mx_i=prof['mx'][ci], mx_j=prof['mx'][cj],
        mass_i=prof['mass'][ci], mass_j=prof['mass'][cj],
        mean_i=prof['mean'][ci], mean_j=prof['mean'][cj],
        overlap=float(np.minimum(qi, qj).sum()),
        T=prof['T'],
        # global precedence prior from the segment corpus, and duration priors
        prior_ij=apri.get((ci, cj), 0.5),
        dur_i=dpri.get(ci, np.nan), dur_j=dpri.get(cj, np.nan),
        cls_i=ci, cls_j=cj,
    )
    for nm in LOC_SOURCES:
        lo = prof.get('locs', {}).get(nm)
        pre = 'L' if nm == LOC_SOURCES[0] else nm.replace('loc_', '').replace('_maps', '')
        if lo is None:
            for kk in ('p_before', 'p_before_n', 'd_centroid', 'd_onset', 'cent_i',
                       'cent_j', 'onset_i', 'onset_j', 'mx_i', 'mx_j', 'overlap'):
                f[f'{pre}_{kk}'] = np.nan
            continue
        li, lj = lo['Q'][ci], lo['Q'][cj]
        lb = float((li * np.concatenate([[0.0], lo['cum'][cj][:-1]])).sum())
        la = float((lj * np.concatenate([[0.0], lo['cum'][ci][:-1]])).sum())
        f.update({f'{pre}_p_before': lb, f'{pre}_p_before_n': lb / (lb + la + 1e-9),
                  f'{pre}_d_centroid': lo['centroid'][ci] - lo['centroid'][cj],
                  f'{pre}_d_onset': lo['onset'][ci] - lo['onset'][cj],
                  f'{pre}_cent_i': lo['centroid'][ci], f'{pre}_cent_j': lo['centroid'][cj],
                  f'{pre}_onset_i': lo['onset'][ci], f'{pre}_onset_j': lo['onset'][cj],
                  f'{pre}_mx_i': lo['mx'][ci], f'{pre}_mx_j': lo['mx'][cj],
                  f'{pre}_overlap': float(np.minimum(li, lj).sum())})
    lo = None
    if lo is not None:
        li, lj = lo['Q'][ci], lo['Q'][cj]
        lb = float((li * np.concatenate([[0.0], lo['cum'][cj][:-1]])).sum())
        la = float((lj * np.concatenate([[0.0], lo['cum'][ci][:-1]])).sum())
        f.update(L_p_before=lb, L_p_before_n=lb / (lb + la + 1e-9),
                 L_d_centroid=lo['centroid'][ci] - lo['centroid'][cj],
                 L_d_onset=lo['onset'][ci] - lo['onset'][cj],
                 L_cent_i=lo['centroid'][ci], L_cent_j=lo['centroid'][cj],
                 L_onset_i=lo['onset'][ci], L_onset_j=lo['onset'][cj],
                 L_mx_i=lo['mx'][ci], L_mx_j=lo['mx'][cj],
                 L_overlap=float(np.minimum(li, lj).sum()))
    else:
        for k in ('L_p_before', 'L_p_before_n', 'L_d_centroid', 'L_d_onset', 'L_cent_i',
                  'L_cent_j', 'L_onset_i', 'L_onset_j', 'L_mx_i', 'L_mx_j', 'L_overlap'):
            f[k] = np.nan
    return f


# --------------------------------------------------------------------------- priors / labels
def build_segment_data(meta):
    """(user, trial) -> {class: (f0, f1)} from the HARn->HAU aligned segments."""
    segs = defaultdict(dict)
    for r in meta[meta.kind == 'train_harn'].itertuples():
        if not np.isfinite(r.f0):
            continue
        c = D.A2I.get(r.action)
        if c is not None:
            segs[(r.user, r.trial)][c] = (r.f0, r.f1)
    return segs


def fit_priors(segs, keys):
    """Duration prior per action and global pairwise precedence prior, from `keys` only."""
    dur = defaultdict(list)
    pc = defaultdict(int)
    for k in keys:
        S = segs.get(k, {})
        for c, (f0, f1) in S.items():
            dur[c].append(f1 - f0)
        for a, b in itertools.permutations(S, 2):
            if S[a][0] < S[b][0]:
                pc[(a, b)] += 1
    dpri = {c: float(np.median(v)) for c, v in dur.items()}
    apri = {}
    for (a, b) in set(list(pc) + [(b, a) for a, b in pc]):
        n1, n2 = pc.get((a, b), 0), pc.get((b, a), 0)
        apri[(a, b)] = (n1 + 1.0) / (n1 + n2 + 2.0)
    return dpri, apri


def build_training_pairs(tr, meta, segs, users, split='oof'):
    """Ordered-pair examples from segment onsets plus from the sequence answers."""
    hau = meta[meta.kind == 'train_hau']
    key_of = {r.qa_path: (r.user, r.trial) for r in hau.itertuples()}
    seq = tr[(tr.source == 'HAU') & (tr.category == 'sequence') & tr.user.isin(users)]
    seq_order = {}
    for _, r in seq.iterrows():
        oo = opts(r)
        cl = [OPT2CLS.get(oo['ABCD'.index(L)]) for L in str(r['answer']) if L in 'ABCD']
        if len(cl) == 4 and all(c is not None for c in cl):
            seq_order[r.path] = cl
    keys = [k for k in segs if k[0] in users]
    dpri, apri = fit_priors(segs, keys)
    X, y, src = [], [], []
    for r in hau.itertuples():
        if r.user not in users:
            continue
        prof = clip_profiles(split, r.qa_path)
        if prof is None:
            continue
        k = key_of[r.qa_path]
        S = segs.get(k, {})
        seen = set()
        # (a) segment-derived order
        for a, b in itertools.permutations(sorted(S), 2):
            X.append(pair_feats(prof, a, b, dpri, apri))
            y.append(int(S[a][0] < S[b][0])); src.append('seg')
            seen.add((a, b))
        # (b) sequence-answer-derived order (covers unsegmented actions)
        cl = seq_order.get(r.qa_path)
        if cl:
            pos = {c: i for i, c in enumerate(cl)}
            for a, b in itertools.permutations(cl, 2):
                if (a, b) in seen:
                    continue
                X.append(pair_feats(prof, a, b, dpri, apri))
                y.append(int(pos[a] < pos[b])); src.append('qa')
    return pd.DataFrame(X), np.array(y), np.array(src), dpri, apri


def fit_pair_model(Xd, y):
    # an all-NaN column (a localizer source with no maps) breaks the histogram binner
    Xd = Xd.loc[:, Xd.notna().any()]
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        # Keep the original setting for reproducibility, but allow constrained
        # runners to perform the same audit without exhausting their memory/time.
        max_iter=int(os.environ.get('CHAMP_SEQPAIR_MAX_ITER', '500')),
        learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        categorical_features=[Xd.columns.get_loc('cls_i'), Xd.columns.get_loc('cls_j')],
        random_state=0)
    clf.fit(Xd.to_numpy(float), y)
    return clf, list(Xd.columns)


def decode(prof, cls4, clf, cols, dpri, apri):
    """Best permutation of the four candidates under the pairwise log-likelihoods."""
    rows, idx = [], {}
    for a, b in itertools.permutations(range(4), 2):
        idx[(a, b)] = len(rows)
        rows.append(pair_feats(prof, cls4[a], cls4[b], dpri, apri))
    Pm = clf.predict_proba(pd.DataFrame(rows).reindex(columns=cols).to_numpy(float))[:, 1]
    Pm = np.clip(Pm, 1e-6, 1 - 1e-6)
    best, bs = None, -np.inf
    for perm in itertools.permutations(range(4)):
        s = sum(np.log(Pm[idx[(perm[i], perm[j])]])
                for i in range(4) for j in range(i + 1, 4))
        if s > bs:
            bs, best = s, perm
    return best, Pm, idx
