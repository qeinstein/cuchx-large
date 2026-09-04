"""Sequence research harness.

Latent-script hypothesis: each HAU session (user, a, b) has ONE total order over its action
pool, and every sequence question in the session is the restriction of that order to its four
options.  Verified on train: 104/104 triples pairwise-order-conflict-free over 1848 pairs.

Consequence: the 2-3 sequence questions of a block are 2-3 noisy views of one latent object and
must be decoded jointly, not independently as the champion does.
"""
import os, sys, json, itertools, collections
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
import dense as D
import decode as DC
from core import load_all, opts

VOCAB = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))
OPT2CLS = {VOCAB['HARN2HAU'][a]: D.A2I[a] for a in D.ACTIONS}
PERMS = [''.join(p) for p in itertools.permutations('ABCD')]


def seq_rows(tr):
    s = tr[(tr.source == 'HAU') & (tr.category == 'sequence')].copy()
    s['blk'] = list(zip(s.user, s.aa, s.bb))
    return s


def dp_scores(lp, ci):
    """24-vector of monotone-DP log scores, one per permutation of the 4 options."""
    _, _, sc = DC.seq_order_dp(lp, ci)
    v = np.array([sc[tuple(ord(c) - 65 for c in p)] for p in PERMS], float)
    return v


def centroid_perm(lp, ci):
    st = DC.presence_stats(lp)
    return ''.join('ABCD'[i] for i in np.argsort([st['centroid'][c] for c in ci]))


def pairwise_from_logits(lp, ci, tau=1.0):
    """log P(option i precedes option j) from the full time marginal, not just the centroid.

    q_i(t) = softmax over time of the option's frame log-prob; P(i<j) = sum_{s<t} q_i(s) q_j(t).
    Much less lossy than comparing centroids when a class has multi-modal support.
    """
    sub = lp[ci]                                   # 4 x T
    q = np.exp((sub - sub.max(1, keepdims=True)) / tau)
    q = q / (q.sum(1, keepdims=True) + 1e-12)
    C = np.cumsum(q, axis=1)
    P = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            # sum_t q_j(t) * P(i < t)
            P[i, j] = float((q[j][1:] * C[i][:-1]).sum())
    P = P / (P + P.T + 1e-12)
    return np.log(np.clip(P, 1e-6, 1 - 1e-6))


def perm_score_from_pair(LP):
    """Score every permutation by the sum of its pairwise log-probabilities."""
    out = np.zeros(24)
    for k, p in enumerate(PERMS):
        idx = [ord(c) - 65 for c in p]
        out[k] = sum(LP[idx[i], idx[j]] for i in range(4) for j in range(i + 1, 4))
    return out


def joint_decode(qs, scorevecs, prior=None, wprior=0.0):
    """qs: list of option lists (each len 4).  scorevecs: list of 24-vectors.
    Enumerate every total order of the union and pick the one maximising the summed score of
    the induced per-question permutations.  Returns list of permutation strings.
    """
    U = sorted(set().union(*[set(o) for o in qs]))
    best, bs = None, -np.inf
    pidx = {p: k for k, p in enumerate(PERMS)}
    for perm in itertools.permutations(U):
        pos = {x: i for i, x in enumerate(perm)}
        tot = 0.0
        induced = []
        for o, sv in zip(qs, scorevecs):
            lab = ''.join(sorted('ABCD', key=lambda L: pos[o[ord(L) - 65]]))
            induced.append(lab)
            tot += sv[pidx[lab]]
        if prior is not None and wprior:
            tot += wprior * sum(prior.get((perm[i], perm[j]), 0.0)
                                for i in range(len(perm)) for j in range(i + 1, len(perm)))
        if tot > bs:
            bs, best = tot, induced
    return best, bs, U
