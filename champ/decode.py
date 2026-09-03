"""Decoders on top of the cached dense frame log-probs."""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LG = None


def logits(split, qa_path):
    global _LG
    if _LG is None:
        _LG = np.load(os.path.join(ROOT, 'champ', os.environ.get('CHAMP_LOGITS', 'dense_logits.npz')))
    k = f'{split}|{qa_path}'
    return _LG[k] if k in _LG else None


def seq_order_dp(lp, cls_idx):
    """Best monotone ordering of the given classes.
    lp: NCLS x T log-probs; cls_idx: list of 4 class indices.
    Returns (best_perm_as_tuple_of_positions_in_cls_idx, score, all_scores).
    States: (j, in_action) with background allowed before/after each action.
    """
    T = lp.shape[1]
    bg = lp[D.BG]
    scores = {}
    for perm in itertools.permutations(range(len(cls_idx))):
        A = [lp[cls_idx[p]] for p in perm]
        n = len(perm)
        NEG = -1e18
        # dp[j][0] = background before action j, dp[j][1] = inside action j
        dp = np.full((n + 1, 2), NEG)
        dp[0, 0] = 0.0
        for t in range(T):
            nd = np.full((n + 1, 2), NEG)
            for j in range(n + 1):
                e_bg = bg[t]
                if dp[j, 0] > NEG / 2:
                    nd[j, 0] = max(nd[j, 0], dp[j, 0] + e_bg)
                    if j < n:
                        nd[j, 1] = max(nd[j, 1], dp[j, 0] + A[j][t])
                if dp[j, 1] > NEG / 2 and j < n:
                    nd[j, 1] = max(nd[j, 1], dp[j, 1] + A[j][t])
                    nd[j + 1, 0] = max(nd[j + 1, 0], dp[j, 1] + e_bg)
                    if j + 1 < n:
                        nd[j + 1, 1] = max(nd[j + 1, 1], dp[j, 1] + A[j + 1][t])
            dp = nd
        scores[perm] = max(dp[n, 0], dp[n - 1, 1])
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def presence_stats(lp, smooth=7):
    """Per-class pooled statistics used as presence evidence."""
    P = np.exp(lp)
    T = P.shape[1]
    if T > smooth:
        k = np.ones(smooth) / smooth
        Ps = np.stack([np.convolve(P[c], k, mode='same') for c in range(P.shape[0])])
    else:
        Ps = P
    srt = np.sort(Ps, axis=1)[:, ::-1]
    topk = srt[:, :max(1, T // 10)].mean(1)
    return dict(mx=Ps.max(1), mean=Ps.mean(1), topk=topk,
                frac=(Ps > 0.3).mean(1), lmx=lp.max(1), lmean=lp.mean(1),
                centroid=(Ps * np.arange(T)).sum(1) / (Ps.sum(1) + 1e-9) / max(1, T - 1),
                peak=Ps.argmax(1) / max(1, T - 1), T=np.full(P.shape[0], T))
