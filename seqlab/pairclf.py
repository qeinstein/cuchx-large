"""Learned pairwise-order model P(action x precedes action y | clip).

Supervision: 2927 HARn segments nested in 790 parent HAU clips give the TRUE onset order of
every co-occurring action pair -- 5445 labelled ordered pairs, versus the 1848 pairs implicit
in the 308 sequence answers.  The champion never used them for ordering; it consumed the same
segments only as frame labels for the dense model, then read the order off a centroid argsort.

Features per (clip, x, y): time-marginal statistics of the two classes under the dense model,
their differences, the LOO count/Gaussian order prior, and clip length.  Fitted subject-disjoint.
"""
import os, sys, json, itertools, collections
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
sys.path.insert(0, os.path.join(ROOT, 'seqlab'))
import dense as D, decode as DC

VOCAB = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))
OPT2CLS = {VOCAB['HARN2HAU'][a]: D.A2I[a] for a in D.ACTIONS}


def clip_stats(lp):
    """Per-class time statistics from one clip's dense log-probs.  lp: NCLS x T."""
    P = np.exp(lp)
    T = P.shape[1]
    tot = P.sum(0, keepdims=True) + 1e-12
    Q = P / tot
    x = np.arange(T) / max(T - 1, 1)
    w = P[:D.BG] / (P[:D.BG].sum(1, keepdims=True) + 1e-12)
    cen = (w * x).sum(1)
    var = ((w * (x[None, :] - cen[:, None]) ** 2).sum(1)) ** 0.5
    q25 = np.array([x[np.searchsorted(np.cumsum(w[c]), 0.25).clip(0, T - 1)] for c in range(D.BG)])
    q50 = np.array([x[np.searchsorted(np.cumsum(w[c]), 0.50).clip(0, T - 1)] for c in range(D.BG)])
    q75 = np.array([x[np.searchsorted(np.cumsum(w[c]), 0.75).clip(0, T - 1)] for c in range(D.BG)])
    pk = x[P[:D.BG].argmax(1)]
    mx = Q[:D.BG].max(1)
    mn = Q[:D.BG].mean(1)
    # first/last frame where the class is the argmax
    am = P.argmax(0)
    first = np.full(D.BG, np.nan); last = np.full(D.BG, np.nan); frac = np.zeros(D.BG)
    for c in range(D.BG):
        idx = np.flatnonzero(am == c)
        if len(idx):
            first[c] = idx[0] / max(T - 1, 1); last[c] = idx[-1] / max(T - 1, 1)
            frac[c] = len(idx) / T
    return dict(cen=cen, sd=var, q25=q25, q50=q50, q75=q75, pk=pk, mx=mx, mn=mn,
                first=first, last=last, frac=frac, T=T, Pcum=np.cumsum(w, axis=1), w=w)


def cdf_pre(st, cx, cy):
    w, C = st['w'], st['Pcum']
    a = float((w[cy][1:] * C[cx][:-1]).sum())
    b = float((w[cx][1:] * C[cy][:-1]).sum())
    return a / (a + b + 1e-12)


def feat(st, ax, ay, prior_lo, onmu):
    cx, cy = OPT2CLS[ax], OPT2CLS[ay]
    f = {}
    for nm in ['cen', 'sd', 'q25', 'q50', 'q75', 'pk', 'mx', 'mn', 'first', 'last', 'frac']:
        vx, vy = st[nm][cx], st[nm][cy]
        f[f'{nm}_x'] = vx; f[f'{nm}_y'] = vy; f[f'{nm}_d'] = vx - vy
    f['cdf_pre'] = cdf_pre(st, cx, cy)
    f['T'] = st['T']
    f['prior'] = prior_lo.get((ax, ay), 0.0)
    f['onmu_x'] = onmu.get(ax, np.nan); f['onmu_y'] = onmu.get(ay, np.nan)
    f['onmu_d'] = f['onmu_x'] - f['onmu_y']
    f['overlap'] = float(min(st['last'][cx], st['last'][cy]) - max(st['first'][cx], st['first'][cy])) \
        if np.isfinite(st['first'][cx]) and np.isfinite(st['first'][cy]) else np.nan
    return f
