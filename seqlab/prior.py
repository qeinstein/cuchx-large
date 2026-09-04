"""Order priors over the 40-action vocabulary, all leave-one-user-out capable.

Three sources of temporal-order supervision, none of which the champion used:
  1. HARn segment onsets nested in their parent HAU clip  (2927 segments / 790 clips)
  2. the normalized-onset distribution per action, as a Gaussian -> a *smoothed* pairwise
     prior with full vocabulary coverage (the count prior covers 92.5% of option pairs)
  3. the HAU sequence answers themselves (308 questions x 6 pairs)
"""
import os, sys, json, itertools, collections
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sources():
    S = pd.read_csv(os.path.join(ROOT, 'seqlab', 'onsets.csv'))
    seg = collections.defaultdict(collections.Counter)          # (x,y) -> {user: n}
    for (u, t), g in S.groupby(['user', 'trial']):
        v = sorted(zip(g.on, g.act))
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[i][1] != v[j][1]:
                    seg[(v[i][1], v[j][1])][u] += 1
    sys.path.insert(0, os.path.join(ROOT, 'champ'))
    from core import load_all
    tr, _, _ = load_all()
    sq = tr[(tr.source == 'HAU') & (tr.category == 'sequence')]
    qa = collections.defaultdict(collections.Counter)
    for _, r in sq.iterrows():
        o = {L: r[L] for L in 'ABCD'}
        ordr = [o[L] for L in str(r['answer'])]
        for i in range(4):
            for j in range(i + 1, 4):
                qa[(ordr[i], ordr[j])][r.user] += 1
    return S, seg, qa


class OrderPrior:
    def __init__(self, S, seg, qa):
        self.S, self.seg, self.qa = S, seg, qa
        self.keys = set(tuple(sorted(k)) for k in list(seg) + list(qa))
        self._c = {}

    def build(self, exclude_user, alpha=1.0, w_qa=1.0, gauss=True, gw=1.0, kappa=6.0):
        key = (exclude_user, alpha, w_qa, gauss, gw, kappa)
        if key in self._c:
            return self._c[key]
        # Gaussian onset model
        gp = {}
        if gauss:
            d = self.S[self.S.user != exclude_user]
            for a, g in d.groupby('act'):
                gp[a] = (float(g.on.mean()), float(g.on.std(ddof=1) if len(g) > 1 else 0.3), len(g))
        out = {}
        for (x, y) in self.keys:
            nxy = sum(v for u, v in self.seg.get((x, y), {}).items() if u != exclude_user) \
                + w_qa * sum(v for u, v in self.qa.get((x, y), {}).items() if u != exclude_user)
            nyx = sum(v for u, v in self.seg.get((y, x), {}).items() if u != exclude_user) \
                + w_qa * sum(v for u, v in self.qa.get((y, x), {}).items() if u != exclude_user)
            n = nxy + nyx
            p_cnt = (nxy + alpha) / (n + 2 * alpha) if n > 0 else None
            p_g = None
            if x in gp and y in gp:
                mx, sx, _ = gp[x]; my, sy, _ = gp[y]
                p_g = float(norm.cdf((my - mx) / max(np.hypot(sx, sy), 1e-6)))
            # shrink the count estimate towards the Gaussian estimate by support
            if p_cnt is None and p_g is None:
                continue
            if p_cnt is None:
                p = p_g
            elif p_g is None:
                p = p_cnt
            else:
                w = n / (n + kappa)
                p = w * p_cnt + (1 - w) * (gw * p_g + (1 - gw) * 0.5)
            p = float(np.clip(p, 1e-3, 1 - 1e-3))
            lo = np.log(p / (1 - p))
            out[(x, y)] = lo; out[(y, x)] = -lo
        self._c[key] = out
        return out

    def gauss_only(self, exclude_user):
        d = self.S[self.S.user != exclude_user]
        gp = {a: (float(g.on.mean()), float(g.on.std(ddof=1) if len(g) > 1 else .3))
              for a, g in d.groupby('act')}
        out = {}
        acts = list(gp)
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                x, y = acts[i], acts[j]
                mx, sx = gp[x]; my, sy = gp[y]
                p = float(np.clip(norm.cdf((my - mx) / max(np.hypot(sx, sy), 1e-6)), 1e-3, 1 - 1e-3))
                lo = np.log(p / (1 - p)); out[(x, y)] = lo; out[(y, x)] = -lo
        return out
