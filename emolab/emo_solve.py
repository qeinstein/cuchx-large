"""Standalone reproduction of the champion's emotion assignment, so that the ONLY thing
that changes between arms is the manner-group posterior.

score(assignment) = sum_i [ w_phys*(log P(group_i) - log prior(group))
                          + w_pos*log P(manner_i | slot i, k)
                          + log P(manner_i | group) ]
maximised over injective assignments of the candidate manner set to the block's clips.
"""
import os, sys, itertools, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'champ')
from core import load_all, opts, gt_letters, mgroup, GROUPS


def fit_priors(em_tr):
    """P(manner | slot, k) with group backoff and P(manner) — fitted on training users only."""
    cnt_mp = collections.Counter(); cnt_m = collections.Counter()
    cnt_gp = collections.Counter(); cnt_g = collections.Counter()
    pm = collections.Counter()
    for b, g in em_tr.groupby('blk'):
        g = g.sort_values('cc'); k = len(g)
        for i, (_, r) in enumerate(g.iterrows()):
            lab = r.lab; gr = mgroup(lab)
            cnt_mp[(lab, i, k)] += 1; cnt_m[(lab, k)] += 1
            cnt_gp[(gr, i, k)] += 1; cnt_g[(gr, k)] += 1
            pm[lab] += 1
    tot = sum(pm.values())
    gprior = collections.Counter()
    for lab, n in pm.items(): gprior[mgroup(lab)] += n

    def ppm(m, slot, k):
        gr = mgroup(m)
        a = cnt_mp.get((m, slot, k), 0); b = cnt_m.get((m, k), 0)
        ag = cnt_gp.get((gr, slot, k), 0); bg = cnt_g.get((gr, k), 0)
        pg = (ag + 1.0) / (bg + float(k)) if bg else 1.0 / max(k, 1)
        lam = b / (b + 4.0)
        return lam * ((a + 1e-3) / (b + 1e-3 * k)) + (1 - lam) * pg if b else pg
    pmg = {m: n / tot for m, n in pm.items()}
    gp = {g_: n / tot for g_, n in gprior.items()}
    return dict(ppm=ppm, pmg=pmg, gprior=gp)


def solve_block(rows, Pg, pri, w_phys=1.0, w_pos=1.0, cand=None):
    """rows: list of QA rows in trial order.  Pg: (k, 5) group posteriors."""
    k = len(rows)
    O = [set(opts(r)) for r in rows]
    I = set.intersection(*O) if k > 1 else set(O[0])
    if cand is None:
        cand = sorted(I) if len(I) >= k else sorted(set().union(*O))
    slots = len(cand) if (k < len(cand) <= 4) else k
    best = None
    for perm in itertools.permutations(range(len(cand)), slots):
        for present in itertools.combinations(range(slots), k):
            lab = [cand[perm[s]] for s in present]
            if any(lab[i] not in O[i] for i in range(k)): continue
            tot = 0.0
            for s in range(slots):
                m = cand[perm[s]]
                tot += w_pos * np.log(max(pri['ppm'](m, s, slots), 1e-9)) \
                     + np.log(max(pri['pmg'].get(m, 1e-4), 1e-6))
            for i, s in enumerate(present):
                g = mgroup(cand[perm[s]])
                tot += w_phys * (np.log(max(Pg[i, GROUPS.index(g)], 1e-9))
                                 - np.log(max(pri['gprior'].get(g, 1e-6), 1e-9)))
            if best is None or tot > best[0]: best = (tot, lab, list(present))
    return best
