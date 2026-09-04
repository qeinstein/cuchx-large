"""Split session blocks that cannot be sessions.

`infer_blocks` is a DP with kmin=2, so it is structurally unable to emit a one-clip block.
The real test set contains orphan clips -- sessions for which only one trial was released --
and every orphan is therefore forcibly glued onto a genuine neighbouring session, corrupting
that session's action pool and manner candidate set for all of its questions.

Two independent, purely test-visible signals identify the damage:

  1. Emotion option-set intersection.  On training data
         P(|I| >= size | one true session)      = 0.990   (795/802 pairs, 262/265 triples)
         P(|I| >= 2    | different sessions)    = 0.0073
     so |I| < size is a ~136:1 likelihood ratio against the block being one session.

  2. The recording clock in champ/meta.csv.  Consecutive trials of a session start
     61-319 s apart (5th-95th pct, median 98 s); a boundary between sessions is drawn from a
     much heavier-tailed distribution (median 214 s, and the test set contains gaps of
     79881 s and 1286771 s INSIDE inferred blocks).

Repair considers every contiguous refinement of a non-conforming block -- sessions are
contiguous in clip index, which holds for all 51 conforming test blocks -- and scores it with
the same intersection/size likelihoods the DP uses, plus a log-normal gap term, allowing
one-clip parts at a tunable penalty.
"""
import os, sys, itertools, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import _inter_ll, _size_ll, opts, napp_score, ACTCAT

SINGLETON_PEN = float(os.environ.get('CHAMP_SINGLE_PEN', 3.0))
W_GAP = float(os.environ.get('CHAMP_W_GAP', 1.0))
# Conformance-first mode.  The likelihood scoring above inherits the DP's size prior, which is
# fitted on training data containing only 6 two-clip and 0 one-clip sessions, so it punishes
# small parts far harder than the real test distribution warrants (21+ pairs of 55 blocks).
# In this mode a refinement in which EVERY multi-clip part conforms (|I| >= size, an event with
# P = 0.990 under one true session and 0.0073 under different sessions) always beats one in
# which some part does not; the likelihood is used only to break ties, preferring the fewest
# one-clip parts.
CONFORM_FIRST = os.environ.get('CHAMP_CONFORM_FIRST', '0') == '1'


def fit_gap_model(tr, meta):
    """Log-normal within-session and between-session t0-gap densities, from training data."""
    h = meta[meta.kind == 'train_hau'].dropna(subset=['t0']).copy()
    tri = h.trial.str.split('-', expand=True)
    h['a'], h['b'], h['c'] = tri[0], tri[1], tri[2].astype(int)
    within, between = [], []
    for (u, a, b), g in h.groupby(['user', 'a', 'b']):
        g = g.sort_values('c'); t = g.t0.tolist(); cs = g.c.tolist()
        for i in range(len(t) - 1):
            if cs[i + 1] - cs[i] == 1 and 0 < t[i + 1] - t[i] < 1e5:
                within.append(t[i + 1] - t[i])
    for u, g in h.sort_values('t0').groupby('user'):
        prev = None
        for r in g.itertuples():
            if prev is not None and (prev.a, prev.b) != (r.a, r.b):
                d = r.t0 - prev.t0
                if 0 < d < 1e7:
                    between.append(d)
            prev = r
    def ln(v):
        v = np.log(np.array(v, float))
        return float(v.mean()), float(v.std() + 1e-6)
    return dict(w=ln(within), b=ln(between), nw=len(within), nb=len(between))


def _lnpdf(x, mu, sd):
    if not np.isfinite(x) or x <= 0:
        return 0.0
    z = (math.log(x) - mu) / sd
    return -0.5 * z * z - math.log(sd) - math.log(x)


def _gap_ll(gaps, boundaries, gmod):
    """gaps[i] is the gap between element i and i+1; boundaries[i] True if a split there."""
    s = 0.0
    for g, bnd in zip(gaps, boundaries):
        if not np.isfinite(g):
            continue
        s += _lnpdf(abs(g), *(gmod['b'] if bnd else gmod['w']))
    return s


def _parts(n):
    """All contiguous partitions of range(n)."""
    for mask in range(1 << max(n - 1, 0)):
        cuts = [i for i in range(n - 1) if mask >> i & 1]
        out, prev = [], 0
        for c in cuts:
            out.append(list(range(prev, c + 1))); prev = c + 1
        out.append(list(range(prev, n)))
        yield out, [(i in cuts) for i in range(n - 1)]


def repair(blocks, vis, gm, gmod, t0_of, verbose=False):
    hau = vis[vis.source == 'HAU']
    eo = {r.idx: set(opts(r)) for _, r in hau[hau.category == 'emotion'].iterrows()}
    arows = {}
    for r in hau[hau.category.isin(ACTCAT)].itertuples():
        arows.setdefault(r.idx, []).append(r)

    def part_ll(idxs):
        k = len(idxs)
        O = [eo[i] for i in idxs if i in eo]
        if len(O) < k:
            return -1e6
        I = set.intersection(*O) if k > 1 else O[0]
        s = _inter_ll(gm, k, len(I)) + (_size_ll(gm, k) if k >= 2 else -SINGLETON_PEN)
        st = gm.get('napp', {}).get(k)
        if st is not None and k >= 2:
            rows = [r for i in idxs for r in arows.get(i, [])]
            v = napp_score(rows)
            if v is not None:
                mp, sp, mn, sn = st
                s += (-0.5 * ((v - mp) / sp) ** 2 - math.log(sp)
                      + 0.5 * ((v - mn) / sn) ** 2 + math.log(sn))
        return s

    out, log = [], []
    for blk in blocks:
        blk = [int(x) for x in blk]
        n = len(blk)
        O = [eo[i] for i in blk if i in eo]
        I = set.intersection(*O) if len(O) > 1 else (O[0] if O else set())
        conform = len(O) == n and len(I) >= n
        if conform or n < 2:
            out.append(blk); continue
        gaps = [t0_of.get(blk[i + 1], np.nan) - t0_of.get(blk[i], np.nan) for i in range(n - 1)]

        def conforms(idxs):
            if len(idxs) < 2:
                return True
            O = [eo[i] for i in idxs if i in eo]
            if len(O) < len(idxs):
                return False
            return len(set.intersection(*O)) >= len(idxs)

        best = None
        for parts, bnds in _parts(n):
            s = sum(part_ll([blk[i] for i in p]) for p in parts) + W_GAP * _gap_ll(gaps, bnds, gmod)
            if CONFORM_FIRST:
                allc = all(conforms([blk[i] for i in p]) for p in parts)
                nsing = sum(1 for p in parts if len(p) == 1)
                key = (int(allc), -nsing, s)
            else:
                key = (s,)
            if best is None or key > best[0]:
                best = (key, parts)
        keep = [[blk[i] for i in p] for p in best[1]]
        out.extend(keep)
        log.append((blk, keep, len(I)))
        if verbose:
            print(f'  repair {blk} (|I|={len(I)}) -> {keep}')
    return out, log
