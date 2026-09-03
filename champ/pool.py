"""Session action-pool solver.

Latent per inferred session block: a set P of atomic actions.
Hard constraints (each measured at ~100% precision on training data):
    sequence     : all four options in P
    single       : exactly one option in P
    combination  : exactly one option-pair fully contained in P
    multi        : at least one option in P
Soft evidence: log-odds that each candidate action belongs to the pool, from the dense
temporal model plus test-visible question structure.
Answers are then read off as `options ∩ P`.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D, decode as DC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
OPT2CLS = {V[a]: D.A2I[a] for a in D.ACTIONS}
ACTCATS = ['single', 'multi', 'combination', 'sequence']

# action co-occurrence prior, fitted from training session pools by fit_cooc()
_COOC = {'p': {}, 'joint': {}}


def fit_cooc(tr):
    """P(b in pool) and P(b in pool | a in pool) over training sessions.

    Pair blocks carry only ~6.5 questions, so the direct option-repetition evidence is thin;
    knowing that Mopping travels with Sweeping is what fills the gap."""
    pools = defaultdict(set)
    for _, r in tr[(tr.source == 'HAU') & tr.category.isin(ACTCATS)].iterrows():
        for L in str(r['answer']):
            if L in 'ABCD':
                for x in str(r[L]).split(','):
                    pools[(r.user, r.aa, r.bb)].add(x.strip())
    n = len(pools)
    marg = Counter()
    joint = defaultdict(Counter)
    for P in pools.values():
        for a in P:
            marg[a] += 1
            for b in P:
                if a != b:
                    joint[a][b] += 1
    _COOC['p'] = {a: (c + 0.5) / (n + 1.0) for a, c in marg.items()}
    _COOC['joint'] = {a: {b: (c + 0.5) / (marg[a] + 1.0) for b, c in d.items()}
                      for a, d in joint.items()}
    _COOC['default'] = 0.5 / (n + 1.0)


def cooc_feats(a, confirmed):
    p = _COOC.get('p', {})
    if not p or not confirmed:
        return dict(co_max=np.nan, co_mean=np.nan, co_n=len(confirmed))
    pa = p.get(a, _COOC['default'])
    v = []
    for c in confirmed:
        if c == a:
            continue
        pj = _COOC['joint'].get(c, {}).get(a, _COOC['default'])
        v.append(np.log(pj / max(pa, 1e-6)))
    if not v:
        return dict(co_max=np.nan, co_mean=np.nan, co_n=len(confirmed))
    return dict(co_max=float(np.max(v)), co_mean=float(np.mean(v)), co_n=len(confirmed))


def qopts(r):
    """Decompose an option into its atomic actions (combination options are pairs)."""
    return [[p.strip() for p in str(getattr(r, L)).split(',')] for L in 'ABCD']


_QIDX = {}


def block_questions(vis, blk):
    """Questions of the given clips.  Indexed once per frame object; the fitting path calls
    this thousands of times and the DataFrame scan dominated the runtime."""
    key = id(vis)
    idx = _QIDX.get(key)
    if idx is None:
        idx = defaultdict(list)
        for r in vis[vis.category.isin(ACTCATS)].itertuples():
            idx[r.idx].append(r)
        if len(_QIDX) > 24:
            _QIDX.clear()
        _QIDX[key] = idx
    return [r for b in blk for r in idx.get(b, [])]


def candidate_evidence(vis, blk, split, statcache):
    """Per-candidate-action feature rows.  Test-visible only."""
    qs = block_questions(vis, blk)
    uni = sorted({a for r in qs for op in qopts(r) for a in op})
    pathof = dict(zip(vis.idx, vis.true_path))
    stats = {}
    for b in blk:
        p = pathof[b]
        if p in statcache:
            stats[b] = statcache[p]
    napp = Counter(); ncat = defaultdict(Counter); nq = 0
    forced = set()
    for r in qs:
        nq += 1
        for op in qopts(r):
            for a in op:
                napp[a] += 1; ncat[a][r.category] += 1
        if r.category == 'sequence':
            forced |= {a for op in qopts(r) for a in op}
    confirmed = set(forced) | {a for a in uni if napp[a] >= max(3, 0.6 * nq)}
    rows = {}
    for a in uni:
        c = OPT2CLS.get(a)
        f = dict(napp=napp[a], napp_frac=napp[a] / max(1, nq),
                 n_single=ncat[a]['single'], n_multi=ncat[a]['multi'],
                 n_comb=ncat[a]['combination'], n_seq=ncat[a]['sequence'],
                 forced=int(a in forced), k=len(blk), uni=len(uni), nq=nq,
                 **cooc_feats(a, confirmed))
        if c is not None and stats:
            for nm in ('mx', 'mean', 'topk', 'frac', 'lmx', 'lmean'):
                v = [stats[b][nm][c] for b in stats]
                f[f'{nm}_max'] = float(np.max(v)); f[f'{nm}_mean'] = float(np.mean(v))
            f['cent_mean'] = float(np.mean([stats[b]['centroid'][c] for b in stats]))
        rows[a] = f
    return uni, rows, qs, forced


def _components(qs, uni, forced):
    """Split a block's constraints into independent connected components over the free atoms.
    Questions whose free atoms are disjoint can be solved separately, which turns one
    2**|free| enumeration into a sum of much smaller ones."""
    free = [a for a in uni if a not in forced]
    pos = {a: i for i, a in enumerate(free)}
    items = []
    for r in qs:
        ops = qopts(r)
        fa = {a for op in ops for a in op if a in pos}
        items.append((r.category, ops, fa))
    parent = list(range(len(free)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for _, _, fa in items:
        fl = sorted(pos[a] for a in fa)
        for j in fl[1:]:
            union(fl[0], j)
    groups = defaultdict(list)
    for a, i in pos.items():
        groups[find(i)].append(a)
    comps = []
    for root, atoms in groups.items():
        aset = set(atoms)
        cq = [(c, ops) for c, ops, fa in items if fa & aset]
        comps.append((sorted(atoms), cq))
    # constraints with no free atoms at all must already be satisfied by `forced`
    fixed_q = [(c, ops) for c, ops, fa in items if not fa]
    return free, comps, fixed_q


def _enum_component(atoms, cq, forced, lo, cap=1 << 22):
    """Return the best assignment (set of atoms to include) for one component, or None if
    unsatisfiable.  Vectorised over all 2**len(atoms) candidate assignments."""
    n = len(atoms)
    if n > 22:
        return None, False
    X = np.arange(1 << n, dtype=np.int64)
    ok = np.ones(1 << n, bool)
    idx = {a: i for i, a in enumerate(atoms)}
    for cat, ops in cq:
        tcount = np.zeros(1 << n, np.int8)
        for op in ops:
            if any(a not in forced and a not in idx for a in op):
                continue
            m = 0
            bad = False
            for a in op:
                if a in idx:
                    m |= 1 << idx[a]
                elif a not in forced:
                    bad = True
            if bad:
                continue
            tcount += ((X & m) == m)
        if cat in ('single', 'combination'):
            ok &= (tcount == 1)
        elif cat == 'multi':
            ok &= (tcount >= 1)
        if not ok.any():
            return None, False
    w = np.array([lo.get(a, 0.0) for a in atoms])
    bits = ((X[:, None] >> np.arange(n)) & 1).astype(np.int8)
    sc = bits @ w - (1 - bits) @ w
    sc[~ok] = -np.inf
    best = int(np.argmax(sc))
    return {atoms[i] for i in range(n) if best >> i & 1}, True


def enumerate_pools(qs, uni, forced, cap=1 << 21):
    """Kept for the diagnostics path: number of satisfying pools is no longer materialised."""
    free = [a for a in uni if a not in forced]
    return None, free


def solve_block(vis, blk, statcache, scorer, split):
    uni, rows, qs, forced = candidate_evidence(vis, blk, split, statcache)
    if not qs:
        return {}, None
    lo = scorer(rows, uni)                     # action -> log-odds of pool membership
    free, comps, _ = _components(qs, uni, forced)
    best = set(forced)
    allsat = True
    for atoms, cq in comps:
        sel, sat = _enum_component(atoms, cq, forced, lo)
        if not sat or sel is None:
            allsat = False
            sel = {a for a in atoms if lo.get(a, -9) > 0}
        best |= sel
    out = {}
    for r in qs:
        ops = qopts(r)
        hit = [i for i, op in enumerate(ops) if all(a in best for a in op)]
        if r.category in ('single', 'combination'):
            if len(hit) != 1:
                hit = [max(range(4), key=lambda i: sum(lo.get(a, 0.0) for a in ops[i]))]
            out[r.qa_id] = 'ABCD'[hit[0]]
        elif r.category == 'multi':
            if not hit:
                hit = [max(range(4), key=lambda i: sum(lo.get(a, 0.0) for a in ops[i]))]
            out[r.qa_id] = ''.join('ABCD'[i] for i in sorted(hit))
        # sequence order is produced by the temporal decoder, not here
    return out, dict(pool=sorted(best), nsol=int(allsat), uni=len(uni), free=len(free),
                     ncomp=len(comps), maxcomp=max((len(a) for a, _ in comps), default=0))


# ------------------------------------------------------------------ pool-membership model
def _clip_truth(vis, blk):
    """Union of the answers of the questions belonging to the given clips."""
    t = set()
    for r in vis[vis.idx.isin(blk)].itertuples():
        if r.category not in ACTCATS:
            continue
        for L in str(r.answer):
            if L in 'ABCD':
                for x in str(getattr(r, L)).split(','):
                    t.add(x.strip())
    return t


def fit_pool_model(tr, meta, blocks_by_user, statcache, augment_subblocks=True):
    """Binary classifier: is candidate action `a` in this session's pool?
    Trained on training subjects only.

    The real test set contains 21 two-trial sessions out of 55 while training has only 7,
    so every training session is also emitted as its sub-blocks of size 2 (and 1).  Without
    this the model is effectively untrained in the regime that covers a quarter of the test
    questions.
    """
    X, y = [], []
    for (u, a_, b_), info in blocks_by_user.items():
        vis, blk, truth = info
        variants = [(blk, truth)]
        if augment_subblocks and len(blk) >= 3:
            for k in (2, len(blk) - 1) if len(blk) > 3 else (2,):
                for sub in itertools.combinations(blk, k):
                    variants.append((list(sub), _clip_truth(vis, list(sub))))
        for bl, tt in variants:
            uni, rows, qs, forced = candidate_evidence(vis, bl, 'oof', statcache)
            for a in uni:
                X.append(rows[a]); y.append(int(a in tt))
    Xd = _filter(pd.DataFrame(X))
    cols = list(Xd.columns)
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=5,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xd[cols].to_numpy(float), y)
    return clf, cols


STRUCT_COLS = ['napp', 'napp_frac', 'n_single', 'n_multi', 'n_comb', 'n_seq', 'forced',
               'k', 'uni', 'nq', 'co_max', 'co_mean', 'co_n']


def feature_mode():
    return os.environ.get('CHAMP_POOL_FEATS', 'all')


def _filter(Xd):
    m = feature_mode()
    if m == 'struct':
        keep = [c for c in Xd.columns if c in STRUCT_COLS]
    elif m == 'dense':
        keep = [c for c in Xd.columns if c not in STRUCT_COLS or c in ('k', 'uni')]
    else:
        keep = list(Xd.columns)
    return Xd[[c for c in Xd.columns if c in keep]]


def make_scorer(clf, cols, clip=6.0):
    def scorer(rows, uni):
        Xd = pd.DataFrame([rows[a] for a in uni]).reindex(columns=cols)
        p = clf.predict_proba(Xd.to_numpy(float))[:, 1]
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return {a: float(np.clip(np.log(p[i] / (1 - p[i])), -clip, clip))
                for i, a in enumerate(uni)}
    return scorer
