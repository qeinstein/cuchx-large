"""Championship core: pseudo-test construction, test-visible session grouping,
and the manner (emotion) assignment solver.

Design rule enforced throughout: a function that runs at inference time may read only
  * the QA frame WITHOUT the answer column
  * clip ordering / index
  * champ/meta.csv and champ/feats.csv (derived purely from raw modality files)
Anything learned from labels must come from `fit_*` calls given TRAINING subjects only.
"""
import os, re, itertools, json
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# per-question decision margins, filled by the solvers for post-hoc auditing
MARGIN = {}

# ----------------------------------------------------------------------------- manner groups
FAST = {'Quickly', 'quickly', 'Rapidly', 'Hastily', 'Hasitly', 'Hurriedly', 'Swiftly',
        'Briskly', 'Urgently', 'Frantically', 'Impatiently', 'Eagerly', 'Forcefully'}
SLOW = {'Slowly', 'Leisurely', 'Unhurriedly', 'Calmly', 'Peacefully', 'Relaxedly', 'Lazily',
        'Gently', 'Softly', 'Quietly', 'Comfortably', 'Soothingly', 'Patiently', 'Lightly',
        'Contently', 'Absentmindedly'}
CARE = {'Carefully', 'Cautiously', 'Meticulously', 'Precisely', 'Thoroughly', 'Deliberately',
        'Methodically', 'Attentively', 'Intently', 'Diligently', 'Earnestly', 'Neatly',
        'Orderly', 'Seriously', 'Serioiusly'}
NERV = {'Nervously', 'Anxiously', 'Tensely', 'Tensly', 'Restlessly'}
GROUPS = ['SLOW', 'CARE', 'NEUT', 'NERV', 'FAST']


def mgroup(s):
    if s in FAST: return 'FAST'
    if s in SLOW: return 'SLOW'
    if s in CARE: return 'CARE'
    if s in NERV: return 'NERV'
    return 'NEUT'


# ----------------------------------------------------------------------------- data loading
def load_all():
    tr = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    fe = pd.read_csv(os.path.join(ROOT, 'champ', 'feats.csv'))
    meta = meta.merge(fe, on='unit_dir', how='left')
    x = tr.path.str.extract(r'HAU/(user\d+)/(\d+)-(\d+)-(\d+)')
    tr['user'] = x[0]; tr['aa'] = x[1]; tr['bb'] = x[2]; tr['cc'] = x[3]
    y = tr.path.str.extract(r'HARn/([^/]+)/(user\d+)/(\d+)-(\d+)-(\d+)')
    tr['user'] = tr.user.fillna(y[1]); tr['harn_action'] = y[0]
    tr['aa'] = tr.aa.fillna(y[2]); tr['bb'] = tr.bb.fillna(y[3]); tr['cc'] = tr.cc.fillna(y[4])
    return tr, te, meta


def opts(r):
    g = (lambda L: r[L]) if hasattr(r, 'keys') or isinstance(r, dict) else (lambda L: getattr(r, L))
    return [str(g(L)).strip() for L in 'ABCD']


def gt_letters(r):
    return [L for L in str(r['answer']) if L in 'ABCD']


# ----------------------------------------------------------------------------- pseudo-test
def make_pseudo(tr, meta, users, seed=0):
    """Turn the held-out users' training rows into a test-shaped problem.

    Mimics the real test construction: HARn units first (index 1..K), then HAU units
    (index K+1..) grouped into contiguous per-session blocks whose internal order is the
    true chronological recording order.  Block order is randomised.
    Returns (pseudo_qa without answers, answer key, clip table).
    """
    rng = np.random.default_rng(seed)
    sub = tr[tr.user.isin(users)].copy()
    tmap = dict(zip(meta.qa_path, meta.t0))

    hau = sub[sub.source == 'HAU']
    blocks = []
    for (u, a, b), g in hau.groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique(), key=lambda p: (tmap.get(p, np.inf), p))
        blocks.append(paths)
    rng.shuffle(blocks)

    harn = sorted(sub[sub.source == 'HARn'].path.unique())
    rng.shuffle(harn)

    clip_idx, i = {}, 1
    for p in harn:
        clip_idx[p] = i; i += 1
    n_harn = i - 1
    for blk in blocks:
        for p in blk:
            clip_idx[p] = i; i += 1

    q = sub.copy()
    q['idx'] = q.path.map(clip_idx)
    q['clip'] = 'LM_pseudo_%s'
    q['clip'] = ['LM_pseudo_%04d' % v for v in q.idx]
    q['true_path'] = q.path
    q = q.sort_values(['idx', 'category']).reset_index(drop=True)
    key = q[['qa_id', 'answer']].copy()
    vis = q.drop(columns=['answer', 'user', 'aa', 'bb', 'cc', 'harn_action'])
    return vis, key, dict(n_harn=n_harn, clip_idx=clip_idx, blocks=blocks,
                          true_block={p: bi for bi, blk in enumerate(blocks) for p in blk})


def real_test_view(te):
    v = te.drop(columns=[c for c in ['prediction'] if c in te.columns]).copy()
    v['idx'] = v.path.str.extract(r'LM_test_(\d+)').astype(int)
    v['clip'] = ['LM_test_%04d' % i for i in v.idx]
    # `true_path` is the key into champ/meta.csv, champ/feats.csv and the cached logits.
    # For the test split that key is the bare clip id, NOT the video path in test_qa.csv.
    v['true_path'] = v['clip']
    return v


# ----------------------------------------------------------------------------- grouping
ACTCAT = ['single', 'multi', 'combination', 'sequence']


def napp_score(rows):
    """Fraction of the candidate-action universe that is 'confirmed' by repeated appearance
    across a block's action questions.  Purely test-visible."""
    nq = 0
    na = Counter()
    for r in rows:
        nq += 1
        for L in 'ABCD':
            for p in str(r[L] if hasattr(r, 'keys') else getattr(r, L)).split(','):
                na[p.strip()] += 1
    if not na or nq == 0:
        return None
    conf = sum(1 for v in na.values() if v >= max(3, 0.5 * nq))
    return conf / len(na)


def fit_group_model(tr):
    """Block-size prior, emotion-option-intersection likelihood, and the napp-concentration
    likelihood under correct vs incorrect grouping."""
    e = tr[tr.category == 'emotion']
    sz, inter = Counter(), defaultdict(Counter)
    for (u, a, b), g in e.groupby(['user', 'aa', 'bb']):
        k = len(g)
        sz[k] += 1
        O = [set(opts(r)) for _, r in g.iterrows()]
        I = set.intersection(*O) if len(O) > 1 else O[0]
        inter[k][len(I)] += 1
    # null model for the intersection size: random k-subsets of clips
    eo_all = {r.path: set(opts(r)) for _, r in e.iterrows()}
    paths = sorted(eo_all)
    rng0 = np.random.default_rng(1)
    inter_null = defaultdict(Counter)
    for _ in range(4000):
        for kk in (2, 3, 4):
            pick = rng0.choice(len(paths), size=kk, replace=False)
            I = set.intersection(*[eo_all[paths[i]] for i in pick])
            inter_null[kk][len(I)] += 1
    # napp positives: true sessions.  negatives: clips drawn from different sessions.
    a_ = tr[(tr.source == 'HAU') & tr.category.isin(ACTCAT)]
    bysess = {k: list(g.itertuples()) for k, g in a_.groupby(['user', 'aa', 'bb'])}
    byclip = {k: list(g.itertuples()) for k, g in a_.groupby('path')}
    pos, neg = defaultdict(list), defaultdict(list)
    keys = sorted(bysess)
    for k in keys:
        rows = bysess[k]
        nclip = len({r.path for r in rows})
        v = napp_score(rows)
        if v is not None:
            pos[nclip].append(v)
    rng = np.random.default_rng(0)
    clips = sorted(byclip)
    for _ in range(3 * len(keys)):
        for kk in (2, 3, 4):
            pick = rng.choice(len(clips), size=kk, replace=False)
            rows = [r for i in pick for r in byclip[clips[i]]]
            v = napp_score(rows)
            if v is not None:
                neg[kk].append(v)
    stat = {}
    for kk in set(list(pos) + list(neg)):
        p = np.array(pos.get(kk, [])); n = np.array(neg.get(kk, []))
        if len(p) < 3 or len(n) < 3:
            continue
        stat[kk] = (float(p.mean()), float(p.std() + 1e-3), float(n.mean()), float(n.std() + 1e-3))
    return dict(size=sz, inter={k: dict(v) for k, v in inter.items()},
                inter_null={k: dict(v) for k, v in inter_null.items()}, napp=stat)


def _inter_ll(gm, k, ni):
    """Log-likelihood ratio of the emotion-option intersection size under a true session
    versus a random group of the same size."""
    c = gm['inter'].get(k, {})
    tot = sum(c.values()) + 6 * 0.5
    pos = np.log((c.get(ni, 0) + 0.5) / tot)
    cn = gm.get('inter_null', {}).get(k)
    if not cn:
        return pos
    totn = sum(cn.values()) + 6 * 0.5
    return pos - np.log((cn.get(ni, 0) + 0.5) / totn)


def _size_ll(gm, k):
    tot = sum(gm['size'].values()) + 4 * 0.5
    return np.log((gm['size'].get(k, 0) + 0.5) / tot)


def infer_blocks(vis, gm, kmin=2, kmax=4, w_napp=1.0):
    """DP segmentation of the HAU clip-index sequence into contiguous session blocks.
    Uses only: clip index order, emotion option sets, action-option repetition, size prior.
    """
    hau = vis[vis.source == 'HAU']
    idxs = sorted(hau.idx.unique())
    eo = {}
    for _, r in hau[hau.category == 'emotion'].iterrows():
        eo[r.idx] = set(opts(r))
    arows = defaultdict(list)
    for r in hau[hau.category.isin(ACTCAT)].itertuples():
        arows[r.idx].append(r)
    n = len(idxs)
    NEG = -1e6
    best = np.full(n + 1, -np.inf); best[0] = 0.0
    back = [None] * (n + 1)
    for j in range(1, n + 1):
        for k in range(kmin, kmax + 1):
            i = j - k
            if i < 0 or not np.isfinite(best[i]):
                continue
            blk = idxs[i:j]
            O = [eo[b] for b in blk if b in eo]
            if len(O) < len(blk):
                s = NEG
            else:
                I = set.intersection(*O) if len(O) > 1 else O[0]
                s = _inter_ll(gm, k, len(I)) + _size_ll(gm, k)
                st = gm.get('napp', {}).get(k)
                if st is not None and w_napp:
                    rows = [r for b in blk for r in arows.get(b, [])]
                    v = napp_score(rows)
                    if v is not None:
                        mp, sp, mn, sn = st
                        s += w_napp * (-0.5 * ((v - mp) / sp) ** 2 - np.log(sp)
                                       + 0.5 * ((v - mn) / sn) ** 2 + np.log(sn))
            if best[i] + s > best[j]:
                best[j] = best[i] + s; back[j] = (i, blk)
    out, j = [], n
    while j > 0:
        i, blk = back[j]
        out.append(blk); j = i
    out.reverse()
    return out


# ----------------------------------------------------------------------------- manner model
PHYS = ['sk_v_mean', 'sk_v_p90', 'sk_vmax_mean', 'sk_a_mean', 'sk_jerk_mean', 'sk_vwrist_mean',
        'sk_vankle_mean', 'sk_hip_path', 'sk_cad_hz', 'sk_cad_pow', 'sk_spec_cent',
        'sk_active_frac', 'sk_v_iqr', 'sk_T', 'imu_acc_std', 'imu_dacc_mean', 'imu_acc_energy',
        'imu_gyr_mean', 'imu_gyr_energy', 'rad_v_p90']
# feature-space velocity of the frozen DINOv2 depth embedding: an appearance-grounded
# motion magnitude, implicitly normalised by what the person is doing
DINO_PHYS = ['dino_v_mean', 'dino_v_std', 'dino_v_p90', 'dino_v_p50', 'dino_acc_mean',
             'dino_cos_mean', 'dino_cos_min', 'dino_v_iqr', 'dino_active', 'dino_path',
             'dino_straight', 'dino_cad_hz', 'dino_cad_pow', 'dino_spec_cent']
if os.environ.get('CHAMP_EMO_DINO', '1') == '1':
    PHYS = PHYS + DINO_PHYS


def block_features(blk, mfeat, k):
    """Per-clip features for a block: absolute + within-block relative.  Test-visible."""
    rows = []
    M = np.array([[mfeat.get(b, {}).get(c, np.nan) for c in PHYS] for b in blk], float)
    with np.errstate(all='ignore'):
        mu = np.nanmean(M, 0); sd = np.nanstd(M, 0) + 1e-9
        Z = (M - mu) / sd
        R = np.argsort(np.argsort(np.where(np.isnan(M), -np.inf, M), 0), 0) / max(1, k - 1)
    for i, b in enumerate(blk):
        d = {f'a_{c}': M[i, j] for j, c in enumerate(PHYS)}
        d.update({f'z_{c}': Z[i, j] for j, c in enumerate(PHYS)})
        d.update({f'r_{c}': R[i, j] for j, c in enumerate(PHYS)})
        d['pos'] = i; d['k'] = k; d['posfrac'] = i / max(1, k - 1)
        rows.append(d)
    return rows


def fit_manner(tr, meta):
    """Hierarchical P(position | manner) with group backoff, P(manner | group),
    and a physical-evidence classifier over manner groups."""
    mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)}
             for r in meta.itertuples()}
    e = tr[tr.category == 'emotion']
    X, y = [], []
    cnt_mp = Counter(); cnt_m = Counter()
    cnt_gp = Counter(); cnt_g = Counter()
    cnt_mr = Counter(); cnt_mr_tot = Counter()
    cnt_gr = Counter(); cnt_gr_tot = Counter()
    lab_by_grp = Counter()
    import itertools as _it
    for (u, a, b), g in e.groupby(['user', 'aa', 'bb']):
        g = g.sort_values('cc')
        rows_all = [r for _, r in g.iterrows()]
        variants = [rows_all]
        if len(rows_all) >= 3:
            # the test set has 21 two-trial sessions of 55; training has 7.  Emit every
            # order-preserving sub-block of size 2 so the k=2 regime is actually learned.
            variants += [list(c) for c in _it.combinations(rows_all, 2)]
        for rows_v in variants:
            blk = [r.path for r in rows_v]
            k = len(blk)
            bf = block_features(blk, mfeat, k)
            for i, r in enumerate(rows_v):
                lab = str(r[gt_letters(r)[0]]).strip()
                gr = mgroup(lab)
                X.append(bf[i]); y.append(gr)
                cnt_mp[(lab, i, k)] += 1; cnt_m[(lab, k)] += 1
            cnt_gp[(gr, i, k)] += 1; cnt_g[(gr, k)] += 1
            rb = 0 if k == 1 else int(round(2 * i / (k - 1)))   # 0=first, 1=middle, 2=last
            cnt_mr[(lab, rb)] += 1; cnt_mr_tot[lab] += 1
            cnt_gr[(gr, rb)] += 1; cnt_gr_tot[gr] += 1
            lab_by_grp[(gr, lab)] += 1
    Xd = pd.DataFrame(X)
    cols = list(Xd.columns)
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xd[cols].to_numpy(float), y)

    def p_pos_given_manner(lab, i, k):
        """Four-level hierarchy: exact (manner, position, size) shrunk toward the manner's
        rank-bucket profile, then its group's rank-bucket profile, then uniform.
        The rank-bucket levels are pooled across block sizes, so blocks of a size that is
        rare in training (two-trial sessions) still get a well-estimated prior."""
        gr = mgroup(lab)
        rb = 0 if k <= 1 else int(round(2 * i / (k - 1)))
        p_gr = (cnt_gr.get((gr, rb), 0) + 1.0) / (cnt_gr_tot.get(gr, 0) + 3.0)
        p_mr = (cnt_mr.get((lab, rb), 0) + 3.0 * p_gr) / (cnt_mr_tot.get(lab, 0) + 3.0)
        p_gp = (cnt_gp.get((gr, i, k), 0) + 2.0 * p_mr) / (cnt_g.get((gr, k), 0) + 2.0)
        base = 0.5 * p_mr + 0.5 * p_gp
        alpha = 4.0
        nm = cnt_m.get((lab, k), 0)
        return (cnt_mp.get((lab, i, k), 0) + alpha * base) / (nm + alpha)

    gtot = Counter()
    for (gr, lab), c in lab_by_grp.items():
        gtot[gr] += c
    pmg = {lab: (c + 0.5) / (gtot[gr] + 0.5 * 60) for (gr, lab), c in lab_by_grp.items()}
    ntot = sum(gtot.values())
    return dict(clf=clf, cols=cols, pmg=pmg, mfeat=mfeat, ppm=p_pos_given_manner,
                gprior={g: (gtot[g] + 1) / (ntot + 5) for g in GROUPS},
                classes=list(clf.classes_))


def solve_emotion(vis, blocks, mm, w_phys=1.0, w_pos=1.0, w_pair=1.0, pool_of=None):
    """Assign manners to the clips of each block.  Returns {qa_id: letter}.

    Unary evidence: slot prior x physical-group classifier x manner prior (as before).
    Pairwise evidence (mm['pm'], optional): for every clip pair, how their physical-feature
    DIFFERENCE favours one orientation of two candidate manners over the swap.  79% of the
    residual errors were pure within-session swaps, which a unary-only objective cannot see.
    The assignment is chosen by enumerating permutations rather than by Hungarian matching,
    since the pairwise term makes the objective non-additive over cells.
    """
    eq = vis[vis.category == 'emotion'].set_index('idx')
    mfeat = mm['mfeat']
    pathof = dict(zip(vis.idx, vis.true_path))
    out = {}
    diag = []
    for blk in blocks:
        blk = [b for b in blk if b in eq.index]
        if not blk:
            continue
        k = len(blk)
        rows = [eq.loc[b] for b in blk]
        O = [set(opts(r)) for r in rows]
        I = set.intersection(*O) if len(O) > 1 else set(O[0])
        cand = sorted(I) if len(I) >= k else sorted(set().union(*O))
        # 20 of the 21 two-clip test blocks have an intersection of size 3: the option sets
        # were generated from a full three-trial session and one trial was withheld.  Assign
        # the clips to three ordered protocol slots and marginalise over the missing one,
        # instead of treating the block as a genuine two-trial session.
        slots = len(cand) if (k < len(cand) <= 4) else k
        bf = block_features([pathof[b] for b in blk], mfeat, k)
        Xd = pd.DataFrame(bf).reindex(columns=mm['cols'])
        P = mm['clf'].predict_proba(Xd.to_numpy(float))
        cls = mm['classes']
        C = np.full((k, len(cand)), 60.0)
        for i in range(k):
            pg_phys = {g: P[i, cls.index(g)] if g in cls else 1e-6 for g in GROUPS}
            own = set(opts(rows[i]))
            for j, m in enumerate(cand):
                if m not in own:
                    continue
                g = mgroup(m)
                lp = (w_phys * (np.log(max(pg_phys[g], 1e-9))
                                - np.log(max(mm['gprior'][g], 1e-9)))
                      + w_pos * np.log(max(mm['ppm'](m, i, k), 1e-9))
                      + np.log(max(mm['pmg'].get(m, 1e-4), 1e-6)))
                C[i, j] = -lp
        own = [set(opts(r)) for r in rows]
        # pairwise log-odds over (clip pair) x (ordered candidate pair)
        plo = {}
        pm = mm.get('pm')
        if pm is not None and w_pair and k >= 2:
            import emopair as EP
            ctx = EP.pool_context((pool_of or {}).get(tuple(blk)))
            plo = EP.pair_logodds(pm, bf, k, cand, mm, ctx, own)

        def pair_score(lab):
            """lab[i] = manner assigned to clip i."""
            if not plo:
                return 0.0
            s = 0.0
            for i, j in itertools.combinations(range(k), 2):
                a, b = cand.index(lab[i]), cand.index(lab[j])
                v = plo.get((i, j, a, b))
                if v is not None:
                    s += v
            return s

        if slots > k:
            bf3 = block_features([pathof[b] for b in blk], mfeat, slots)
            X3 = pd.DataFrame(bf3).reindex(columns=mm['cols'])
            P3 = mm['clf'].predict_proba(X3.to_numpy(float))
            best = None
            for perm in itertools.permutations(range(slots)):
                # perm[s] = index into cand occupying protocol slot s
                for present in itertools.combinations(range(slots), k):
                    lab = [cand[perm[s]] for s in present]
                    if any(lab[i] not in own[i] for i in range(k)):
                        continue
                    tot = 0.0
                    for s in range(slots):
                        m = cand[perm[s]]
                        tot += (w_pos * np.log(max(mm['ppm'](m, s, slots), 1e-9))
                                + np.log(max(mm['pmg'].get(m, 1e-4), 1e-6)))
                    for i, s in enumerate(present):
                        m = cand[perm[s]]
                        g = mgroup(m)
                        pg = P3[i, cls.index(g)] if g in cls else 1e-6
                        tot += w_phys * (np.log(max(pg, 1e-9))
                                         - np.log(max(mm['gprior'][g], 1e-9)))
                    tot += w_pair * pair_score(lab)
                    if best is None or tot > best[0]:
                        best = (tot, lab)
            if best is not None:
                ri = list(range(k))
                ci = [cand.index(best[1][i]) for i in range(k)]
            else:
                ri, ci = linear_sum_assignment(C)
        elif plo and k <= 4 and len(cand) <= 5:
            # unary cost is -C; add the pairwise term and enumerate injective assignments
            best = None
            for sel in itertools.permutations(range(len(cand)), k):
                lab = [cand[s] for s in sel]
                if any(lab[i] not in own[i] for i in range(k)):
                    continue
                tot = -sum(C[i, sel[i]] for i in range(k)) + w_pair * pair_score(lab)
                if best is None or tot > best[0]:
                    best = (tot, list(sel))
            if best is not None:
                ri, ci = list(range(k)), best[1]
            else:
                ri, ci = linear_sum_assignment(C)
        else:
            ri, ci = linear_sum_assignment(C)
        # margin: for each clip, the objective loss incurred by forcing a different manner
        try:
            sel = list(ci)
            base = -sum(C[i, sel[i]] for i in range(k)) + w_pair * pair_score(
                [cand[s_] for s_ in sel])
            for i in range(k):
                alt = None
                for s2 in range(len(cand)):
                    if s2 == sel[i] or cand[s2] not in own[i]:
                        continue
                    trial = list(sel); trial[i] = s2
                    if len(set(trial)) != k:
                        continue
                    v = -sum(C[t, trial[t]] for t in range(k)) + w_pair * pair_score(
                        [cand[s_] for s_ in trial])
                    alt = v if alt is None else max(alt, v)
                MARGIN[rows[i].qa_id] = float(base - alt) if alt is not None else float('inf')
        except Exception:
            pass
        for i, j in zip(ri, ci):
            r = rows[i]
            m = cand[j]
            L = next((L for L in 'ABCD' if str(r[L]).strip() == m), None)
            if L is None:
                # assignment infeasible for this clip -> fall back to best own option
                sc = []
                pg_phys = {g: P[i, cls.index(g)] if g in cls else 1e-6 for g in GROUPS}
                for LL in 'ABCD':
                    mo = str(r[LL]).strip(); g = mgroup(mo)
                    sc.append((w_phys * (np.log(max(pg_phys[g], 1e-9))
                                         - np.log(max(mm['gprior'][g], 1e-9)))
                               + w_pos * np.log(max(mm['ppm'](mo, i, k), 1e-9))
                               + np.log(max(mm['pmg'].get(mo, 1e-4), 1e-6)), LL))
                L = max(sc)[1]
            out[r.qa_id] = L
            diag.append(dict(qa_id=r.qa_id, k=k, pos=i, nI=len(I), used_inter=len(I) >= k))
    return out, pd.DataFrame(diag)
