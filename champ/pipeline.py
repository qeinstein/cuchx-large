"""End-to-end championship pipeline.

One entry point solves every question category from a test-visible view, and the same code
path is used for the pseudo-test folds and for the real Kaggle test set.
"""
import os, sys, json, itertools, pickle
import numpy as np, pandas as pd
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_all, make_pseudo, real_test_view, fit_group_model, infer_blocks,
                  fit_manner, solve_emotion, opts)
import dense as D, decode as DC, pool as PL, harn as H
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
OPT2CLS = {V[a]: D.A2I[a] for a in D.ACTIONS}
LAM_POOL = float(os.environ.get('CHAMP_LAM_POOL', 10.0))
W_HCLF = float(os.environ.get('CHAMP_W_HCLF', 1.0))
# the dense-model-at-interval term measurably hurts once the sequence classifier and the
# recovered pool are both in play (HARn single 0.893 -> 0.821 when it is switched on)
W_DENSE_INTERVAL = float(os.environ.get('CHAMP_W_DENSE_IV', 0.0))
# frozen DINOv2 depth features as a second, partially independent action-identity signal.
# Standalone it is weaker than skeleton+IMU (35.4% vs 52.7% top-1) but it corrects 222 of
# the 1370 skeleton errors, and a late-fusion weight of 0.3 was the held-out optimum:
#   action top-1 0.5266 -> 0.5501 ; HARn single 409/429 -> 417/429 ; object 119 -> 121
W_HDINO = float(os.environ.get('CHAMP_W_HDINO', 0.3))
# learned prior over which protocol slots a two-clip block contains; 0.0 = champion behaviour
W_SLOT = float(os.environ.get('CHAMP_W_SLOT', 0.0))
# split inferred blocks that cannot be one session (see champ/repair.py); 0 = champion
W_REPAIR = float(os.environ.get('CHAMP_REPAIR', 0.0))
ACT = ['single', 'multi', 'combination', 'sequence']

# ------------------------------------------------------------------ shared caches
_C = {}


def caches(meta):
    if _C:
        return _C
    _C['meta_idx'] = meta.set_index('qa_path')
    _C['mi'] = set(_C['meta_idx'].index)
    _C['skel'] = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    _C['lg'] = np.load(os.path.join(ROOT, 'champ', os.environ.get(
        'CHAMP_LOGITS', 'dense_logits.npz')))
    # HARn -> parent nesting, from timestamps and global frame indices only
    for kind, hkind in (('train_harn', 'train_hau'), ('test', 'test')):
        pass
    _C['nest'] = {}
    for scope, hk, nk in (('train', 'train_hau', 'train_harn'), ('test', 'test', 'test')):
        hl = [(r.qa_path, r.t0, r.t1, r.f0, r.f1, r.nf)
              for r in meta[meta.kind == hk].itertuples() if np.isfinite(r.t0)]
        if scope == 'test':
            hl = [x for x in hl if int(x[0].split('_')[-1]) >= 65]
            cand = [r for r in meta[meta.kind == nk].itertuples()
                    if np.isfinite(r.t0) and int(r.qa_path.split('_')[-1]) < 65]
        else:
            cand = [r for r in meta[meta.kind == nk].itertuples() if np.isfinite(r.t0)]
        for r in cand:
            c = [h for h in hl if h[1] <= r.t0 + 0.5 and h[2] >= r.t1 - 0.5
                 and h[3] <= r.f0 and h[4] >= r.f1 and h[0] != r.qa_path]
            if len(c) == 1:
                _C['nest'][r.qa_path] = c[0][0]
    hp = os.path.join(ROOT, 'champ', 'harn_clf.npz')
    if os.path.exists(hp):
        z = np.load(hp, allow_pickle=True)
        _C['hcls'] = [str(x) for x in z['classes']]
        _C['hlog'] = {k: z[k] for k in z.files if '|' in k}
    else:
        _C['hcls'], _C['hlog'] = None, {}
    dp = os.path.join(ROOT, 'champ', 'harn_dino.npz')
    if os.path.exists(dp):
        z = np.load(dp, allow_pickle=True)
        _C['dcls'] = [str(x) for x in z['classes']]
        _C['dlog'] = {k: z[k] for k in z.files if '|' in k}
    else:
        _C['dcls'], _C['dlog'] = None, {}
    _C['stat'] = {}
    for k in _C['lg'].files:
        sp, p = k.split('|', 1)
        _C['stat'][(sp, p)] = None            # lazily filled
    return _C


def stat_of(split, p):
    key = (split, p)
    c = _C['stat']
    if key not in c:
        return None
    if c[key] is None:
        c[key] = DC.presence_stats(_C['lg'][f'{split}|{p}'])
    return c[key]


def statcache_view(split):
    class V2(dict):
        def __contains__(self, p):
            return (split, p) in _C['stat']
        def __getitem__(self, p):
            return stat_of(split, p)
        def __iter__(self):
            return iter(p for (s, p) in _C['stat'] if s == split)
        def keys(self):
            return list(iter(self))
    return V2()


# ------------------------------------------------------------------ fitting
def true_pool(tr):
    pool = defaultdict(set)
    for _, r in tr[(tr.source == 'HAU') & tr.category.isin(ACT)].iterrows():
        for L in str(r['answer']):
            if L in 'ABCD':
                for p in str(r[L]).split(','):
                    pool[(r.user, r.aa, r.bb)].add(p.strip())
    return pool


def training_blocks(tr, users, tpool):
    out = {}
    h = tr[(tr.source == 'HAU') & tr.user.isin(users)]
    for (u, a, b), g in h.groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique())
        idx = {p: i for i, p in enumerate(paths)}
        v = g.copy(); v['idx'] = v.path.map(idx); v['true_path'] = v.path
        out[(u, a, b)] = (v, list(range(len(paths))), tpool[(u, a, b)])
    return out


def fit_all(tr, meta, hold_users, split_train='oof'):
    tpool = true_pool(tr)
    trn_users = [u for u in sorted(tr.user.dropna().unique()) if u not in hold_users]
    trn = tr[tr.user.isin(trn_users)]
    PL.fit_cooc(trn)
    sc = statcache_view(split_train)
    clfp, colsp = PL.fit_pool_model(trn, meta, training_blocks(tr, trn_users, tpool), sc)
    mm = fit_manner(trn, meta)
    if os.environ.get('CHAMP_EMO_PAIR', '1') == '1':
        import emopair as EP
        mm['pm'] = EP.fit(trn, meta, mm, pool_of=tpool)
    ctx = dict(gm=fit_group_model(trn), mm=mm,
               scorer=PL.make_scorer(clfp, colsp),
               obj=H.fit_object_prior(tr, hold_users))
    ctx['aclf'], ctx['acols'] = H.fit_action_clf(meta, hold_users)
    ctx['acls'] = list(ctx['aclf'].classes_)
    if W_SLOT:
        import slotprior as SPR
        ctx['slotp'] = SPR.fit(tr, meta, hold_users)
    if W_REPAIR:
        import repair as RP
        ctx['gmod'] = RP.fit_gap_model(trn, meta)
    return ctx


# ------------------------------------------------------------------ solving
def solve(vis, ctx, split, diag=None):
    """vis: test-visible QA view with columns idx, clip, true_path, source, category, A..D.

    `diag` accumulates an explicit ledger of cache coverage and of every fallback decision.
    A question whose evidence is entirely missing is returned as None rather than being
    silently answered 'A'; the caller decides what to do with it.
    """
    if diag is None:
        diag = defaultdict(int)
    diag.setdefault('fallback_qids', [])
    pred = {}
    hau = vis[vis.source == 'HAU']
    blocks = infer_blocks(vis, ctx['gm']) if len(hau) else []
    if W_REPAIR and blocks:
        import repair as RP
        t0_of = {}
        for r in hau.drop_duplicates('idx').itertuples():
            if r.true_path in _C['mi']:
                t0_of[r.idx] = _C['meta_idx'].loc[r.true_path, 't0']
        blocks, rlog = RP.repair(blocks, vis, ctx['gm'], ctx['gmod'], t0_of)
        diag['blocks_repaired'] = len(rlog)
    sc = statcache_view(split)
    # --- cache coverage audit
    for p_ in vis.true_path.unique():
        row = _C['meta_idx'].loc[p_] if p_ in _C['mi'] else None
        diag['feat_ok' if (row is not None and np.isfinite(row.get('sk_v_mean', np.nan)))
              else 'feat_missing'] += 1
        diag['logits_ok' if f'{split}|{p_}' in _C['lg'] else 'logits_missing'] += 1
    # --- session pools first: the manner model conditions on the action pool
    pool_of, _blk_pred = {}, []
    for blk in blocks:
        pp0, bd0 = PL.solve_block(vis, blk, sc, ctx['scorer'], split)
        _blk_pred.append((pp0, bd0))
        if bd0:
            for b in blk:
                pool_of[b] = set(bd0['pool'])
    _emo_pool_ctx = {tuple(blk): pool_of.get(blk[0], set()) for blk in blocks}
    # --- emotion
    if len(hau):
        # weights from the held-out emotion sweep: w_phys 0.5 / w_pos 1.0 / w_pair 1.0
        # gave 740/809 vs 731/809 for the unary-only checkpoint
        pe, _ = solve_emotion(vis, blocks, ctx['mm'],
                              w_phys=float(os.environ.get('CHAMP_W_PHYS', 1.0)),
                              w_pos=1.0,
                              w_pair=float(os.environ.get('CHAMP_W_PAIR', 1.0)),
                              pool_of=_emo_pool_ctx,
                              slotp=ctx.get('slotp'), w_slot=W_SLOT)
        pred.update(pe)
    # --- pool -> single / multi / combination (reuse the solve above)
    for pp, bd in _blk_pred:
        pred.update(pp)
        if bd:
            diag['pool_sat' if bd['nsol'] else 'pool_unsat'] += 1
    # --- sequence, from the dense temporal model restricted to the four options
    for r in hau[hau.category == 'sequence'].itertuples():
        oo = opts(r)
        st = stat_of(split, r.true_path)
        if st is None or not all(o in OPT2CLS for o in oo):
            diag['seq_no_evidence'] += 1
            diag['fallback'] += 1
            diag['fallback_qids'].append((r.qa_id, 'sequence_no_dense_logits'))
            pred[r.qa_id] = None
            continue
        diag['seq_ok'] += 1
        ci = [OPT2CLS[o] for o in oo]
        lp = _C['lg'][f'{split}|{r.true_path}']
        sub = np.concatenate([lp[ci], lp[D.BG:D.BG + 1]], 0)
        sub = sub - np.log(np.exp(sub).sum(0, keepdims=True))
        T = sub.shape[1]
        w = np.exp(sub[:4]); w = w / (w.sum(1, keepdims=True) + 1e-9)
        cen = (w * np.arange(T)).sum(1)
        pred[r.qa_id] = ''.join('ABCD'[i] for i in np.argsort(cen))
    # --- HARn single / object
    S2A = H.S2A
    aclf, acols, acls = ctx['aclf'], ctx['acols'], ctx['acls']
    pri, glob = ctx['obj']
    mi, mrow = _C['mi'], _C['meta_idx']
    nrows = vis[vis.source == 'HARn']
    have = [r.true_path for r in nrows.itertuples() if r.true_path in mi]
    PC = {}
    if have:
        Xh = mrow.loc[sorted(set(have)), acols].to_numpy(float)
        pp = aclf.predict_proba(Xh)
        for p, row in zip(sorted(set(have)), pp):
            PC[p] = row
    idx_of = dict(zip(vis.true_path, vis.idx))
    for r in nrows.itertuples():
        par = _C['nest'].get(r.true_path)
        dsc = np.zeros(len(D.ACTIONS))
        if par is not None and f'{split}|{par}' in _C['lg'] and r.true_path in mi:
            lp = _C['lg'][f'{split}|{par}']
            Fp = _C['skel'][mrow.loc[par, 'unit_dir'] + '|F']
            m = (Fp >= mrow.loc[r.true_path, 'f0']) & (Fp <= mrow.loc[r.true_path, 'f1'])
            if m.sum() >= 2:
                dsc = lp[:, m].mean(1)[:D.BG]
        pl = pool_of.get(idx_of.get(par), set()) if par is not None else set()
        pc = PC.get(r.true_path)
        hl = _C['hlog'].get(f'{split}|{r.true_path}')
        hcls = _C['hcls']
        dl = _C['dlog'].get(f'{split}|{r.true_path}')
        dcls = _C['dcls']

        # modality-availability biconditional: no skeleton <=> action is one of the four
        # classes that were never recorded with wearables (measured 50/50 both directions)
        clip_has_sensor = (r.true_path in mi
                           and np.isfinite(mrow.loc[r.true_path, 'f0']))
        allowed = (set(D.ACTIONS) | set(acls)) - H.NO_SENSOR_ACTIONS if clip_has_sensor \
            else set(H.NO_SENSOR_ACTIONS)
        diag['harn_sensor' if clip_has_sensor else 'harn_no_sensor'] += 1

        has_clf = hl is not None and hcls is not None
        has_dino = dl is not None and dcls is not None
        has_agg = pc is not None
        diag['harn_dino_ok' if has_dino else 'harn_dino_missing'] += 1
        has_pool = bool(pl)
        diag['harn_clf_ok' if has_clf else 'harn_clf_missing'] += 1
        diag['harn_pool_ok' if has_pool else 'harn_pool_missing'] += 1

        def score(a):
            s = W_DENSE_INTERVAL * (dsc[D.A2I[a]] if a in D.A2I else -2.0)
            if has_clf and a in hcls:
                s += W_HCLF * hl[hcls.index(a)]
                if has_dino and a in dcls:
                    s += W_HDINO * dl[dcls.index(a)]
            elif has_dino and a in dcls:
                s += W_HDINO * dl[dcls.index(a)]
            elif has_agg and a in acls:
                s += 1.5 * np.log(max(pc[acls.index(a)], 1e-6))
            if has_pool and a in V:
                s += LAM_POOL * (1.0 if V[a] in pl else -1.0)
            return s
        oo = opts(r)
        if not (has_clf or has_dino or has_agg or has_pool) and clip_has_sensor:
            diag['fallback'] += 1
            diag['harn_no_evidence'] += 1
            diag['fallback_qids'].append((r.qa_id, 'harn_no_evidence'))
            pred[r.qa_id] = None
            continue
        if r.category == 'single':
            sv = [score(S2A[o]) if (o in S2A and S2A[o] in allowed) else -1e6 for o in oo]
            if max(sv) <= -1e5:      # filter left nothing -> fall back to unfiltered
                sv = [score(S2A[o]) if o in S2A else -1e6 for o in oo]
                diag['harn_filter_empty'] += 1
            if len(set(np.round(sv, 9))) == 1:
                diag['fallback'] += 1
                diag['harn_flat_scores'] += 1
                diag['fallback_qids'].append((r.qa_id, 'harn_flat_score_vector'))
                pred[r.qa_id] = None
                continue
            o_ = sorted(sv, reverse=True)
            from core import MARGIN
            MARGIN[r.qa_id] = float(o_[0] - o_[1])
            pred[r.qa_id] = 'ABCD'[int(np.argmax(sv))]
        else:
            cand_a = allowed or (set(D.ACTIONS) | set(acls) | set(dcls or []))
            if clip_has_sensor:
                # evidence is available, so commit to the best allowed action
                a = max(cand_a, key=score)
                cnt = [(pri[a][oo[i]], glob[oo[i]]) for i in range(4)]
            else:
                # no wearable evidence exists; the biconditional narrows the action to four,
                # so marginalise the object prior over exactly those (measured 10/10)
                cnt = [(sum(pri[a][oo[i]] for a in cand_a), glob[oo[i]]) for i in range(4)]
            if len(set(cnt)) == 1:
                diag['fallback'] += 1
                diag['object_flat_prior'] += 1
                diag['fallback_qids'].append((r.qa_id, 'object_flat_prior'))
                pred[r.qa_id] = None
                continue
            o_ = sorted([c[0] for c in cnt], reverse=True)
            from core import MARGIN
            MARGIN[r.qa_id] = float(o_[0] - o_[1])
            pred[r.qa_id] = 'ABCD'[max(range(4), key=lambda i: cnt[i])]
    # every question must have been visited
    missing = [q for q in vis.qa_id if q not in pred]
    if missing:
        diag['never_predicted'] = len(missing)
        diag['fallback'] += len(missing)
        diag['fallback_qids'] += [(q, 'never_predicted') for q in missing]
        for q in missing:
            pred[q] = None
    return pred, blocks, pool_of, diag
