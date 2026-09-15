"""Joint sequence decoding over inferred session blocks (T2 validation mechanism).

Reimplements two class-(a) sequence mechanisms WITHOUT dense logits, using only
test-visible inputs + already-fitted artifacts:

  1. SEQTMPL: exact-pool template transfer with the transitive-closure six-pair gate.
     Faithful port of research/test_template_order_20260904.py choose/choose_closure.
     The recovered test pools come from the STRUCT pool solver (CHAMP_POOL_FEATS=struct,
     empty dense statcache) instead of the missing test_pool_snapshot.pkl.
  2. CONSIST: cross-question mutual-inconsistency repair (the testb006/testb021 fixes that
     produced test_0335 DBAC->DBCA and test_0647 DCAB->DCBA). All sequence answers in one
     block must be restrictions of ONE latent total order; on contradiction the weaker-
     evidenced question is repaired with the minimal Kendall-tau move that resolves a
     contradiction without creating a new one.

Skeleton timing (champ/skel_seq.npz) is profiled per clip for audit only: without
action-labeled segments, motion energy cannot order actions, so it is NOT a decision
input. The tie-break is exact-pool donor agreement, then historical margin.

Usage:
  python3 research/t2_valid/mech_seq_joint.py --base submission_final.csv
  python3 research/t2_valid/mech_seq_joint.py --base submission_095614_327of342_CHAMPION.csv
"""
import argparse
import itertools
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
sys.path.insert(0, ROOT)

os.environ.setdefault('CHAMP_POOL_FEATS', 'struct')

from core import real_test_view  # noqa: E402
import pool as PL  # noqa: E402
import pipeline as PIPE  # noqa: E402  (training_blocks/true_pool only; no dense cache load)


# ---------------------------------------------------------------- test blocks
def load_blocks():
    with open(os.path.join(ROOT, 'champ', 'test_blocks_repaired.json')) as f:
        blocks = [tuple(int(x) for x in b) for b in json.load(f)]
    return blocks


def seq_questions():
    te = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
    sq = te[te.category == 'sequence'].copy()
    sq['idx'] = sq.path.str.extract(r'LM_test_(\d+)').astype(int)
    return sq


# ------------------------------------------------- training pools + pair edges
def train_pools_edges(tr):
    tr = tr.copy()
    tr['user'] = tr.path.str.extract(r'(user\d+)')
    ab = tr.path.str.extract(r'/(\d+)-(\d+)-(\d+)')
    tr['aa'], tr['bb'] = ab[0], ab[1]
    tr['blk'] = list(zip(tr.user, tr.aa, tr.bb))
    act = tr[(tr.source == 'HAU')
             & tr.category.isin(['single', 'multi', 'combination', 'sequence'])]

    def letters(a):
        return [x for x in str(a) if x in 'ABCD']

    pools, edges = {}, {}
    for b, g in act.groupby('blk'):
        p = set()
        for _, r in g.iterrows():
            for x in letters(r.answer):
                p.update(t.strip() for t in str(r[x]).split(','))
        pools[b] = frozenset(p)
    seq = tr[(tr.source == 'HAU') & (tr.category == 'sequence')]
    for b, g in seq.groupby('blk'):
        e = Counter()
        for _, r in g.iterrows():
            o = [str(r[x]).strip() for x in 'ABCD']
            a = letters(r.answer)
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    e[(o[ord(a[i]) - 65], o[ord(a[j]) - 65])] += 1
        edges[b] = e
    return pools, edges


# ------------------------------------------- template transfer (faithful port)
def choose_closure(qopts, sources, pool, edges):
    score = Counter()
    for b in sources:
        e = edges.get(b, {})
        for x, y in itertools.permutations(pool, 2):
            score[(x, y)] += e.get((x, y), 0) - e.get((y, x), 0)
    graph = {x: set() for x in pool}
    for (x, y), v in score.items():
        if v > 0:
            graph[x].add(y)
        elif v < 0:
            graph[y].add(x)
    reach = {x: set() for x in pool}
    for x in pool:
        todo = list(graph[x])
        while todo:
            y = todo.pop()
            if y in reach[x]:
                continue
            reach[x].add(y)
            todo.extend(graph[y])
    out = Counter()
    for x, y in itertools.permutations(qopts, 2):
        if y in reach[x]:
            out[(x, y)] = 1.0
        elif x in reach[y]:
            out[(y, x)] = 1.0
    cov = sum(bool(out[(x, y)] or out[(y, x)])
              for x, y in itertools.combinations(qopts, 2))
    best = None
    second = None
    for p in itertools.permutations(qopts):
        v = sum(out[(p[i], p[j])] for i in range(4) for j in range(i + 1, 4))
        if best is None or v > best[0]:
            second, best = best, (v, p)
        elif second is None or v > second[0]:
            second = (v, p)
    pred = ''.join('ABCD'[qopts.index(x)] for x in best[1])
    return pred, cov, best[0] - (second[0] if second else best[0])


def recover_pools_struct(blocks):
    """Recovered action pool per test block via the STRUCT-only pool solver."""
    from core import load_all
    tr, te, meta = load_all()
    trn = tr[tr.user.notna()]
    PL.fit_cooc(trn)
    tpool = PIPE.true_pool(tr)
    users = sorted(trn.user.dropna().unique())
    bbu = PIPE.training_blocks(tr, users, tpool)
    clf, cols = PL.fit_pool_model(trn, meta, bbu, {})
    scorer = PL.make_scorer(clf, cols)
    vis = real_test_view(te)
    pool_of = {}
    for blk in blocks:
        _pp, bd = PL.solve_block(vis, list(blk), {}, scorer, 'test')
        if bd:
            pool_of[blk] = frozenset(bd['pool'])
    return pool_of


def template_transfer(sq, base_pred, blocks, pool_of, pools, edges):
    block_of = {}
    for b in blocks:
        for i in b:
            block_of[i] = b
    out, audit = {}, []
    for _, r in sq.iterrows():
        b = block_of.get(int(r.idx))
        if b is None:
            continue
        p = pool_of.get(b)
        if p is None:
            audit.append(dict(qa_id=r.qa_id, mech='SEQTMPL', action='skipped',
                              why='no recovered pool'))
            continue
        qopts = [str(r[x]).strip() for x in 'ABCD']
        src = [x for x in pools if pools[x] == p]
        pred_c, cov_c, margin_c = choose_closure(qopts, src, p, edges)
        old = str(base_pred[r.qa_id])
        fire = (len(src) == 1 and cov_c == 6 and pred_c != old)
        audit.append(dict(qa_id=r.qa_id, mech='SEQTMPL',
                          action='applied' if fire else 'no-fire',
                          old=old, new=pred_c, nsrc=len(src),
                          closure_coverage=cov_c, closure_margin=margin_c,
                          sources=';'.join(map(str, src))))
        if fire:
            out[r.qa_id] = pred_c
    return out, audit


# ------------------------------------------------------- consistency repair
def order_of(row, pred):
    o = [str(row[x]).strip() for x in 'ABCD']
    return [o[ord(c) - 65] for c in str(pred) if c in 'ABCD']


def contradictions(orders):
    """pair -> {+1: [qa...], -1: [qa...]} for pairs ordered differently."""
    votes = {}
    for qa, actions in orders.items():
        pos = {a: i for i, a in enumerate(actions)}
        for x, y in itertools.combinations(sorted(set(actions)), 2):
            if x not in pos or y not in pos:
                continue
            s = 1 if pos[x] < pos[y] else -1
            votes.setdefault((x, y), {}).setdefault(s, []).append(qa)
    return {p: v for p, v in votes.items() if len(v) > 1}


def donor_agreement(actions, test_pool, pools, edges):
    """Total exact-pool donor edge weight agreeing with this order."""
    src = [x for x in pools if pools[x] == test_pool]
    pos = {a: i for i, a in enumerate(actions)}
    tot = 0
    for b in src:
        e = edges.get(b, {})
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                x, y = actions[i], actions[j]
                tot += e.get((x, y), 0) - e.get((y, x), 0)
    return tot, len(src)


def kendall(a, b):
    pos = {x: i for i, x in enumerate(b)}
    n = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if pos[a[i]] > pos[a[j]]:
                n += 1
    return n


def repair_block(qs, preds, test_pool, pools, edges, margins, audit,
                 anchors):
    """Joint anchored decode: template-anchored questions are locked; the rest are
    searched jointly (24^k combos) for min (contradictions, tau-move, -agreement).

    A block with no anchor abstains: repairing two untrusted answers against each
    other propagates errors (measured: all 4 v1 wrong repairs were unanchored,
    all 3 v1 correct repairs were template-anchored). This mirrors history: the
    testb006/testb021 repairs aligned the S answer to the template-backed sibling.
    """
    rows = [(r.qa_id, [str(r[x]).strip() for x in 'ABCD']) for _, r in qs.iterrows()]
    cur = {qa: order_of(qs[qs.qa_id == qa].iloc[0], preds[qa]) for qa, _ in rows}
    locked = {qa for qa, _ in rows if qa in anchors}
    free = [qa for qa, _ in rows if qa not in anchors]
    if not locked:
        n = len(contradictions(cur))
        if n:
            audit.append(dict(qa_id=';'.join(sorted(cur)), mech='CONSIST',
                              action='abstained',
                              why=f'{n} contradicting pairs but no '
                                  f'template-anchored question in block'))
        return cur
    if not free or not contradictions(cur):
        return cur
    qo = dict(rows)
    perms = {qa: [list(p) for p in itertools.permutations(qo[qa])] for qa in free}
    base_contra = len(contradictions(cur))
    best = None
    for combo in itertools.product(*(perms[qa] for qa in free)):
        trial = dict(cur)
        for qa, p in zip(free, combo):
            trial[qa] = p
        nc = len(contradictions(trial))
        tau = sum(kendall(cur[qa], trial[qa]) for qa in free)
        ag = sum(donor_agreement(trial[qa], test_pool, pools, edges)[0]
                 for qa in free)
        key = (nc, tau, -ag)
        if best is None or key < best[0]:
            best = (key, trial)
    (nc, _tau, _ag), trial = best
    if nc >= base_contra:
        audit.append(dict(qa_id=';'.join(sorted(free)), mech='CONSIST',
                          action='unresolved',
                          why=f'joint optimum keeps {nc} contradicting pairs '
                              f'(base {base_contra})'))
        return cur
    for qa in free:
        if trial[qa] != cur[qa]:
            audit.append(dict(
                qa_id=qa, mech='CONSIST', action='applied',
                old=''.join('ABCD'[qo[qa].index(x)] for x in cur[qa]),
                new=''.join('ABCD'[qo[qa].index(x)] for x in trial[qa]),
                kendall_tau=kendall(cur[qa], trial[qa]),
                donor_agree=donor_agreement(trial[qa], test_pool, pools,
                                            edges)[0],
                contra_before=base_contra, contra_after=nc,
                anchors=';'.join(sorted(locked))))
    return trial


def skeleton_timing_audit(qs, meta):
    """Per-clip motion-energy timing profile (AUDIT ONLY, not a decision input)."""
    out = {}
    try:
        z = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    except Exception as e:
        return {'error': str(e)}
    unit_of = {}
    for r in meta[meta.kind == 'test'].itertuples():
        m = re.search(r'LM_test_(\d+)', str(r.qa_path))
        if m:
            unit_of[int(m.group(1))] = r.unit_dir
    keys = set(z.files)
    for _, r in qs.iterrows():
        idx = int(r.idx)
        u = unit_of.get(idx)
        k = (u + '|K') if u else None
        if not k or k not in keys:
            out[r.qa_id] = {'frames': None}
            continue
        K = np.asarray(z[k], dtype=float)  # (T, 17, 3)
        if K.shape[0] < 2:
            out[r.qa_id] = {'frames': int(K.shape[0])}
            continue
        v = np.diff(K, axis=0)
        e = np.sqrt((v ** 2).sum(-1)).sum(-1)  # per-frame motion energy
        t = np.arange(len(e))
        tot = e.sum()
        out[r.qa_id] = {'frames': int(K.shape[0]),
                        'energy_centroid_frac': float((e * t).sum() / tot / max(1, len(e) - 1))
                        if tot > 0 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--audit', default=None)
    args = ap.parse_args()
    os.chdir(ROOT)

    base = pd.read_csv(args.base)
    base_pred = dict(zip(base.qa_id, base['prediction'].astype(str)))
    champ = pd.read_csv(os.path.join(ROOT, 'submissions/submission_097076_332of342_CHAMPION.csv'))
    champ_pred = dict(zip(champ.qa_id, champ['prediction'].astype(str)))
    tr = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
    blocks = load_blocks()
    sq = seq_questions()
    try:
        margins = json.load(open(os.path.join(ROOT, 'champ', 'test_margins.json')))
        margins = {k: (float(v) if v is not None else float('inf'))
                   for k, v in margins.items()}
    except Exception:
        margins = {}

    pools, edges = train_pools_edges(tr)
    print(f'training blocks: {len(pools)}; test blocks: {len(blocks)}; '
          f'test seq questions: {len(sq)}')
    pool_of = recover_pools_struct(blocks)
    print(f'recovered struct pools for {len(pool_of)}/{len(blocks)} blocks')

    audit = []
    # stage 1: template transfer
    tmpl, a1 = template_transfer(sq, base_pred, blocks, pool_of, pools, edges)
    audit += a1
    pred = dict(base_pred)
    pred.update(tmpl)
    print(f'SEQTMPL: {len(tmpl)} flips: {sorted(tmpl.items())}')

    # stage 2: anchored joint consistency repair per block.
    # Anchors = questions whose post-stage-1 answer equals the exact-pool closure
    # order (nsrc==1, coverage==6): just-fired transfers plus already-correct rows.
    closure_of = {a['qa_id']: a for a in a1}
    anchors = {q for q, a in closure_of.items()
               if a.get('nsrc') == 1 and a.get('closure_coverage') == 6
               and str(pred.get(q)) == str(a.get('new'))}
    print(f'template anchors: {len(anchors)} {sorted(anchors)}')
    block_of = {}
    for b in blocks:
        for i in b:
            block_of[i] = b
    sq['block'] = sq.idx.map(block_of)
    n_repaired = 0
    for b, g in sq.groupby('block', sort=False):
        if len(g) < 2:
            continue
        p = pool_of.get(tuple(b) if isinstance(b, tuple) else b)
        orders = repair_block(g, pred, p, pools, edges, margins, audit, anchors)
        for _, r in g.iterrows():
            qopts = [str(r[x]).strip() for x in 'ABCD']
            new = ''.join('ABCD'[qopts.index(x)] for x in orders[r.qa_id])
            if new != str(pred[r.qa_id]):
                n_repaired += 1
                pred[r.qa_id] = new
    print(f'CONSIST: {n_repaired} repaired rows')

    # skeleton timing audit (non-decisional)
    from core import load_all
    _tr, _te, meta = load_all()
    skel = skeleton_timing_audit(sq, meta)
    n_prof = sum(1 for v in skel.values()
                 if isinstance(v, dict) and v.get('frames'))
    print(f'skeleton timing profiles: {n_prof}/{len(sq)} clips (audit only)')

    # score vs champion on sequence rows
    flips = [q for q in base_pred
             if str(base_pred[q]) != str(pred[q]) and q.startswith('test_')]
    seq_ids = set(sq.qa_id)
    sflips = [q for q in flips if q in seq_ids]
    match = [q for q in sflips if str(pred[q]) == str(champ_pred[q])]
    print(f'total flips: {len(flips)}; sequence flips: {len(sflips)}; '
          f'matching champion: {len(match)}')
    for q in sorted(sflips):
        flag = 'MATCH' if q in match else 'DIFF '
        print(f'  {flag} {q}: {base_pred[q]} -> {pred[q]} '
              f'(champ {champ_pred[q]})')
    # champion sequence rows NOT reproduced from this base
    champ_seq = [q for q in seq_ids
                 if str(champ_pred[q]) != str(base_pred.get(q))]
    missed = [q for q in champ_seq if str(pred.get(q)) != str(champ_pred[q])]
    print(f'champion-seq diffs from base: {len(champ_seq)}; '
          f'still missed: {len(missed)} {sorted(missed)}')

    tag = os.path.basename(args.base).replace('.csv', '')
    out_p = args.out or os.path.join(ROOT, 'research', 't2_valid',
                                     f'seqjoint_from_{tag}.csv')
    au_p = args.audit or os.path.join(ROOT, 'research', 't2_valid',
                                      f'seqjoint_from_{tag}.audit.csv')
    pd.DataFrame([dict(qa_id=q, prediction=pred[q])
                  for q in base.qa_id]).to_csv(out_p, index=False)
    pd.DataFrame(audit).to_csv(au_p, index=False)
    json.dump(skel, open(os.path.join(ROOT, 'research', 't2_valid',
                                      f'seqjoint_from_{tag}.skel.json'), 'w'),
              indent=1)
    print('wrote', out_p, 'and', au_p)


if __name__ == '__main__':
    main()
