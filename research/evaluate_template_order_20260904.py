"""Evaluate transfer of action-script order between repeated HAU session templates.

The test clips expose their action pools and session blocks, but not user IDs.  Training
sessions with the same (a,b) family and/or the same recovered action pool are therefore a
possible non-visual source of sequence supervision.  This script uses only held-out target
sequence labels for scoring; source orders are fit from other users.
"""
import itertools
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
TR = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
TR['user'] = TR.path.str.extract(r'(user\d+)')
TR['aa'] = TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[0]
TR['bb'] = TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[1]
TR['blk'] = list(zip(TR.user, TR.aa, TR.bb))
SEQ = TR[(TR.source == 'HAU') & (TR.category == 'sequence')].copy()

def letters(answer):
    return [x for x in str(answer) if x in 'ABCD']

def options(row):
    return [str(row[x]).strip() for x in 'ABCD']

def pool_of_block(block):
    q = TR[(TR.source == 'HAU') & TR.blk.map(lambda x: x == block) &
           TR.category.isin(['single', 'multi', 'combination', 'sequence'])]
    out = set()
    for _, r in q.iterrows():
        for x in letters(r.answer):
            out.update(p.strip() for p in str(r[x]).split(','))
    return frozenset(out)

BLOCKS = {b: g for b, g in SEQ.groupby('blk')}
POOLS = {b: pool_of_block(b) for b in BLOCKS}

def source_edges(block, user_exclude=None):
    """Return oriented action-pair counts and a partial order for one source block."""
    g = BLOCKS[block]
    if user_exclude is not None and block[0] == user_exclude:
        return {}
    e = Counter()
    for _, r in g.iterrows():
        o = options(r)
        ans = letters(r.answer)
        for i in range(len(ans)):
            for j in range(i + 1, len(ans)):
                x, y = o[ord(ans[i]) - 65], o[ord(ans[j]) - 65]
                if x != y:
                    e[(x, y)] += 1
    return e

EDGES = {b: source_edges(b) for b in BLOCKS}

def pair_vote(target, sources, question_options, source_weight=1.0):
    """Score an order on one four-option question from source partial orders."""
    out = Counter()
    for src, w in sources:
        e = EDGES[src]
        for x, y in itertools.permutations(question_options, 2):
            a, b = e.get((x, y), 0), e.get((y, x), 0)
            if a or b:
                out[(x, y)] += source_weight * w * (a - b)
    coverage = sum(bool(out[(x, y)] or out[(y, x)])
                   for x, y in itertools.combinations(question_options, 2))
    best = None
    second = None
    for p in itertools.permutations(question_options):
        s = sum(out[(p[i], p[j])] for i in range(4) for j in range(i + 1, 4))
        if best is None or s > best[0]:
            second = best
            best = (s, p)
        elif second is None or s > second[0]:
            second = (s, p)
    margin = float(best[0] - (second[0] if second is not None else best[0]))
    return ''.join('ABCD'[question_options.index(x)] for x in best[1]), margin, coverage

def pair_vote_closure(target, sources, question_options):
    """Use transitive closure of a source partial order for otherwise unseen pairs."""
    score = Counter()
    for src, w in sources:
        e = EDGES[src]
        for x, y in itertools.permutations(POOLS[target], 2):
            score[(x, y)] += w * (e.get((x, y), 0) - e.get((y, x), 0))
    graph = defaultdict(set)
    for (x, y), v in score.items():
        if v > 0:
            graph[x].add(y)
        elif v < 0:
            graph[y].add(x)
    reach = defaultdict(set)
    for x in POOLS[target]:
        todo = list(graph[x])
        while todo:
            y = todo.pop()
            if y in reach[x]:
                continue
            reach[x].add(y); todo.extend(graph[y])
    out = Counter()
    for x, y in itertools.permutations(question_options, 2):
        if y in reach[x]: out[(x, y)] = 1.0
        elif x in reach[y]: out[(y, x)] = 1.0
    coverage = sum(bool(out[(x, y)] or out[(y, x)])
                   for x, y in itertools.combinations(question_options, 2))
    best = None; second = None
    for p in itertools.permutations(question_options):
        v = sum(out[(p[i], p[j])] for i in range(4) for j in range(i + 1, 4))
        if best is None or v > best[0]:
            second, best = best, (v, p)
        elif second is None or v > second[0]:
            second = (v, p)
    return ''.join('ABCD'[question_options.index(x)] for x in best[1]), float(best[0] - (second[0] if second else best[0])), coverage

def sources_for(target, mode, exclude_user):
    pool = POOLS[target]
    out = []
    if mode.startswith('nearest_pool'):
        cand = [(b, len(pool & POOLS[b]) / max(1, len(pool | POOLS[b])))
                for b in BLOCKS if b[0] != exclude_user]
        best = max((s for _, s in cand), default=0.0)
        threshold = float(mode.split('_')[-1])
        return [(b, 1.0) for b, s in cand if s >= threshold and abs(s - best) < 1e-12]
    for b in BLOCKS:
        if b[0] == exclude_user:
            continue
        if mode == 'exact_pool' and POOLS[b] == pool:
            out.append((b, 1.0))
        elif mode == 'same_family' and b[1:] == target[1:]:
            out.append((b, 1.0))
        elif mode == 'paired_exact_pool' and abs(int(b[0][4:]) - int(exclude_user[4:])) == 10 and POOLS[b] == pool:
            out.append((b, 1.0))
        elif mode == 'paired_family' and int(b[0][4:]) == int(exclude_user[4:]) + 10 and b[1:] == target[1:]:
            out.append((b, 1.0))
    return out

def build_pred(target, mode, exclude_user):
    src = sources_for(target, mode, exclude_user)
    out, margin = {}, {}
    for _, r in BLOCKS[target].iterrows():
        o = options(r)
        out[r.qa_id], margin[r.qa_id], coverage = pair_vote(target, src, o)
        margin[(r.qa_id, 'coverage')] = coverage
    return out, margin, src

def build_pred_closure(target, mode, exclude_user):
    src = sources_for(target, mode, exclude_user)
    out, margin = {}, {}
    for _, r in BLOCKS[target].iterrows():
        o = options(r)
        out[r.qa_id], margin[r.qa_id], coverage = pair_vote_closure(target, src, o)
        margin[(r.qa_id, 'coverage')] = coverage
    return out, margin, src

# Mechanism-S baseline on the 305 rows with dense coverage.
s_path = os.path.join(ROOT, 'seqlab', 'audit_gated.csv')
baseline_col = 'new'
if not os.path.exists(s_path):
    s_path = os.path.join(ROOT, 'seqlab', 'audit_run18.csv')
    # audit_run18's ``new`` column is the experimental nested-weight arm; the shipped
    # sequence mechanism S is its ``base`` column.
    baseline_col = 'base'
s_audit = pd.read_csv(s_path)
s_audit = s_audit.set_index('qa')
truth = {r.qa_id: ''.join(letters(r.answer)) for _, r in SEQ.iterrows()}
fold = {r.qa: int(r.fold) for _, r in s_audit.reset_index().iterrows()}

def audit(pred, base):
    keys = [q for q in base if q in pred]
    bc = sum(base[q] == truth[q] for q in keys)
    nc = sum(pred[q] == truth[q] for q in keys)
    flips = [q for q in keys if pred[q] != base[q]]
    wr = sum(base[q] != truth[q] and pred[q] == truth[q] for q in flips)
    rw = sum(base[q] == truth[q] and pred[q] != truth[q] for q in flips)
    return dict(n=len(keys), base=bc, new=nc, flips=len(flips), wr=wr, rw=rw,
                precision=wr / max(1, wr + rw), net=wr - rw)

BASE = s_audit[baseline_col].to_dict()
print('sequence rows', len(SEQ), 'S audit rows', len(BASE))
print('blocks', len(BLOCKS), 'pool sizes', Counter(map(len, POOLS.values())))
for mode in ['same_family', 'exact_pool', 'paired_exact_pool', 'paired_family',
             'nearest_pool_0.75', 'nearest_pool_0.85']:
    pred, margins, used = {}, {}, {}
    for b in BLOCKS:
        p, m, src = build_pred(b, mode, b[0])
        pred.update(p); margins.update(m); used[b] = src
    print(mode, audit(pred, BASE))
    print('  covered target blocks', sum(bool(used[b]) for b in BLOCKS),
          'source-counts', Counter(len(v) for v in used.values()))
    rows = []
    for q, p in pred.items():
        if q not in BASE:
            continue
        r = SEQ.loc[SEQ.qa_id == q].iloc[0]
        rows.append(dict(qa=q, user=r.user, blk=r.blk, ans=truth[q], base=BASE[q], new=p,
                         bc=BASE[q] == truth[q], nc=p == truth[q], margin=margins[q],
                         coverage=margins.get((q, 'coverage'), 0)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'research', f'template_{mode}_audit.csv'), index=False)

    for min_cov in [4, 5, 6]:
        gated = {q: p for q, p in pred.items()
                 if margins.get((q, 'coverage'), 0) >= min_cov}
        print('  gate coverage >=', min_cov, audit(gated, BASE))

    if mode in ['exact_pool', 'nearest_pool_0.75', 'nearest_pool_0.85']:
        cp, cm, _ = {}, {}, {}
        for b in BLOCKS:
            p, m, src = build_pred_closure(b, mode, b[0])
            cp.update(p); cm.update(m); cm.update({(q, 'coverage'): m.get((q, 'coverage'), 0) for q in p})
        for min_cov in [5, 6]:
            gated = {q: p for q, p in cp.items() if cm.get((q, 'coverage'), 0) >= min_cov}
            print('  transitive gate coverage >=', min_cov, audit(gated, BASE))

# Pairing diagnostics: compare exact action-pool and edge orientation stability for u and u+10.
print('\npaired-template diagnostics')
for u in sorted(TR.user.dropna().unique(), key=lambda x: int(x[4:])):
    v = f'user{int(u[4:]) + 10}'
    if v not in set(TR.user.dropna()):
        continue
    common = set(BLOCKS) & {(v, a, b) for _, a, b in BLOCKS if False}
    matches = [(b, (v, b[1], b[2])) for b in BLOCKS if b[0] == u and (v, b[1], b[2]) in BLOCKS]
    if not matches:
        continue
    same_pool = sum(POOLS[a] == POOLS[b] for a, b in matches)
    orient = total = 0
    for a, b in matches:
        ea, eb = EDGES[a], EDGES[b]
        for x, y in set(ea) | set(eb):
            if (x, y) not in ea or (y, x) not in eb:
                continue
            total += 1
            orient += int((ea[(x, y)] - ea.get((y, x), 0)) *
                          (eb[(x, y)] - eb.get((y, x), 0)) > 0)
    print(u, v, 'common_ab', len(matches), 'same_pool', same_pool,
          'edge_agree', f'{orient}/{total}' if total else 'n/a')
