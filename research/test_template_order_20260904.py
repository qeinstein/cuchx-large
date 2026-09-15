import itertools
import json
import os
import pickle
import re
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
TR = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
TE = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
SUB = pd.read_csv(os.path.join(ROOT, 'submissions/submission_093859_SUBMITTED.csv')).set_index('qa_id')
SNAP = pickle.load(open(os.path.join(ROOT, 'research', 'test_pool_snapshot.pkl'), 'rb'))

TR['user'] = TR.path.str.extract(r'(user\d+)')
TR['aa'] = TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[0]
TR['bb'] = TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[1]
TR['blk'] = list(zip(TR.user, TR.aa, TR.bb))
SEQ = TR[(TR.source == 'HAU') & (TR.category == 'sequence')]

def letters(a):
    return [x for x in str(a) if x in 'ABCD']

def opts(r):
    return [str(r[x]).strip() for x in 'ABCD']

def pool(block):
    q = TR[(TR.source == 'HAU') & TR.blk.map(lambda x: x == block) &
           TR.category.isin(['single', 'multi', 'combination', 'sequence'])]
    out = set()
    for _, r in q.iterrows():
        for x in letters(r.answer):
            out.update(p.strip() for p in str(r[x]).split(','))
    return frozenset(out)

blocks = {b: g for b, g in SEQ.groupby('blk')}
edges = {}
pools = {}
for b, g in blocks.items():
    pools[b] = pool(b)
    e = Counter()
    for _, r in g.iterrows():
        o = opts(r); a = letters(r.answer)
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                e[(o[ord(a[i]) - 65], o[ord(a[j]) - 65])] += 1
    edges[b] = e

def choose(qopts, sources):
    score = Counter()
    for b in sources:
        e = edges[b]
        for x, y in itertools.permutations(qopts, 2):
            score[(x, y)] += e.get((x, y), 0) - e.get((y, x), 0)
    cov = sum(bool(score[(x, y)] or score[(y, x)])
              for x, y in itertools.combinations(qopts, 2))
    best = None; second = None
    for p in itertools.permutations(qopts):
        v = sum(score[(p[i], p[j])] for i in range(4) for j in range(i + 1, 4))
        if best is None or v > best[0]:
            second, best = best, (v, p)
        elif second is None or v > second[0]:
            second = (v, p)
    margin = best[0] - (second[0] if second else best[0])
    return ''.join('ABCD'[qopts.index(x)] for x in best[1]), cov, margin

def choose_closure(qopts, sources, pool):
    score = Counter()
    for b in sources:
        e = edges[b]
        for x, y in itertools.permutations(pool, 2):
            score[(x, y)] += e.get((x, y), 0) - e.get((y, x), 0)
    graph = {x: set() for x in pool}
    for (x, y), v in score.items():
        if v > 0: graph[x].add(y)
        elif v < 0: graph[y].add(x)
    reach = {x: set() for x in pool}
    for x in pool:
        todo = list(graph[x])
        while todo:
            y = todo.pop()
            if y in reach[x]: continue
            reach[x].add(y); todo.extend(graph[y])
    out = Counter()
    for x, y in itertools.permutations(qopts, 2):
        if y in reach[x]: out[(x, y)] = 1.0
        elif x in reach[y]: out[(y, x)] = 1.0
    cov = sum(bool(out[(x, y)] or out[(y, x)]) for x, y in itertools.combinations(qopts, 2))
    best = None; second = None
    for p in itertools.permutations(qopts):
        v = sum(out[(p[i], p[j])] for i in range(4) for j in range(i + 1, 4))
        if best is None or v > best[0]: second, best = best, (v, p)
        elif second is None or v > second[0]: second = (v, p)
    return ''.join('ABCD'[qopts.index(x)] for x in best[1]), cov, best[0] - (second[0] if second else best[0])

by_idx = {int(re.search(r'LM_test_(\d+)', p).group(1)): p for p in TE.path}
seq = TE[TE.category == 'sequence'].copy()
seq['idx'] = seq.path.str.extract(r'LM_test_(\d+)').astype(int)
block_for = {}
for b in SNAP['blocks']:
    for i in b:
        block_for[int(i)] = tuple(int(x) for x in b)

rows = []
rows_closure = []
for _, r in seq.iterrows():
    b = block_for.get(int(r.idx))
    if b is None:
        continue
    p = frozenset(SNAP['pool_of'].get(b[0], []))
    src = [x for x in blocks if pools[x] == p]
    pred, cov, margin = choose(opts(r), src)
    pred_c, cov_c, margin_c = choose_closure(opts(r), src, p)
    old = SUB.loc[r.qa_id, 'prediction']
    rows.append(dict(qa_id=r.qa_id, idx=int(r.idx), block=str(b), pool='|'.join(sorted(p)),
                     nsrc=len(src), coverage=cov, margin=margin, old=old, new=pred,
                     flip=old != pred, closure=pred_c, closure_coverage=cov_c,
                     closure_margin=margin_c, closure_flip=old != pred_c,
                     sources=';'.join(map(str, src))))

for threshold in [0.70, 0.75, 0.80]:
    nearest_rows = []
    for _, r in seq.iterrows():
        b = block_for.get(int(r.idx))
        if b is None: continue
        p = frozenset(SNAP['pool_of'].get(b[0], []))
        sims = [(x, len(p & pools[x]) / max(1, len(p | pools[x]))) for x in blocks]
        best_sim = max((s for _, s in sims), default=0.0)
        src = [x for x, s in sims if s >= threshold and abs(s - best_sim) < 1e-12]
        pred, cov, margin = choose_closure(opts(r), src, p)
        nearest_rows.append(dict(qa_id=r.qa_id, idx=int(r.idx), sim=best_sim,
                                 nsrc=len(src), coverage=cov, margin=margin,
                                 old=SUB.loc[r.qa_id, 'prediction'], new=pred,
                                 flip=SUB.loc[r.qa_id, 'prediction'] != pred,
                                 sources=';'.join(map(str, src))))
    nd = pd.DataFrame(nearest_rows)
    nd.to_csv(os.path.join(ROOT, 'research', f'test_template_nearest_{threshold:.2f}.csv'), index=False)
    print('nearest', threshold, 'covered', sum(nd.nsrc > 0),
          'complete', sum((nd.nsrc > 0) & (nd.coverage == 6)),
          'gated flips', sum((nd.nsrc > 0) & (nd.coverage == 6) & nd.flip))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, 'research', 'test_template_order_audit.csv'), index=False)
print('test sequence rows', len(out), 'template-covered', sum(out.nsrc > 0),
      'complete-pair', sum(out.coverage == 6), 'flips', sum(out.flip),
      'closure-complete', sum((out.nsrc > 0) & (out.closure_coverage == 6)),
      'closure-flips', sum((out.nsrc > 0) & (out.closure_coverage == 6) & out.closure_flip))
print(out.to_string(index=False))
