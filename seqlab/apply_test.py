"""Apply the validated sequence mechanism to the real test set.

Mechanism (OOF: 172/305 -> 225/305, 112 flips at 0.853 precision, positive in all 5 folds):
  score(total order of the block's option union)
      = 4*crossclip_pairwise(tau=0.5) + 4*crossclip_pairwise(tau=1.0) + 1*learned_pairwise
  decoded jointly over the block, so the 2-3 sequence questions of a session are forced to be
  restrictions of ONE latent order (verified conflict-free on 104/104 training triples).
Blocks come from the champion's own DP segmentation; nothing here reads a test answer.
"""
import os, sys, json, itertools, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0, 'champ'); sys.path.insert(0, 'seqlab')
from core import load_all, real_test_view, fit_group_model, infer_blocks, opts
import dense as D, decode as DC
from pairclf import clip_stats, feat, OPT2CLS as O2C
from prior import load_sources
from fast import Block, PERMS
from sklearn.ensemble import HistGradientBoostingClassifier

tr, te, meta = load_all()
vis = real_test_view(te)
gm = fit_group_model(tr)
blocks = infer_blocks(vis, gm)
print('inferred %d test blocks; sizes %s' % (len(blocks),
      dict(collections.Counter(len(b) for b in blocks))))
hau = vis[vis.source == 'HAU']
seq = hau[hau.category == 'sequence']
print('test sequence questions:', len(seq))
idx2clip = dict(zip(vis.idx, vis.true_path))
blk_of = {}
for b in blocks:
    for i in b: blk_of[i] = tuple(b)
sq_blocks = collections.defaultdict(list)
for r in seq.itertuples(): sq_blocks[blk_of[r.idx]].append(r)
print('blocks containing sequence questions: %d  (questions per block %s)' % (
    len(sq_blocks), dict(collections.Counter(len(v) for v in sq_blocks.values()))))
for b, v in sorted(sq_blocks.items()):
    print('   block', b, '-> seq clips', [r.idx for r in v])

# ---------------- evidence, trained on ALL 18 training users
S = pd.read_csv('seqlab/onsets.csv'); Ssrc, seg, qa = load_sources()
onmu = {a: float(g.on.mean()) for a, g in S.groupby('act')}
prlo = {}
for (x, y) in set(tuple(sorted(k)) for k in list(seg) + list(qa)):
    nxy = sum(seg.get((x, y), {}).values()) + sum(qa.get((x, y), {}).values())
    nyx = sum(seg.get((y, x), {}).values()) + sum(qa.get((y, x), {}).values())
    n = nxy + nyx
    if n == 0: continue
    p = (nxy + 1) / (n + 2); lo = float(np.log(p / (1 - p)))
    prlo[(x, y)] = lo; prlo[(y, x)] = -lo

ST_tr = {}
for p in tr[tr.source == 'HAU'].path.unique():
    lp = DC.logits('oof', p)
    if lp is not None: ST_tr[p] = clip_stats(lp)
segord = collections.defaultdict(list)
for r in S.itertuples(): segord[(r.user, r.trial)].append((r.on, r.act))
LAB = []
for (u, t), v in segord.items():
    p = f'HAU/{u}/{t}'
    if p not in ST_tr: continue
    v = sorted(v)
    for i in range(len(v)):
        for j in range(len(v)):
            if i == j or v[i][1] == v[j][1] or v[i][0] == v[j][0]: continue
            if v[i][1] in O2C and v[j][1] in O2C: LAB.append((p, v[i][1], v[j][1], 1))
sq_tr = tr[(tr.source == 'HAU') & (tr.category == 'sequence')]
for _, r in sq_tr.iterrows():
    if r.path not in ST_tr: continue
    o = {L: r[L] for L in 'ABCD'}; ordr = [o[L] for L in str(r['answer'])]
    for i in range(4):
        for j in range(4):
            if i != j and ordr[i] in O2C and ordr[j] in O2C:
                LAB.append((r.path, ordr[i], ordr[j], int(i < j)))
X = pd.DataFrame([feat(ST_tr[p], x, y, prlo, onmu) for (p, x, y, _) in LAB])
yv = np.array([l[3] for l in LAB])
clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=6,
                                     l2_regularization=1.0, random_state=0).fit(X.to_numpy(float), yv)
cols = list(X.columns)
print('\nlearned pairwise model fitted on %d labelled pairs (all 18 users)' % len(LAB))

def pairlo_hand(lp, acts, tau, conf=True):
    ci = [O2C[a] for a in acts]; sub = lp[ci]
    q = np.exp((sub - sub.max(1, keepdims=True)) / tau); q = q / (q.sum(1, keepdims=True) + 1e-12)
    C = np.cumsum(q, axis=1); n = len(acts); raw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: raw[i, j] = (q[j][1:] * C[i][:-1]).sum()
    P = np.exp(lp); tot = P.sum(0) + 1e-12
    cf = np.array([float((P[c] / tot).max()) for c in ci]) if conf else np.ones(n)
    R = {}
    for i in range(n):
        for j in range(i + 1, n):
            p = raw[i, j] / (raw[i, j] + raw[j, i] + 1e-12); p = float(np.clip(p, 1e-3, 1 - 1e-3))
            lo = np.log(p / (1 - p)) * float(np.sqrt(cf[i] * cf[j]))
            R[(acts[i], acts[j])] = lo; R[(acts[j], acts[i])] = -lo
    return R

W = (4.0, 4.0, 1.0)
out = {}
audit = []
for b, rows in sorted(sq_blocks.items()):
    rows = sorted(rows, key=lambda r: r.idx)
    qo = [opts(r) for r in rows]
    U = sorted(set().union(*[set(o) for o in qo]))
    Ua = [a for a in U if a in O2C]
    if len(Ua) < len(U):
        audit.append(dict(block=str(b), note='option outside 40-action vocab; skipped'))
        continue
    B = Block(U, qo)
    hand = {}
    for tau in (0.5, 1.0):
        acc = collections.Counter(); ncl = 0
        for i in b:
            lp = DC.logits('test', idx2clip[i])
            if lp is None: continue
            ncl += 1
            for k, v in pairlo_hand(lp, Ua, tau).items(): acc[k] += v
        hand[tau] = {k: v / max(ncl, 1) for k, v in acc.items()}
    frows = []; key = []
    for i in b:
        p = idx2clip[i]; lp = DC.logits('test', p)
        if lp is None: continue
        st = clip_stats(lp)
        for a in range(len(Ua)):
            for c in range(len(Ua)):
                if a != c: frows.append(feat(st, Ua[a], Ua[c], prlo, onmu)); key.append((Ua[a], Ua[c]))
    ag = collections.defaultdict(list)
    if frows:
        pp = clf.predict_proba(pd.DataFrame(frows).reindex(columns=cols).to_numpy(float))[:, 1]
        for k, v in zip(key, pp):
            v = float(np.clip(v, 1e-3, 1 - 1e-3)); ag[k].append(np.log(v / (1 - v)))
    learn = {}
    for (x, y) in set(ag):
        m = (np.mean(ag[(x, y)]) - np.mean(ag[(y, x)])) / 2 if (y, x) in ag else np.mean(ag[(x, y)])
        learn[(x, y)] = float(m)
    t = B.score([(W[0], hand[0.5]), (W[1], hand[1.0]), (W[2], learn)],
                [(0.0, np.zeros(24)) for _ in rows])
    k0 = int(np.argmax(t))
    order = [B.U[j] for j in np.argsort(B.rank[k0])]
    for qi, r in enumerate(rows):
        lab = PERMS[int(B.L[qi][k0])]
        other = t[B.L[qi] != B.L[qi][k0]]
        out[r.qa_id] = lab
        audit.append(dict(qa_id=r.qa_id, clip=r.idx, block=str(b), pred=lab,
                          margin=float(t[k0] - other.max()) if len(other) else np.inf,
                          latent_order=' < '.join(order)))
A = pd.DataFrame(audit)
A.to_csv('seqlab/test_sequence_audit.csv', index=False)
json.dump(out, open('seqlab/test_sequence_pred.json', 'w'), indent=1)
print('\ndecoded %d test sequence questions' % len(out))
sub = pd.read_csv('submission_092105_SUBMITTED.csv').set_index('qa_id')
ch = {q: sub.loc[q, 'prediction'] for q in out}
nfl = sum(1 for q in out if out[q] != ch[q])
print('flips vs the 0.92105 champion: %d of %d' % (nfl, len(out)))
A['champ'] = [ch.get(q) for q in A.qa_id] if 'qa_id' in A else None
A['flip'] = A.pred != A.champ
A.to_csv('seqlab/test_sequence_audit.csv', index=False)
print(A[['qa_id','clip','block','champ','pred','flip','margin']].to_string())
