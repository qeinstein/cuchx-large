"""Which ACTION PAIRS does the sequence decoder invert, and is the pattern systematic?

Mechanism S leaves 80 of 305 held-out sequence questions wrong, and 47 of those 80 are a
SINGLE pairwise inversion from the truth -- the latent order is nearly right, one adjacent
pair is swapped.  Sequence is 4.9 of the ~20 remaining public errors, the largest pool not yet
bounded, so this asks whether the inversions concentrate on identifiable action pairs.

Three outcomes are possible and they imply different things:
  * a few action pairs invert repeatedly across sessions  -> a learnable per-pair correction
  * inversions spread thinly over many pairs             -> per-session evidence noise, no fix
  * inversions concentrate on pairs with no HARn segment -> a coverage problem, not a model one

The last is testable directly: `seqpair`'s supervision comes from nested HARn segment onsets,
which cover only ~86.9% of each clip's action pool.
"""
import os, sys, json, pickle, itertools, collections
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'seqlab'))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all

cache = pickle.load(open(os.path.join(ROOT, 'seqlab', 'cache.pkl'), 'rb'))
A = pd.read_csv(os.path.join(ROOT, 'seqlab', 'audit_run18.csv'))


def inversions(truth, pred, opts):
    """-> list of (earlier_action, later_action) pairs that pred gets the wrong way round."""
    rt = {c: i for i, c in enumerate(truth)}
    rp = {c: i for i, c in enumerate(pred)}
    out = []
    for x, y in itertools.combinations('ABCD', 2):
        if x not in rt or y not in rt or x not in rp or y not in rp:
            continue
        if (rt[x] < rt[y]) != (rp[x] < rp[y]):
            a, b = (x, y) if rt[x] < rt[y] else (y, x)
            out.append((opts['ABCD'.index(a)], opts['ABCD'.index(b)]))
    return out


rows = []
for r in A.itertuples():
    e = cache.get(r.qa)
    if e is None:
        continue
    o = e['opts']
    t, p = str(r.ans), str(r.new)
    if len(t) != 4 or len(p) != 4:
        continue
    inv = inversions(t, p, o)
    rows.append(dict(qa=r.qa, fold=r.fold, blk=str(e['blk']), ninv=len(inv),
                     correct=int(t == p), ncov=r.ncov, inv=inv))
D = pd.DataFrame(rows)
print(f'sequence questions audited: {len(D)}   correct {D.correct.sum()}   '
      f'wrong {int((~D.correct.astype(bool)).sum())}')
print('\ninversion count distribution among WRONG answers:')
print(D[D.correct == 0].ninv.value_counts().sort_index().to_string())

W = D[D.correct == 0]
cnt = collections.Counter(p for r in W.itertuples() for p in r.inv)
tot_pairs = sum(cnt.values())
print(f'\ntotal inverted (earlier, later) action pairs among errors: {tot_pairs}'
      f'   distinct: {len(cnt)}')

# how concentrated?  count how many distinct pairs cover half the inversions
run, half = 0, tot_pairs / 2
need = 0
for _, c in cnt.most_common():
    run += c; need += 1
    if run >= half:
        break
print(f'distinct pairs needed to cover half of all inversions: {need} of {len(cnt)}')
print(f'  -> {"CONCENTRATED (learnable)" if need <= 8 else "DIFFUSE (per-session noise)"}')

print('\ntop 20 inverted pairs (truth: first precedes second; decoder said otherwise):')
for (a, b), c in cnt.most_common(20):
    rev = cnt.get((b, a), 0)
    print(f'  {c:3d}  {a:28s} BEFORE {b:28s}   (reverse direction also seen {rev}x)')

# is the same unordered pair inverted in both directions?  that is noise, not a bias.
und = collections.Counter()
for (a, b), c in cnt.items():
    und[tuple(sorted((a, b)))] += c
both = sum(1 for k in und if cnt.get(k, 0) and cnt.get((k[1], k[0]), 0))
print(f'\nunordered pairs inverted in BOTH directions across sessions: {both} of {len(und)}')
print('  (a pair inverted both ways carries no exploitable directional bias)')

print('\n--- accuracy vs how many distinct actions the block pools ---')
D['nact'] = [len(set(cache[r.qa]['opts'])) for r in D.itertuples()]
print(D.groupby('nact').correct.agg(n='size', ok='sum', acc='mean').round(3).to_string())

D.drop(columns=['inv']).to_csv(os.path.join(ROOT, 'seqlab', 'inversion_atlas.csv'), index=False)
pd.DataFrame([dict(earlier=a, later=b, n=c, reverse=cnt.get((b, a), 0))
              for (a, b), c in cnt.most_common()]).to_csv(
    os.path.join(ROOT, 'seqlab', 'inverted_pairs.csv'), index=False)
print('\nwrote seqlab/inversion_atlas.csv and seqlab/inverted_pairs.csv')
