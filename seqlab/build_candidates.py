"""Build sequence-corrected candidates over the 0.92105 champion.

Mechanism S: the sequence questions of one session block are decoded JOINTLY as restrictions
of a single latent total order over their option union.  Justification and validation:
  * the latent-script property is exact on training data: 104/104 session triples are
    pairwise-order-conflict-free over 1848 observed pairs;
  * every one of the 39 test sequence questions sits in a block whose siblings also carry a
    sequence question (9 triples + 6 pairs), so the constraint binds everywhere;
  * evidence = cross-clip pairwise order from the dense model (two temperatures, presence
    weighted) + a pairwise-order model learned on the 2927 nested HARn segment onsets, a
    supervision source the champion used only as frame labels;
  * 5-fold subject-disjoint OOF: 172/305 -> 225/305, 112 flips, 64 W->R / 11 R->W
    (precision 0.853), positive in all five folds; nested-CV weight selection gives 209/305.

Variant A = every flip.  Variant B = flips only inside blocks where the champion's own
predictions are mutually INCONSISTENT with any single total order, which is a label-free
proof that it errs there (OOF: champion 0.370 in such blocks vs 0.793 elsewhere;
91 flips at 0.883 precision).
"""
import os, sys, json, itertools, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'champ'); sys.path.insert(0, 'seqlab')
from core import load_all

tr, te, meta = load_all()
A = pd.read_csv('seqlab/test_sequence_audit.csv')
champ = pd.read_csv('submission_092105_SUBMITTED.csv')
assert len(champ) == 682 and champ.prediction.notna().all()
teq = te.set_index('qa_id')

def consistent(po):
    U = sorted(set().union(*[set(o) for o, _ in po]))
    for perm in itertools.permutations(U):
        pos = {x: i for i, x in enumerate(perm)}
        if all(''.join(sorted('ABCD', key=lambda L: pos[o[ord(L) - 65]])) == p for o, p in po):
            return True
    return False

cons = {}
for b, g in A.groupby('block'):
    po = [([teq.loc[q, L] for L in 'ABCD'], c) for q, c in zip(g.qa_id, g.champ)]
    cons[b] = consistent(po)
A['champ_block_consistent'] = A.block.map(cons)

for tag, mask in [('A_all', A.flip),
                  ('B_gated', A.flip & (~A.champ_block_consistent))]:
    sub = champ.copy().set_index('qa_id')
    ch = A[mask]
    for r in ch.itertuples(): sub.loc[r.qa_id, 'prediction'] = r.pred
    sub = sub.reset_index()
    assert len(sub) == 682 and sub.prediction.notna().all()
    assert (sub.qa_id.values == champ.qa_id.values).all()
    n = int((sub.prediction.values != champ.prediction.values).sum())
    fn = f'submission_seqjoint_{tag}.csv'
    sub.to_csv(fn, index=False)
    print(f'{fn:34s} changes vs champion: {n}')

A[['qa_id','clip','block','champ','pred','flip','champ_block_consistent','margin','latent_order']]\
 .to_csv('seqlab/test_sequence_audit.csv', index=False)
print('\n=== per-block audit ===')
for b, g in A.groupby('block'):
    print(f'  block {b}  champ_consistent={bool(g.champ_block_consistent.iloc[0])}  '
          f'flips={int(g.flip.sum())}/{len(g)}')
    print(f'    latent order: {g.latent_order.iloc[0]}')
print('\n=== the flips ===')
print(A[A.flip][['qa_id','clip','champ','pred','champ_block_consistent','margin']].to_string(index=False))
print('\nflips in inconsistent blocks: %d ; in consistent blocks: %d'
      % (int((A.flip & ~A.champ_block_consistent).sum()), int((A.flip & A.champ_block_consistent).sum())))
