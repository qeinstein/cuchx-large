"""Validate Mechanism S in the two-question-block regime.

The OOF set has 300 sequence questions in 3-question blocks and only 4 in 2-question blocks,
but 8 of the 17 test flips fall in two-clip blocks.  So simulate that regime: for every
training triple, drop one clip and re-decode from the remaining two questions only.
"""
import os, sys, itertools, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *
from fast import Block, PERMS

R = pickle.load(open('seqlab/res11.pkl', 'rb')); EV = R['EV']; KS = R['KS']
cache = pickle.load(open('seqlab/cache.pkl', 'rb'))
fold = {}
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q] = f
W = (4.0, 4.0, 1.0)

def decode(blk, ks):
    """Joint decode using only the questions in ks; evidence matrices are block-level."""
    qo = [cache[k]['opts'] for k in ks]
    U = sorted(set().union(*[set(o) for o in qo]))
    B = Block(U, qo)
    e = EV[blk]
    t = B.score([(W[0], e['h05']), (W[1], e['h10']), (W[2], e['learn'])],
                [(0.0, np.zeros(24)) for _ in ks])
    k0 = int(np.argmax(t))
    return {k: PERMS[int(L[k0])] for k, L in zip(ks, B.L)}

rows = []
for blk, ks in KS.items():
    ks = sorted(ks)
    if len(ks) != 3:
        continue
    for drop in range(3):
        keep = [k for i, k in enumerate(ks) if i != drop]
        pr = decode(blk, keep)
        for k in keep:
            rows.append(dict(qa=k, fold=fold.get(k, -1), blk=str(blk), regime='pair',
                             ans=cache[k]['ans'], champ=cache[k]['cen'], new=pr[k]))
    pr3 = decode(blk, ks)
    for k in ks:
        rows.append(dict(qa=k, fold=fold.get(k, -1), blk=str(blk), regime='triple',
                         ans=cache[k]['ans'], champ=cache[k]['cen'], new=pr3[k]))
A = pd.DataFrame(rows)
A['cc'] = A.champ == A.ans; A['nc'] = A.new == A.ans
print('=== Mechanism S by block regime (pair regime simulated by dropping one clip) ===')
for reg, g in A.groupby('regime'):
    fl = g[g.champ != g.new]
    wr = int(((~fl.cc) & fl.nc).sum()); rw = int((fl.cc & (~fl.nc)).sum())
    print(f'  {reg:7s} n={len(g):4d}  champion {g.cc.sum():4d} ({g.cc.mean():.4f})  '
          f'S {g.nc.sum():4d} ({g.nc.mean():.4f})  delta {g.nc.mean()-g.cc.mean():+.4f}')
    print(f'          flips {len(fl):4d}  W->R {wr:3d}  R->W {rw:3d}  W->W {len(fl)-wr-rw:3d}  '
          f'precision {wr/max(wr+rw,1):.3f}  net/100 {100*(wr-rw)/max(len(fl),1):.1f}')
print('\n  per fold, PAIR regime:')
p = A[A.regime == 'pair']
for f in sorted(p.fold.unique()):
    d = p[p.fold == f]; fd = d[d.champ != d.new]
    w = int(((~fd.cc) & fd.nc).sum()); l = int((fd.cc & (~fd.nc)).sum())
    print(f'    fold {f}: champ {d.cc.sum():4d}/{len(d):4d} S {d.nc.sum():4d}  '
          f'net {d.nc.sum()-d.cc.sum():+4d}  flips {len(fd):3d} W->R {w:3d} R->W {l:3d} '
          f'prec {w/max(w+l,1):.2f}')
A.to_csv('seqlab/audit_regime.csv', index=False)
