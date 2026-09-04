import os, sys, itertools, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *

tr, te, meta = load_all()
s = seq_rows(tr).set_index('qa_id')
cache = pickle.load(open('seqlab/cache.pkl', 'rb'))
fold = {}
for f in range(5):
    for q in pd.read_csv(f'splits/fold_{f}_val.csv').qa_id: fold[q] = f

byblk = collections.defaultdict(list)
for k, v in cache.items(): byblk[v['blk']].append(k)

base = {k: v['cen'] for k, v in cache.items()}
joint = {}
for blk, ks in byblk.items():
    ks = sorted(ks)
    lab, bs, U = joint_decode([cache[k]['opts'] for k in ks], [cache[k]['dp'] for k in ks])
    for k, l in zip(ks, lab): joint[k] = l

rows = []
for k, v in cache.items():
    rows.append(dict(qa=k, fold=fold.get(k, -1), blk=str(v['blk']), nq=len(byblk[v['blk']]),
                     ans=v['ans'], base=base[k], joint=joint[k],
                     bc=base[k] == v['ans'], jc=joint[k] == v['ans']))
R = pd.DataFrame(rows)
print('=== joint(dp) vs champion centroid ===')
print(f"  base {R.bc.sum()}/{len(R)}   joint {R.jc.sum()}/{len(R)}   net {R.jc.sum()-R.bc.sum():+d}")
fl = R[R.base != R.joint]
wr = int(((~fl.bc) & fl.jc).sum()); rw = int((fl.bc & (~fl.jc)).sum()); ww = int(((~fl.bc) & (~fl.jc)).sum())
print(f"  flips {len(fl)}  W->R {wr}  R->W {rw}  W->W {ww}  precision(of decided) {wr/max(wr+rw,1):.3f}  net/100 {100*(wr-rw)/len(fl):.1f}")
print('\n  per fold:')
for f in sorted(R.fold.unique()):
    d = R[R.fold == f]
    fd = d[d.base != d.joint]
    w = int(((~fd.bc) & fd.jc).sum()); l = int((fd.bc & (~fd.jc)).sum())
    print(f"    fold {f}: base {d.bc.sum():3d}/{len(d):3d}  joint {d.jc.sum():3d}  net {d.jc.sum()-d.bc.sum():+3d}   flips {len(fd)} W->R {w} R->W {l}")
print('\n  by block size (nq):')
for n in sorted(R.nq.unique()):
    d = R[R.nq == n]
    print(f"    nq={n}: n={len(d):3d}  base {d.bc.mean():.3f}  joint {d.jc.mean():.3f}  net {d.jc.sum()-d.bc.sum():+d}")
print('\n  BLOCK-level exact (all questions in block right):')
gb = R.groupby('blk').agg(nb=('qa','size'), b=('bc','sum'), j=('jc','sum'))
print(f"    base all-right blocks {int((gb.b==gb.nb).sum())}/{len(gb)}   joint {int((gb.j==gb.nb).sum())}/{len(gb)}")
R.to_csv('seqlab/audit_joint.csv', index=False)
