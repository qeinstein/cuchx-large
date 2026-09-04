import os, sys, itertools, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqcore import *

tr, te, meta = load_all()
s = seq_rows(tr)
print('sequence rows', len(s), 'blocks', s.blk.nunique())

cache = {}
skip = 0
for _, r in s.iterrows():
    lp = DC.logits('oof', r.path)
    oo = opts(r)
    if lp is None or not all(o in OPT2CLS for o in oo):
        skip += 1; continue
    ci = [OPT2CLS[o] for o in oo]
    cache[r.qa_id] = dict(blk=r.blk, opts=oo, ci=ci, ans=str(r['answer']),
                          dp=dp_scores(lp, ci),
                          pw=perm_score_from_pair(pairwise_from_logits(lp, ci)),
                          cen=centroid_perm(lp, ci))
print('cached', len(cache), 'skipped', skip)
np.save('seqlab/cache_keys.npy', np.array(list(cache.keys())))
import pickle; pickle.dump(cache, open('seqlab/cache.pkl','wb'))

def report(name, preds):
    ids = [k for k in cache if k in preds]
    ok = sum(preds[k] == cache[k]['ans'] for k in ids)
    print(f'  {name:38s} {ok:3d}/{len(ids)} = {ok/len(ids):.4f}')
    return ok

print('\n--- INDEPENDENT decoders (308 sequence Q, OOF dense logits) ---')
report('champion: centroid argsort', {k: v['cen'] for k, v in cache.items()})
report('monotone DP argmax', {k: PERMS[int(np.argmax(v['dp']))] for k, v in cache.items()})
report('time-marginal pairwise argmax', {k: PERMS[int(np.argmax(v['pw']))] for k, v in cache.items()})

print('\n--- JOINT block decoding (single latent total order per session) ---')
byblk = collections.defaultdict(list)
for k, v in cache.items(): byblk[v['blk']].append(k)
for sname in ['dp', 'pw']:
    preds = {}
    for blk, ks in byblk.items():
        ks = sorted(ks)
        lab, _, U = joint_decode([cache[k]['opts'] for k in ks], [cache[k][sname] for k in ks])
        for k, l in zip(ks, lab): preds[k] = l
    report(f'joint({sname})', preds)
# dp+pw sum
for w in [0.5, 1.0, 2.0]:
    preds = {}
    for blk, ks in byblk.items():
        ks = sorted(ks)
        sv = [cache[k]['dp'] / 100.0 + w * cache[k]['pw'] for k in ks]
        lab, _, U = joint_decode([cache[k]['opts'] for k in ks], sv)
        for k, l in zip(ks, lab): preds[k] = l
    report(f'joint(dp/100 + {w}*pw)', preds)
