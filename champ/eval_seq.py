import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, opts
import dense as D, decode as DC

V = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'champ', 'vocab.json')))['HARN2HAU']
OPT2CLS = {V[a]: D.A2I[a] for a in D.ACTIONS}

tr, te, meta = load_all()
seq = tr[(tr.source == 'HAU') & (tr.category == 'sequence')]
n = c_dp = c_cen = 0
miss = 0
for _, r in seq.iterrows():
    lp = DC.logits('oof', r.path)
    if lp is None:
        miss += 1
        continue
    oo = opts(r)
    if not all(o in OPT2CLS for o in oo):
        miss += 1
        continue
    ci = [OPT2CLS[o] for o in oo]
    perm, sc, _ = DC.seq_order_dp(lp, ci)
    pred = ''.join('ABCD'[p] for p in perm)
    st = DC.presence_stats(lp)
    cen = sorted(range(4), key=lambda i: st['centroid'][ci[i]])
    pred_c = ''.join('ABCD'[i] for i in cen)
    n += 1
    c_dp += int(pred == str(r['answer']))
    c_cen += int(pred_c == str(r['answer']))
print(f'SEQUENCE  n={n} (skipped {miss})')
print(f'  monotone permutation DP : {c_dp}/{n} = {c_dp/n:.4f}')
print(f'  centroid sort           : {c_cen}/{n} = {c_cen/n:.4f}')
print(f'  v7 baseline             : 148/308 = 0.4805')
