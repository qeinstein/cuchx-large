"""Derive the HARn-folder <-> QA-option-string vocabulary maps from training data.

Two maps:
  HARN2HAU  : 'HARn/<folder>' action -> HAU action-option string (multi/single/seq/comb vocab)
  HARN2SELF : folder -> HARn single-question answer string
Both are derived, then verified, from training labels only (they are label-derived constants,
not per-clip information, so they are safe to use at test time).
"""
import os, sys, json
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build(tr, meta):
    # ---- HARn folder -> its own single-question answer
    n = tr[(tr.source == 'HARn') & (tr.category == 'single')]
    m2 = defaultdict(Counter)
    for _, r in n.iterrows():
        m2[r.harn_action][str(r[str(r['answer'])]).strip()] += 1
    HARN2SELF = {k: v.most_common(1)[0][0] for k, v in m2.items()}
    purity = np.mean([v.most_common(1)[0][1] / sum(v.values()) for v in m2.values()])

    # ---- session pools from HAU action questions
    h = tr[(tr.source == 'HAU') & tr.category.isin(['single', 'multi', 'combination', 'sequence'])]
    pool = defaultdict(set)
    for _, r in h.iterrows():
        for L in str(r['answer']):
            if L in 'ABCD':
                for p in str(r[L]).split(','):
                    pool[(r.user, r.aa, r.bb)].add(p.strip())
    # ---- HARn segments per session, from the modality package
    segs = defaultdict(set)
    for r in meta[meta.kind == 'train_harn'].itertuples():
        a, b = r.trial.split('-')[0], r.trial.split('-')[1]
        segs[(r.user, a, b)].add(r.action)

    folders = sorted({a for v in segs.values() for a in v})
    options = sorted({o for v in pool.values() for o in v})
    C = np.zeros((len(folders), len(options)))
    for k in set(segs) & set(pool):
        for a in segs[k]:
            for o in pool[k]:
                C[folders.index(a), options.index(o)] += 1
    # normalise to a Jaccard-like score then solve the assignment
    fa = C.sum(1, keepdims=True) + 1e-9
    oa = C.sum(0, keepdims=True) + 1e-9
    S = C / np.sqrt(fa * oa)
    ri, ci = linear_sum_assignment(-S)
    HARN2HAU = {folders[i]: options[j] for i, j in zip(ri, ci)}
    score = {folders[i]: float(S[i, j]) for i, j in zip(ri, ci)}
    return HARN2HAU, HARN2SELF, purity, score, folders, options


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core import load_all
    tr, te, meta = load_all()
    H2H, H2S, purity, score, folders, options = build(tr, meta)
    print('HARn single answer purity: %.4f' % purity)
    print('unmatched HAU options:', sorted(set(options) - set(H2H.values())))
    print('\n%-42s %-40s %s' % ('HARn folder', 'HAU option', 'score'))
    for f in folders:
        print('%-42s %-40s %.3f' % (f, H2H[f], score[f]))
    json.dump(dict(HARN2HAU=H2H, HARN2SELF=H2S),
              open(os.path.join(ROOT, 'champ', 'vocab.json'), 'w'), indent=1)
