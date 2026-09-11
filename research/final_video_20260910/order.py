"""Specialist B v1: segmental template order for sequence rows.

Per-action thermal centroids from STRONG single-action HARn clips
(outer-train users only). For each HAU sequence clip: split thermal frames
into 4 equal quarters, cosine-score quarters x option-actions, Hungarian
assign quarters to actions (one-to-one), order actions by quarter index,
emit option letters. Deterministic; no thresholds. Grouped folds match
oof_driver (pseudotest.folds(users, 5, seed=7)).
"""
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
sys.path.insert(0, os.path.join(ROOT, 'research', 'final_video_20260910'))
from pseudotest import folds  # noqa: E402
from presence import ATOMS, canon_to_atom  # noqa: E402


def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def main():
    th = np.load(os.path.join(ROOT, 'champ', 'thermal_mnv3_frames.npz'),
                 allow_pickle=True)
    # index clips
    qmap = {}  # qa-path -> key base
    for k in th.files:
        if k.endswith('|X'):
            qmap[k[:-2]] = k
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'training_qa.csv'),
                                    encoding='utf-8-sig')))
    hau = [r for r in rows if r['path'].startswith('HAU/')]
    clips = defaultdict(dict)
    for r in hau:
        clips[r['path']][r['category']] = r

    def O(r):
        return [norm(r[k]) for k in 'ABCD']

    # HARn strong clips from cropped thermal segments: (atom, user, meanvec)
    sg = np.load(os.path.join(ROOT, 'champ',
                              'thermal_mnv3_cropped_segments.npz'),
                 allow_pickle=True)
    harn = []
    for key in sg.files:
        m = re.match(r'HARn/([^/]+)/([^/]+)/([^/]+)$', key)
        if not m:
            continue
        a = canon_to_atom(m.group(1))
        if a is None:
            continue
        v = np.asarray(sg[key], dtype=np.float64).reshape(-1)
        harn.append((a, m.group(2), v))
    print('HARn thermal clips:', len(harn), flush=True)

    users = sorted(set(h[1] for h in harn)
                   | set(p.split('/')[1] for p in clips))
    aid = {a: i for i, a in enumerate(ATOMS)}
    recs = []
    for fi, hold in enumerate(folds(users, 5)):
        hold = set(hold)
        # centroids (576-d): outer-train HARn segments (w=3) + outer-train
        # HAU full-clip means attributed to each set member (w=1)
        acc = defaultdict(list)
        for a, u, v in harn:
            if u in hold:
                continue
            for _ in range(3):
                acc[a].append(v.reshape(-1, 576).mean(axis=0))
        for p, c in clips.items():
            if p.split('/')[1] in hold:
                continue
            base = p if p in qmap else None
            if base is None:
                continue
            s = set()
            for cat in ('multi', 'sequence'):
                if cat in c:
                    o = O(c[cat])
                    s.update(o['ABCD'.index(x)] for x in c[cat]['answer'])
            if not s:
                continue
            X = np.asarray(th[base + '|X'], dtype=np.float64)
            m = X.mean(axis=0)
            for a in s:
                if a in aid:
                    acc[a].append(m)
        cent = {a: np.stack(v).mean(axis=0) for a, v in acc.items() if v}
        cn = {a: v / (np.linalg.norm(v) + 1e-9) for a, v in cent.items()}
        nq = 0
        for p, c in clips.items():
            if p.split('/')[1] not in hold or 'sequence' not in c:
                continue
            base = p if p in qmap else None
            if base is None:
                continue
            r = c['sequence']
            o = O(r)
            acts = [o['ABCD'.index(x)] for x in r['answer']]
            if any(a not in cn for a in set(o)):
                # need all four option actions to have centroids
                recs.append(dict(qa_id=r['qa_id'], fold=fi,
                                 category='sequence', pred=None,
                                 answer=r['answer'], correct=0))
                continue
            X = np.asarray(th[base + '|X'], dtype=np.float64)
            T = X.shape[0]
            qs = [X[(T * i) // 4:(T * (i + 1)) // 4].mean(axis=0)
                  for i in range(4)]
            qn = [q / (np.linalg.norm(q) + 1e-9) for q in qs]
            S = np.array([[qn[i] @ cn[a] for a in o] for i in range(4)])
            ri, ci = linear_sum_assignment(-S)
            order = [None] * 4
            for i, j in zip(ri, ci):
                order[j] = i
            pred = ''.join('ABCD'[order.index(k)] for k in range(4))
            nq += 1
            recs.append(dict(qa_id=r['qa_id'], fold=fi, category='sequence',
                             pred=pred, answer=r['answer'],
                             correct=int(pred == r['answer'])))
        print(f'  fold {fi} scored {nq}', flush=True)
    d = pd.DataFrame(recs)
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       'oof_orderB_v1.csv')
    d.to_csv(out, index=False)
    g = d[d['pred'].notna()]
    print(f'scored {len(g)}/{len(d)} '
          f'{int(g["correct"].sum())}/{len(g)}={g["correct"].mean():.4f}',
          flush=True)


if __name__ == '__main__':
    main()
