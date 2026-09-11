"""Specialist B v3: per-frame thermal action scorer -> centroid order.

Trains multinomial logistic on HARn thermal frames (clean per-frame
labels: every frame = clip action; canonical dir -> atom), one model per
outer fold (fold users excluded). Applies to HAU sequence clips,
orders options by score-centroid time (pipeline aggregation, new
evidence). Compares vs pipeline on identical rows. Grouped folds match
oof_driver.
"""
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
sys.path.insert(0, os.path.join(ROOT, 'research', 'final_video_20260910'))
from pseudotest import folds  # noqa: E402
from presence import ATOMS, canon_to_atom  # noqa: E402

AID = {a: i for i, a in enumerate(ATOMS)}


def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def main():
    th = np.load(os.path.join(ROOT, 'champ', 'thermal_mnv3_frames.npz'),
                 allow_pickle=True)
    qmap = {k[:-2] for k in th.files if k.endswith('|X')}
    sg = np.load(os.path.join(ROOT, 'champ',
                              'thermal_mnv3_cropped_segments.npz'),
                 allow_pickle=True)
    # HARn frames: prefer full frame sequences when present, else segments
    harn = []  # (atom, user, vec(T,576))
    for key in sg.files:
        m = re.match(r'HARn/([^/]+)/([^/]+)/([^/]+)$', key)
        if not m:
            continue
        a = canon_to_atom(m.group(1))
        if a is None:
            continue
        harn.append((a, m.group(2),
                     np.asarray(sg[key], dtype=np.float64)))
    print('HARn segment clips:', len(harn), flush=True)
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'training_qa.csv'),
                                    encoding='utf-8-sig')))
    hau = [r for r in rows if r['path'].startswith('HAU/')]
    clips = defaultdict(dict)
    for r in hau:
        clips[r['path']][r['category']] = r

    def O(r):
        return [norm(r[k]) for k in 'ABCD']

    users = sorted(set(h[1] for h in harn)
                   | set(p.split('/')[1] for p in clips))
    recs = []
    for fi, hold in enumerate(folds(users, 5)):
        hold = set(hold)
        Xa, ya = [], []
        for a, u, v in harn:
            if u in hold:
                continue
            Xa.append(v.reshape(-1, 576)[::3])
            ya += [AID[a]] * len(Xa[-1])
        Xa = np.concatenate(Xa)
        ya = np.array(ya)
        sc = StandardScaler().fit(Xa)
        clf = LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced')
        clf.fit(sc.transform(Xa), ya)
        # frame-level diagnostic on held-out HARn clips
        Va, vya = [], []
        for a, u, v in harn:
            if u not in hold:
                continue
            Va.append(v.reshape(-1, 576)[::7])
            vya += [AID[a]] * len(Va[-1])
        if Va:
            Va = sc.transform(np.concatenate(Va))
            vya = np.array(vya)
            pa = clf.predict(Va)
            print(f'  fold {fi} HARn-frame acc={(pa == vya).mean():.4f} '
                  f'n={len(vya)}', flush=True)
        for p, c in clips.items():
            if p.split('/')[1] not in hold or 'sequence' not in c:
                continue
            r = c['sequence']
            o = O(r)
            base = p if p in qmap else None
            if base is None or not all(x in AID for x in set(o)):
                continue
            X = np.asarray(th[base + '|X'], dtype=np.float64)
            P = clf.predict_proba(sc.transform(X))
            T = X.shape[0]
            cls = list(clf.classes_)
            if any(AID[x] not in cls for x in set(o)):
                recs.append(dict(qa_id=r['qa_id'], fold=fi,
                                 category='sequence', pred=None,
                                 answer=r['answer'], correct=0))
                continue
            ci = [cls.index(AID[x]) for x in o]
            sub = P[:, ci]
            sub = sub / (sub.sum(1, keepdims=True) + 1e-9)
            cen = (sub * np.arange(T)[:, None]).sum(0)
            order = list(np.argsort(cen))
            pr = ''.join('ABCD'[order.index(k)] for k in range(4))
            recs.append(dict(qa_id=r['qa_id'], fold=fi, category='sequence',
                             pred=pr, answer=r['answer'],
                             correct=int(pr == r['answer'])))
        print(f'  fold {fi} done', flush=True)
    d = pd.DataFrame(recs)
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       'oof_orderlog_v1.csv')
    d.to_csv(out, index=False)
    print(f'scored {len(d)} {int(d["correct"].sum())}/{len(d)}='
          f'{d["correct"].mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
