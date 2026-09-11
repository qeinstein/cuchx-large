"""Specialist C v1: thermal object classifier for object_interaction rows.

Trains 13-way logistic on HARn object-clip thermal means (cropped
segments), leave-user-out grouped folds matching oof_driver. Decodes all
133 HARn object OOF rows by argmax over options. Compares vs pipeline.
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


def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def main():
    sg = np.load(os.path.join(ROOT, 'champ',
                              'thermal_mnv3_cropped_segments.npz'),
                 allow_pickle=True)
    feat = {}
    for key in sg.files:
        m = re.match(r'HARn/([^/]+)/([^/]+)/([^/]+)$', key)
        if not m:
            continue
        v = np.asarray(sg[key], dtype=np.float64).reshape(-1, 576)
        feat['HARn/%s/%s/%s' % (m.group(1), m.group(2), m.group(3))] = \
            v.mean(axis=0)
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'training_qa.csv'),
                                    encoding='utf-8-sig')))
    obj = [r for r in rows if r['source'] == 'HARn'
           and r['category'] == 'object_interaction']
    users = sorted(set(r['path'].split('/')[2] for r in obj))
    recs = []
    for fi, hold in enumerate(folds(users, 5)):
        hold = set(hold)
        Xa, ya, paths = [], [], []
        for r in obj:
            if r['path'].split('/')[2] in hold or r['path'] not in feat:
                continue
            Xa.append(feat[r['path']])
            o = [norm(r[k]) for k in 'ABCD']
            ya.append(o['ABCD'.index(r['answer'])])
            paths.append(r['path'])
        Xa = np.stack(Xa)
        sc = StandardScaler().fit(Xa)
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(sc.transform(Xa), np.array(ya))
        for r in obj:
            if r['path'].split('/')[2] not in hold or r['path'] not in feat:
                continue
            o = [norm(r[k]) for k in 'ABCD']
            P = clf.predict_proba(sc.transform(feat[r['path']].reshape(1, -1)))[0]
            order = np.argsort([P[list(clf.classes_).index(x)] if x in list(clf.classes_) else -1 for x in o])
            pr = 'ABCD'[int(order[-1])]
            recs.append(dict(qa_id=r['qa_id'], fold=fi,
                             category='object_interaction', pred=pr,
                             answer=r['answer'],
                             correct=int(pr == r['answer'])))
        print(f'  fold {fi} done', flush=True)
    d = pd.DataFrame(recs)
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       'oof_objectC_v1.csv')
    d.to_csv(out, index=False)
    print(f'scored {len(d)} {int(d["correct"].sum())}/{len(d)}='
          f'{d["correct"].mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
