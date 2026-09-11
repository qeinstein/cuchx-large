"""Sequence fusion experiment: dense-centroid order + thermal-quarter order.

Replicates pipeline seq (dense temporal centroid over 4 options + BG),
adds independent thermal-quarter Hungarian order (HAU-augmented centroids,
outer-train only), fuses by equal-weight mean time (pre-registered, no
tuning; ties -> dense order). Also reports peak-density variant.
Grouped folds match oof_driver. Compares vs pipeline on identical rows.
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
import pipeline as P  # noqa: E402
from presence import ATOMS, canon_to_atom  # noqa: E402


def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def main():
    base = pd.read_csv(os.path.join(ROOT, 'research', 'final_video_20260910',
                                    'oof_pipeline_base.csv'), dtype=str)
    PIPESEQ = dict(zip(base[base.category == 'sequence']['qa_id'],
                       base[base.category == 'sequence']['pred']))
    lg = np.load(os.path.join(ROOT, 'champ',
                              os.environ.get('CHAMP_LOGITS',
                                             'dense_logits_screen1_bucket.npz')),
                 allow_pickle=True)
    th = np.load(os.path.join(ROOT, 'champ', 'thermal_mnv3_frames.npz'),
                 allow_pickle=True)
    qmap = {k[:-2] for k in th.files if k.endswith('|X')}
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
        harn.append((a, m.group(2),
                     np.asarray(sg[key], dtype=np.float64)))
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
        acc = defaultdict(list)
        for a, u, v in harn:
            if u in hold:
                continue
            for _ in range(3):
                acc[a].append(v.reshape(-1, 576).mean(axis=0))
        for p, c in clips.items():
            if p.split('/')[1] in hold or p not in qmap:
                continue
            s = set()
            for cat in ('multi', 'sequence'):
                if cat in c:
                    o = O(c[cat])
                    s.update(o['ABCD'.index(x)] for x in c[cat]['answer'])
            X = np.asarray(th[p + '|X'], dtype=np.float64)
            m = X.mean(axis=0)
            for a in s:
                acc[a].append(m)
        cent = {a: np.stack(v).mean(axis=0) for a, v in acc.items() if v}
        cn = {a: v / (np.linalg.norm(v) + 1e-9) for a, v in cent.items()}
        for p, c in clips.items():
            if p.split('/')[1] not in hold or 'sequence' not in c:
                continue
            r = c['sequence']
            o = O(r)
            oraw = [r[k] for k in 'ABCD']
            key = f'oof|{p}'
            if key not in lg.files or p not in qmap:
                for tag in ('dense', 'fused', 'peak'):
                    recs.append(dict(qa_id=r['qa_id'], fold=fi, variant=tag,
                                     category='sequence', pred=None,
                                     answer=r['answer'], correct=0))
                continue
            if not all(x in cn for x in set(o)):
                for tag in ('dense', 'fused', 'peak'):
                    recs.append(dict(qa_id=r['qa_id'], fold=fi, variant=tag,
                                     category='sequence', pred=None,
                                     answer=r['answer'], correct=0))
                continue
            if any(x not in P.OPT2CLS for x in oraw):
                for tag in ('dense', 'fused', 'peak'):
                    recs.append(dict(qa_id=r['qa_id'], fold=fi, variant=tag,
                                     category='sequence', pred=None,
                                     answer=r['answer'], correct=0))
                continue
            # pipeline letters (copied from baseline OOF: no replication)
            _po = PIPESEQ.get(r['qa_id'], '')
            po = '' if (not isinstance(_po, str) or _po == 'nan') else _po
            # thermal quarters
            X = np.asarray(th[p + '|X'], dtype=np.float64)
            Tt = X.shape[0]
            qs = [X[(Tt * i) // 4:(Tt * (i + 1)) // 4].mean(axis=0)
                  for i in range(4)]
            qn = [q / (np.linalg.norm(q) + 1e-9) for q in qs]
            S = np.array([[qn[i] @ cn[a] for a in o] for i in range(4)])
            ri, ci2 = linear_sum_assignment(-S)
            qof = [None] * 4
            for i, j in zip(ri, ci2):
                qof[j] = i
            # Convention: order = option indices in temporal order.
            # po letters are already in temporal order.
            order_pipe = (['ABCD'.index(L) for L in po]
                          if po else None)
            pos_pipe = None
            if order_pipe is not None:
                pos_pipe = [0] * 4
                for t, j in enumerate(order_pipe):
                    pos_pipe[j] = t
            pos_th = list(qof)
            order_th = sorted(range(4), key=lambda k: pos_th[k])
            outs = [('thermal', order_th)]
            if pos_pipe is not None:
                outs.append(('dense', order_pipe))
                borda = [pos_pipe[k] + pos_th[k] for k in range(4)]
                order_f = sorted(range(4),
                                 key=lambda k: (borda[k], pos_pipe[k]))
                outs.append(('fused', order_f))
            else:
                outs.append(('fused', order_th))
            for tag, order in outs:
                pr = ''.join('ABCD'[j] for j in order)
                recs.append(dict(qa_id=r['qa_id'], fold=fi, variant=tag,
                                 category='sequence', pred=pr,
                                 answer=r['answer'],
                                 correct=int(pr == r['answer'])))
        print(f'  fold {fi} done', flush=True)
    d = pd.DataFrame(recs)
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       'oof_seqfuse_v1.csv')
    d.to_csv(out, index=False)
    for tag, g in d.groupby('variant'):
        g = g[g['pred'].notna()]
        print(f'  {tag:8s} {int(g["correct"].sum())}/{len(g)}='
              f'{g["correct"].mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
