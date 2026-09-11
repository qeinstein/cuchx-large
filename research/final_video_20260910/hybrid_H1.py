"""Hybrid rule H1: object override from single-action mapping (dual clips).

For HARn clips with both single and object_interaction rows: map the
pipeline's single-predicted action to an object via outer-fold-only
sibling-single/object co-occurrence. Override the object prediction iff
the mapped object text appears among the options (else abstain = keep
pipeline). Writes full OOF CSV = pipeline preds + overrides.
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def main():
    base = pd.read_csv(os.path.join(ROOT, 'research', 'final_video_20260910',
                                    'oof_pipeline_base.csv'), dtype=str)
    base['correct'] = base['correct'].astype(int)
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'training_qa.csv'),
                                    encoding='utf-8-sig')))
    har = [r for r in rows if r['source'] == 'HARn']
    byclip = defaultdict(dict)
    for r in har:
        byclip[r['path']][r['category']] = r

    def O(r):
        return [norm(r[k]) for k in 'ABCD']

    bmap = dict(zip(base['qa_id'], zip(base['pred'], base['fold'])))
    out = base.copy()
    n_over = n_abstain = 0
    for p, d in byclip.items():
        if 'object_interaction' not in d or 'single' not in d:
            continue
        o, s = d['object_interaction'], d['single']
        if o['qa_id'] not in bmap or s['qa_id'] not in bmap:
            continue
        spred, _ = bmap[s['qa_id']]
        f = bmap[o['qa_id']][1]
        if not isinstance(spred, str) or spred == 'nan':
            continue
        sact = O(s)['ABCD'.index(spred)]
        votes = Counter()
        for r in har:
            if r['category'] != 'object_interaction':
                continue
            if r['qa_id'] not in bmap or bmap[r['qa_id']][1] == f:
                continue
            sib = byclip[r['path']].get('single')
            if sib is None:
                continue
            so = O(sib)
            if so['ABCD'.index(sib['answer'])] == sact:
                oo = O(r)
                votes[oo['ABCD'.index(r['answer'])]] += 1
        if not votes:
            continue
        obj = votes.most_common(1)[0][0]
        oo = O(o)
        if obj not in oo:
            n_abstain += 1
            continue
        new = 'ABCD'[oo.index(obj)]
        out.loc[out['qa_id'] == o['qa_id'], 'pred'] = new
        out.loc[out['qa_id'] == o['qa_id'], 'correct'] = int(new == o['answer'])
        n_over += 1
    print(f'overrides {n_over} abstains {n_abstain}', flush=True)
    outp = os.path.join(ROOT, 'research', 'final_video_20260910',
                        'oof_hybrid_H1.csv')
    out.to_csv(outp, index=False)
    print('wrote', outp, flush=True)


if __name__ == '__main__':
    main()
