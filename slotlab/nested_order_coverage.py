"""Can answered HARn single questions supply exact order constraints for sequence questions?

A HARn clip is a labelled temporal SEGMENT of a parent HAU clip: `champ/pipeline.py` recovers
the nesting from timestamps and global frame indices alone, and HARn `single` accuracy is
0.9580 held-out.  So every confidently answered HARn single question is effectively a direct
observation "action X occupies frames [f0,f1] of parent clip P" -- and two such observations
in the same parent give a pairwise order constraint on that session's latent action order at
near-certainty, which is exactly the supervision the sequence decoder is short of.

`seqlab/run17.py` attempted this and reported `0 observations over 0 parent clips`, which for
2905 uniquely nested train clips and 429 QA-referenced HARn singles is not a plausible
negative.  This script measures the coverage that mechanism can actually have, on both the
training split (where it can be scored) and the real test split (where it would be applied),
before any decoder work is done.
"""
import os, sys, itertools, json
from collections import defaultdict, Counter
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all, real_test_view, opts, gt_letters
import harn as H


def build_nest(meta, scope):
    """Replicates pipeline.caches()' nesting for one scope -> {harn_path: hau_path}."""
    hk, nk = ('train_hau', 'train_harn') if scope == 'train' else ('test', 'test')
    hl = [(r.qa_path, r.t0, r.t1, r.f0, r.f1) for r in meta[meta.kind == hk].itertuples()
          if np.isfinite(r.t0)]
    if scope == 'test':
        hl = [x for x in hl if int(x[0].split('_')[-1]) >= 65]
        cand = [r for r in meta[meta.kind == nk].itertuples()
                if np.isfinite(r.t0) and int(r.qa_path.split('_')[-1]) < 65]
    else:
        cand = [r for r in meta[meta.kind == nk].itertuples() if np.isfinite(r.t0)]
    nest = {}
    for r in cand:
        c = [h for h in hl if h[1] <= r.t0 + 0.5 and h[2] >= r.t1 - 0.5
             and h[3] <= r.f0 and h[4] >= r.f1 and h[0] != r.qa_path]
        if len(c) == 1:
            nest[r.qa_path] = c[0][0]
    return nest


def sec(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78, flush=True)


def main():
    tr, te, meta = load_all()
    mrow = meta.set_index('qa_path')
    S2A = H.S2A
    V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']

    for scope in ('train', 'test'):
        sec(f'{scope} split')
        nest = build_nest(meta, scope)
        print(f'uniquely nested HARn clips: {len(nest)}')

        if scope == 'train':
            q = tr.copy()
            q['cpath'] = q.path
        else:
            q = real_test_view(te)
            q['cpath'] = q.true_path

        # HARn single questions, keyed by the clip they are about
        hs = q[(q.category == 'single') & (q.cpath.isin(nest.keys()))]
        print(f'HARn single questions on a nested clip: {len(hs)}')

        # sequence questions, keyed by parent HAU clip
        sq = q[q.category == 'sequence']
        print(f'sequence questions: {len(sq)}  on {sq.cpath.nunique()} distinct clips')

        # children per sequence-bearing parent
        kids = defaultdict(list)
        for r in hs.itertuples():
            kids[nest[r.cpath]].append(r.cpath)
        seq_parents = set(sq.cpath.unique())
        nk = Counter(len(kids.get(p, [])) for p in seq_parents)
        print(f'\nHARn children per sequence-bearing parent clip: {dict(sorted(nk.items()))}')
        cov = sum(1 for p in seq_parents if len(kids.get(p, [])) >= 2)
        print(f'parents with >=2 nested children (an order constraint is possible): '
              f'{cov}/{len(seq_parents)}')
        npair = sum(len(kids.get(p, [])) * (len(kids.get(p, [])) - 1) // 2 for p in seq_parents)
        print(f'orderable child pairs on sequence-bearing parents: {npair}')

        # do the children's actions actually appear among the sequence options?
        if scope == 'train':
            optsets = {}
            for r in sq.itertuples():
                optsets.setdefault(r.cpath, set()).update(opts(r))
            hit = tot = 0
            usable = 0
            for p in seq_parents:
                ch = kids.get(p, [])
                acts = []
                for c in ch:
                    row = tr[(tr.path == c) & (tr.category == 'single')]
                    if not len(row):
                        continue
                    rr = row.iloc[0]
                    L = gt_letters(rr)
                    if not L:
                        continue
                    a = str(rr[L[0]]).strip()
                    acts.append(S2A.get(a, a))
                mapped = [V.get(a, a) for a in acts]
                inopt = [m for m in mapped if m in optsets.get(p, set())]
                tot += len(mapped); hit += len(inopt)
                if len(set(inopt)) >= 2:
                    usable += 1
            print(f'\nchild actions that appear among their parent\'s sequence options: '
                  f'{hit}/{tot}')
            print(f'parents with >=2 DISTINCT in-option child actions '
                  f'(a usable constraint): {usable}/{len(seq_parents)}')


if __name__ == '__main__':
    main()
