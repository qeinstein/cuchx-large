"""Nested-HARn order constraints, counted per SESSION BLOCK rather than per clip.

The latent action order is a property of the session, not of a single clip (all 104 training
session triples are pairwise order-conflict-free over 1848 observed pairs), so an order
observation harvested from ANY clip in a block constrains every sequence question in that
block.  `slotlab/nested_order_coverage.py` counted per parent clip and found the coverage
thin; this counts at the unit the constraint actually applies to.

A usable constraint needs two nested HARn children, in the same block, whose actions are
distinct, both in the block's sequence option universe, and whose frame intervals are
disjoint enough to order.  Children of the same clip order directly by frame index; children
of different clips in the same block order via the shared latent order only if their actions
differ.
"""
import os, sys, json, itertools
from collections import defaultdict, Counter
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all, real_test_view, fit_group_model, infer_blocks, opts, gt_letters
import harn as H
from nested_order_coverage import build_nest


def sec(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78, flush=True)


def main():
    tr, te, meta = load_all()
    mrow = meta.set_index('qa_path')
    S2A = H.S2A
    V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']

    # ---------------------------------------------------------------- TRAIN
    sec('TRAIN: constraints per true session')
    nest = build_nest(meta, 'train')
    single = tr[(tr.category == 'single') & (tr.source == 'HARn')]
    # action label of each HARn clip, from its own single question's answer
    act_of = {}
    for r in single.itertuples():
        L = [c for c in str(r.answer) if c in 'ABCD']
        if not L:
            continue
        a = str(getattr(r, L[0])).strip()
        act_of[r.path] = V.get(S2A.get(a, a), S2A.get(a, a))

    hau = tr[(tr.source == 'HAU')]
    sess_of = {}
    for (u, a, b), g in hau.groupby(['user', 'aa', 'bb']):
        for p in g.path.unique():
            sess_of[p] = (u, a, b)

    seq = tr[tr.category == 'sequence']
    sess_seq = defaultdict(set)
    for r in seq.itertuples():
        if r.path in sess_of:
            sess_seq[sess_of[r.path]].update(opts(r))

    # children grouped by session
    kids = defaultdict(list)
    for hp, parent in nest.items():
        if parent in sess_of and hp in act_of:
            f0 = mrow.loc[hp, 'f0'] if hp in mrow.index else np.nan
            kids[sess_of[parent]].append((parent, act_of[hp], f0))

    nc, usable, npair = Counter(), 0, 0
    for s, opt in sess_seq.items():
        ch = [(p, a, f) for p, a, f in kids.get(s, []) if a in opt]
        nc[len(ch)] += 1
        acts = {a for _, a, _ in ch}
        if len(acts) >= 2:
            usable += 1
            npair += len(list(itertools.combinations(acts, 2)))
    print(f'sessions carrying a sequence question: {len(sess_seq)}')
    print(f'in-option nested children per such session: {dict(sorted(nc.items()))}')
    print(f'sessions with >=2 DISTINCT in-option child actions: {usable}/{len(sess_seq)} '
          f'= {usable/max(len(sess_seq),1):.3f}')
    print(f'distinct orderable action pairs harvested: {npair}')
    print(f'  (compare: per-CLIP counting gave 32/308 usable parents)')

    # ---------------------------------------------------------------- TEST
    sec('TEST: constraints per inferred block')
    vis = real_test_view(te)
    blocks = infer_blocks(vis, fit_group_model(tr))
    pathof = dict(zip(vis.idx, vis.true_path))
    idx_of = dict(zip(vis.true_path, vis.idx))
    tnest = build_nest(meta, 'test')
    blk_of = {}
    for bi, b in enumerate(blocks):
        for i in b:
            blk_of[pathof[int(i)]] = bi

    seqt = vis[vis.category == 'sequence']
    blk_seq = defaultdict(set)
    for r in seqt.itertuples():
        if r.true_path in blk_of:
            blk_seq[blk_of[r.true_path]].update(opts(r))
    print(f'inferred blocks carrying a sequence question: {len(blk_seq)}')
    print(f'sequence questions: {len(seqt)}')

    tkids = defaultdict(list)
    for hp, parent in tnest.items():
        if parent in blk_of:
            tkids[blk_of[parent]].append((parent, hp))
    nck = Counter(len(tkids.get(b, [])) for b in blk_seq)
    print(f'nested children per sequence-bearing block: {dict(sorted(nck.items()))}')
    ge2 = sum(1 for b in blk_seq if len(tkids.get(b, [])) >= 2)
    print(f'blocks with >=2 nested children: {ge2}/{len(blk_seq)}')
    nq = sum(1 for r in seqt.itertuples()
             if blk_of.get(r.true_path) is not None
             and len(tkids.get(blk_of[r.true_path], [])) >= 2)
    print(f'sequence QUESTIONS in such blocks: {nq}/{len(seqt)}  '
          f'(~{nq/2:.0f} of them on the public half)')

    rows = []
    for b in sorted(blk_seq):
        ch = tkids.get(b, [])
        rows.append(dict(block=str([int(x) for x in blocks[b]]), n_children=len(ch),
                         n_seq_q=sum(1 for r in seqt.itertuples()
                                     if blk_of.get(r.true_path) == b),
                         children=';'.join(h for _, h in ch)))
    D = pd.DataFrame(rows)
    print()
    print(D.to_string(index=False))
    D.to_csv(os.path.join(ROOT, 'slotlab', 'test_seq_block_children.csv'), index=False)
    print('\nwrote slotlab/test_seq_block_children.csv')


if __name__ == '__main__':
    main()
