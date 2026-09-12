"""Fit the PU action model on each training fold and audit stable test flips."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, real_test_view
from pseudotest import folds
import pool_pu as PU
import eval_pool_pu as EV
import make_pool_pu_candidate as MC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get(
    'CHAMP_POOL_PU_BASE',
    os.path.join(ROOT, 'research', 'final_video_20260910',
                 'submission_seqpair_stable_iter500_v1.csv'))
OUT = os.environ.get(
    'CHAMP_POOL_PU_STABILITY',
    os.path.join(ROOT, 'research', 'pool_pu_test_stability.csv'))


def main():
    tr, te, meta = load_all()
    vis = real_test_view(te)
    base = pd.read_csv(BASE).set_index('qa_id').prediction.to_dict()
    with open(os.path.join(ROOT, 'champ', 'test_blocks_repaired.json')) as f:
        blocks = [[int(x) for x in b] for b in json.load(f)]
    cat = dict(zip(vis.qa_id, vis.category))
    sc = MC.statcache({'oof': tr[tr.source == 'HAU'].path.unique(),
                       'test': vis.true_path.unique()})
    users = sorted(tr.user.dropna().unique())
    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        trn = tr[~tr.user.isin(hold)]
        latent = PU.segment_pools(meta, trn.user.unique())
        weak = PU.selected_pools(trn)
        PU.set_cooc(PU.fit_segment_cooc(meta, trn.user.unique()))
        clf, cols = PU.fit_model(EV.training_blocks(tr, trn.user.unique(), weak),
                                 latent, weak, sc)
        scorer = PU.make_scorer(clf, cols)
        proposed = {}
        for blk in blocks:
            p, _ = PU.solve_block(vis, blk, sc, scorer)
            proposed.update(p)
        for qid, old in base.items():
            new = proposed.get(qid, old)
            qcat = cat[qid]
            accept = (qcat == 'combination' or
                      (qcat == 'multi' and len(new) < len(old)))
            rows.append(dict(fold=fi, qa_id=qid, category=qcat, base=old,
                             proposal=new, accepted=int(accept and new != old)))
        print(f'  fold {fi} complete', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    a = d[d.accepted.eq(1)]
    votes = a.groupby(['qa_id', 'category', 'base', 'proposal']).fold.nunique()
    print(f'wrote {OUT}')
    if len(a):
        print(a.groupby(['qa_id', 'category', 'base', 'proposal']).size().sort_values(ascending=False).to_string())
        print('stable 5/5:', int((votes == 5).sum()), 'stable >=4/5:', int((votes >= 4).sum()))
    else:
        print('no accepted changes')


if __name__ == '__main__':
    main()
