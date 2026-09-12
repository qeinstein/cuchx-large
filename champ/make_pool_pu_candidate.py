"""Build an unsent test candidate from the OOF-validated PU action branch.

The candidate starts from the stable sequence submission and changes only:
  * combination answers in known complete three-clip blocks;
  * multi answers in those blocks when the PU answer is strictly shorter.

The withheld-pair audit is negative, so pair and singleton blocks remain frozen.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, real_test_view
import decode as DC
import pool_pu as PU
import eval_pool_pu as EV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get(
    'CHAMP_POOL_PU_BASE',
    os.path.join(ROOT, 'research', 'final_video_20260910',
                 'submission_seqpair_stable_iter500_v1.csv'))
OUT = os.environ.get(
    'CHAMP_POOL_PU_OUT',
    os.path.join(ROOT, 'research', 'final_video_20260910',
                 'submission_pool_pu_complete3_gate_v1.csv'))
LOGITS = os.environ.get('CHAMP_LOGITS', 'dense_logits_full40.npz')


def statcache(paths):
    logits_path = LOGITS if os.path.isabs(LOGITS) else os.path.join(ROOT, 'champ', LOGITS)
    DC._LG = np.load(logits_path)
    out = {}
    for split, ps in paths.items():
        for p in ps:
            lp = DC.logits(split, p)
            if lp is not None:
                out[p] = DC.presence_stats(lp)
    return out


def main():
    tr, te, meta = load_all()
    base = pd.read_csv(BASE).set_index('qa_id').prediction.to_dict()
    vis = real_test_view(te)
    with open(os.path.join(ROOT, 'champ', 'test_blocks_repaired.json')) as f:
        blocks = [[int(x) for x in b] for b in json.load(f)]

    latent = PU.segment_pools(meta)
    weak = PU.selected_pools(tr)
    PU.set_cooc(PU.fit_segment_cooc(meta))
    sc = statcache({
        'oof': tr[tr.source == 'HAU'].path.unique(),
        'test': vis.true_path.unique(),
    })
    tb = EV.training_blocks(tr, tr.user.unique(), weak)
    clf, cols = PU.fit_model(tb, latent, weak, sc)
    scorer = PU.make_scorer(clf, cols)

    proposed = {}
    for blk in blocks:
        pred, _diag = PU.solve_block(vis, blk, sc, scorer)
        proposed.update(pred)

    out = dict(base)
    changes = []
    cat = dict(zip(vis.qa_id, vis.category))
    for blk in blocks:
        if len(blk) != 3:
            continue
        qids = vis[vis.idx.isin(blk)].qa_id
        for qid in qids:
            qid = str(qid)
            old = out.get(qid)
            new = proposed.get(qid)
            if old is None or new is None or old == new:
                continue
            accept = (cat[qid] == 'combination' or
                      (cat[qid] == 'multi' and len(new) < len(old)))
            if accept:
                out[qid] = new
                changes.append(dict(qa_id=qid, category=cat[qid], old=old,
                                    new=new, block_size=3))

    result = pd.DataFrame({'qa_id': list(base),
                           'prediction': [out[q] for q in base]})
    result.to_csv(OUT, index=False)
    print(f'base={BASE}')
    print(f'out={OUT}')
    print(f'blocks={len(blocks)} sizes={pd.Series([len(x) for x in blocks]).value_counts().to_dict()}')
    print(f'accepted changes={len(changes)} by_category={pd.Series([x["category"] for x in changes]).value_counts().to_dict() if changes else {}}')
    for x in changes:
        print(f'  {x["qa_id"]} {x["category"]}: {x["old"]} -> {x["new"]}')
    if len(result) != len(te) or set(result.qa_id) != set(te.qa_id):
        raise RuntimeError('candidate does not match test QA ids')


if __name__ == '__main__':
    main()
