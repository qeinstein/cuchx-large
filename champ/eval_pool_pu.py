"""Subject-disjoint OOF audit for the noisy-channel action-pool decoder."""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, make_pseudo, fit_group_model, infer_blocks
from pseudotest import folds, thin_to_pairs
import decode as DC
import pool as PL
import pool_pu as PU

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGITS = os.environ.get('CHAMP_LOGITS', 'dense_logits.npz')
BASELINE = os.environ.get(
    'CHAMP_OOF_BASE',
    os.path.join(ROOT, 'research', 'final_video_20260910', 'oof_pipeline_base.csv'))


def build_statcache(paths, split='oof'):
    logits_path = LOGITS if os.path.isabs(LOGITS) else os.path.join(ROOT, 'champ', LOGITS)
    DC._LG = np.load(logits_path)
    out = {}
    for path in paths:
        lp = DC.logits(split, path)
        if lp is not None:
            out[path] = DC.presence_stats(lp)
    return out


def training_blocks(tr, users, weak):
    out = {}
    h = tr[(tr.source == 'HAU') & tr.user.isin(users)]
    for (u, a, b), g in h.groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique())
        idx = {p: i for i, p in enumerate(paths)}
        v = g.copy()
        v['idx'] = v.path.map(idx)
        v['true_path'] = v.path
        key = PU._session_key(u, a, b)
        out[key] = (v, list(range(len(paths))), weak.get(key, set()))
    return out


def run(nfold=5, verbose=True):
    tr, _te, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    baseline = pd.read_csv(BASELINE).set_index('qa_id')
    action_cats = ['single', 'multi', 'combination']
    total = defaultdict(lambda: [0, 0])
    base_total = defaultdict(lambda: [0, 0])
    transitions = defaultdict(lambda: [0, 0])  # baseline wrong->new right, opposite
    rows_out = []

    for fi, hold in enumerate(folds(users, nfold)):
        trn = tr[~tr.user.isin(hold)]
        latent = PU.segment_pools(meta, trn.user.unique())
        weak = PU.selected_pools(trn)
        cooc = PU.fit_segment_cooc(meta, trn.user.unique())
        PU.set_cooc(cooc)
        # Raw dense logits are test-visible, so the cache may include held-out
        # clips; only the pool targets/co-occurrence below are fold restricted.
        statcache = build_statcache(tr[tr.source == 'HAU'].path.unique())
        tb = training_blocks(tr, trn.user.unique(), weak)
        clf, cols = PU.fit_model(tb, latent, weak, statcache)
        scorer = PU.make_scorer(clf, cols)

        if os.environ.get('CHAMP_POOL_PU_PAIR', '0') == '1':
            tr_eval = thin_to_pairs(tr, hold, frac=1.0, seed=fi, policy='ends')
        else:
            tr_eval = tr
        vis, key, aux = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
        gm = fit_group_model(trn)
        blocks = infer_blocks(vis, gm)
        answers = dict(zip(key.qa_id, key.answer))
        cat_of = dict(zip(vis.qa_id, vis.category))
        users_hit = {str(u) for u in hold}
        for blk in blocks:
            pred, diag = PU.solve_block(vis, blk, statcache, scorer)
            for qa_id, guess in pred.items():
                cat = cat_of[qa_id]
                if cat not in action_cats:
                    continue
                truth = answers[qa_id]
                new_ok = int(guess == truth)
                old_ok = int(baseline.loc[qa_id, 'pred'] == truth)
                total[cat][0] += new_ok; total[cat][1] += 1
                base_total[cat][0] += old_ok; base_total[cat][1] += 1
                transitions['W2R'][0] += int(not old_ok and new_ok)
                transitions['R2W'][0] += int(old_ok and not new_ok)
                rows_out.append(dict(fold=fi, qa_id=qa_id, category=cat,
                                     truth=truth, baseline=baseline.loc[qa_id, 'pred'],
                                     prediction=guess, new_ok=new_ok, base_ok=old_ok,
                                     block_size=len(blk)))
        if verbose:
            print(f'  fold {fi} hold={len(hold)} blocks={len(blocks)}', flush=True)

    out = pd.DataFrame(rows_out)
    out_path = os.environ.get('CHAMP_POOL_PU_OOF',
                              os.path.join(ROOT, 'research', 'pool_pu_oof.csv'))
    out.to_csv(out_path, index=False)
    print('\n  PU pool vs final-video OOF baseline:')
    for cat in action_cats:
        nk, nn = total[cat]; bk, bn = base_total[cat]
        print(f'    {cat:12s} new {nk:4d}/{nn:4d}={nk/max(1,nn):.4f}  '
              f'base {bk:4d}/{bn:4d}={bk/max(1,bn):.4f}  delta={nk-bk:+d}')
    nk, nn = sum(v[0] for v in total.values()), sum(v[1] for v in total.values())
    bk, bn = sum(v[0] for v in base_total.values()), sum(v[1] for v in base_total.values())
    print(f'    ACTION TOTAL  new {nk}/{nn}={nk/max(1,nn):.4f}  '
          f'base {bk}/{bn}={bk/max(1,bn):.4f}  delta={nk-bk:+d}')
    print(f'    transitions W->R={int(out[(out.base_ok==0)&(out.new_ok==1)].shape[0])} '
          f'R->W={int(out[(out.base_ok==1)&(out.new_ok==0)].shape[0])}')
    print(f'  wrote {out_path}')
    return out


if __name__ == '__main__':
    run()
