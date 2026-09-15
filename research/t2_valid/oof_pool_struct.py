"""CPU-only grouped-subject 5-fold STRUCT pool-solver OOF + real-test pool run.

Same protocol as oof_driver.py (folds seed=7, make_pseudo seed=100+fi) but the
pool-membership model uses ONLY structural features (CHAMP_POOL_FEATS=struct,
empty dense statcache). Scores single/multi/combination (sequence order needs
dense). Compares per-row vs oof_pipeline_base.csv (single 1216/1238, multi
782/809, combo 786/790) and reports the real-test letters for the T-affected
rows (test_0137/0289) plus test_0206.

Usage: PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 research/t2_valid/oof_pool_struct.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'champ'))

os.environ['CHAMP_POOL_FEATS'] = 'struct'

from core import (load_all, make_pseudo, real_test_view, fit_group_model,  # noqa: E402
                  infer_blocks)
from pseudotest import folds  # noqa: E402
import pool as PL  # noqa: E402
import pipeline as PIPE  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--only-test', action='store_true')
    ap.add_argument('--skip-test', action='store_true')
    a = ap.parse_args()
    if a.only_test:
        run_test_view()
        return
    want = {int(x) for x in a.folds.split(',') if x.strip() != ''}
    tr, te, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    base = pd.read_csv(os.path.join(
        ROOT, 'research', 'final_video_20260910', 'oof_pipeline_base.csv'))
    base_a = base[base.category.isin(['single', 'multi', 'combination'])].set_index(
        ['qa_id', 'fold'])

    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        if fi not in want:
            continue
        trn = tr[tr.user.isin([u for u in users if u not in hold])]
        PL.fit_cooc(trn)
        tpool = PIPE.true_pool(trn)
        bbu = PIPE.training_blocks(trn, sorted(trn.user.dropna().unique()), tpool)
        clf, cols = PL.fit_pool_model(trn, None, bbu, {})
        scorer = PL.make_scorer(clf, cols)
        gm = fit_group_model(trn)
        vis, key, _aux = make_pseudo(tr, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, gm)
        ans = dict(zip(key.qa_id, key.answer))
        pred = {}
        n_greedy = 0
        for blk in blocks:
            uni, rows_a, qs, forced = PL.candidate_evidence(vis, list(blk),
                                                            'oof', {})
            _free, comps, _fixed = PL._components(qs, uni, forced)
            if any(len(a) > 20 for a, _ in comps):
                # Oversized for exact enumeration on this box: greedy fallback
                # (unconstrained argmax pool + best-option decode), counted.
                lo = scorer(rows_a, uni)
                best = set(forced) | {a for a in uni if lo.get(a, 0) > 0}
                for r in qs:
                    ops = PL.qopts(r)
                    hit = [i for i, op in enumerate(ops)
                           if all(a in best for a in op)]
                    if r.category in ('single', 'combination'):
                        if len(hit) != 1:
                            hit = [max(range(4), key=lambda i: sum(
                                lo.get(a, 0.0) for a in ops[i]))]
                        pred[r.qa_id] = 'ABCD'[hit[0]]
                    elif r.category == 'multi':
                        if not hit:
                            hit = [max(range(4), key=lambda i: sum(
                                lo.get(a, 0.0) for a in ops[i]))]
                        pred[r.qa_id] = ''.join('ABCD'[i] for i in sorted(hit))
                n_greedy += 1
                continue
            pp, _bd = PL.solve_block(vis, list(blk), {}, scorer, 'oof')
            pred.update(pp)
        if n_greedy:
            print(f'  fold {fi}: greedy fallback on {n_greedy} blocks',
                  flush=True)
        for q, pv in pred.items():
            bro = base_a.loc[(q, fi)] if (q, fi) in base_a.index else None
            cat = vis[vis.qa_id == q].category.iloc[0]
            rows.append(dict(qa_id=q, fold=fi, category=cat, pred=pv,
                             answer=ans[q],
                             base_pred=(bro['pred'] if bro is not None else None),
                             base_correct=(int(bro['correct'])
                                           if bro is not None else None)))
        dd = [r for r in rows if r['fold'] == fi]
        c = sum(1 for r in dd if r['pred'] == r['answer'])
        print(f'fold {fi}: struct-pool {c}/{len(dd)}={c / len(dd):.4f}', flush=True)
        pd.DataFrame(dd).to_csv(
            os.path.join(ROOT, 'research', 't2_valid',
                         f'oof_pool_struct.fold{fi}.csv'), index=False)

    d = pd.DataFrame(rows)
    print('--- struct-pool OOF by category ---')
    for cat, g in d.groupby('category'):
        c = int((g.pred == g.answer).sum())
        print(f'{cat:14s} {c}/{len(g)} = {c / len(g):.4f}')
    c = int((d.pred == d.answer).sum())
    print(f'{"TOTAL":14s} {c}/{len(d)} = {c / len(d):.4f}')
    bb = d[d.base_correct.notna()]
    print(f'baseline rows matched: {len(bb)}/{len(d)}')
    for cat, g in bb.groupby('category'):
        wr = int(((g.base_pred != g.answer) & (g.pred == g.answer)).sum())
        rw = int(((g.base_pred == g.answer) & (g.pred != g.answer)).sum())
        print(f'{cat:14s} vs baseline: W->R {wr}, R->W {rw}, net {wr - rw:+d} '
              f'(base acc {g.base_correct.mean():.4f})')
    out = os.path.join(ROOT, 'research', 't2_valid', 'oof_pool_struct.csv')
    d.to_csv(out, index=False)
    print('wrote', out)

    # ---- real-test run (fit on all users) ----
    if not a.skip_test:
        run_test_view()


def run_test_view():
    tr, te, meta = load_all()
    print('--- real-test struct-pool run (all-users fit) ---', flush=True)
    PL.fit_cooc(tr)
    tpool = PIPE.true_pool(tr)
    bbu = PIPE.training_blocks(tr, sorted(tr.user.dropna().unique()), tpool)
    clf, cols = PL.fit_pool_model(tr, None, bbu, {})
    scorer = PL.make_scorer(clf, cols)
    gm = fit_group_model(tr)
    vis = real_test_view(te)
    blocks = infer_blocks(vis, gm)
    pred = {}
    for blk in blocks:
        pp, _bd = PL.solve_block(vis, list(blk), {}, scorer, 'test')
        pred.update(pp)
    pipe = pd.read_csv(os.path.join(ROOT, 'submissions/submission_final.csv'))
    pipe_pred = dict(zip(pipe.qa_id, pipe['prediction'].astype(str)))
    champ = pd.read_csv(os.path.join(
        ROOT, 'submissions/submission_097076_332of342_CHAMPION.csv'))
    champ_pred = dict(zip(champ.qa_id, champ['prediction'].astype(str)))
    for q in ['test_0137', 'test_0206', 'test_0289']:
        r = vis[vis.qa_id == q].iloc[0]
        print(f'{q} [{r.category}]: pipeline={pipe_pred[q]} '
              f'champion={champ_pred[q]} struct-pool={pred.get(q)}')
    pd.DataFrame([dict(qa_id=q, prediction=p) for q, p in pred.items()]).to_csv(
        os.path.join(ROOT, 'research', 't2_valid', 'test_pool_struct.csv'),
        index=False)


if __name__ == '__main__':
    main()
