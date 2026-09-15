"""CPU-only grouped-subject 5-fold emotion OOF + real-test emotion run (T2 valid).

Mirrors research/final_video_20260910/oof_driver.py protocol exactly
(folds seed=7, make_pseudo seed=100+fi, pipeline default weights) but solves
ONLY the emotion category via champ/core.py fit_manner + solve_emotion with the
emopair head (CHAMP_EMO_PAIR=1 default). No dense logits anywhere.

Two pool-context arms (pool context only enters via emopair pair_logodds):
  A: pool_of=None (pure CPU, no pool evidence)
  B: pool_of=struct pools (CHAMP_POOL_FEATS=struct solver, empty statcache)

Compares per-row against research/final_video_20260910/oof_pipeline_base.csv
emotion rows (758/809 = 0.9370 baseline) and, on the real test view, reports the
17 champion-diff emotion rows (tests PRIORFIX reproducibility: 0427/0429/0458).

Usage: PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 research/t2_valid/oof_emotion_cpu.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'champ'))

os.environ.setdefault('CHAMP_POOL_FEATS', 'struct')

from core import (load_all, make_pseudo, real_test_view, fit_group_model,  # noqa: E402
                  infer_blocks, fit_manner, solve_emotion)
from pseudotest import folds  # noqa: E402
import emopair as EP  # noqa: E402
import pool as PL  # noqa: E402
import pipeline as PIPE  # noqa: E402

DIFF17 = ['test_0385', 'test_0386', 'test_0387', 'test_0418', 'test_0426',
          'test_0427', 'test_0429', 'test_0431', 'test_0436', 'test_0443',
          'test_0444', 'test_0458', 'test_0459', 'test_0461', 'test_0464',
          'test_0470', 'test_0679']


def fit_struct_scorer(trn):
    PL.fit_cooc(trn)
    tpool = PIPE.true_pool(trn)
    users = sorted(trn.user.dropna().unique())
    bbu = PIPE.training_blocks(trn, users, tpool)
    clf, cols = PL.fit_pool_model(trn, None, bbu, {})
    return PL.make_scorer(clf, cols)


def struct_pools(vis, blocks, scorer):
    pool_of = {}
    skipped = 0
    for blk in blocks:
        uni, rows, qs, forced = PL.candidate_evidence(vis, list(blk), 'oof', {})
        _free, comps, _fixed = PL._components(qs, uni, forced)
        # _enum_component materialises 2**n assignments (n=22 -> ~700MB
        # transient); skip pool context for oversized blocks (2GB box).
        if any(len(a) > 20 for a, _ in comps):
            skipped += 1
            continue
        _pp, bd = PL.solve_block(vis, list(blk), {}, scorer, 'oof')
        if bd:
            pool_of[tuple(blk)] = set(bd['pool'])
    if skipped:
        print(f'  struct_pools: skipped {skipped} oversized blocks', flush=True)
    return pool_of


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--skip-test', action='store_true')
    ap.add_argument('--only-test', action='store_true')
    ap.add_argument('--no-struct', action='store_true',
                    help='skip the struct-pool arm (lean: no pool fit/solve)')
    a = ap.parse_args()
    if a.only_test:
        run_test_view(struct=not a.no_struct)
        return
    want = {int(x) for x in a.folds.split(',') if x.strip() != ''}
    tr, te, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    base = pd.read_csv(os.path.join(
        ROOT, 'research', 'final_video_20260910', 'oof_pipeline_base.csv'))
    base_e = base[base.category == 'emotion'].set_index(['qa_id', 'fold'])

    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        if fi not in want:
            continue
        trn = tr[tr.user.isin([u for u in users if u not in hold])]
        gm = fit_group_model(trn)
        vis, key, _aux = make_pseudo(tr, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, gm)
        mm = fit_manner(trn, meta)
        tpool = PIPE.true_pool(trn)
        mm['pm'] = EP.fit(trn, meta, mm, pool_of=tpool)
        ans = dict(zip(key.qa_id, key.answer))
        emo_q = vis[vis.category == 'emotion'].qa_id.tolist()

        pe_a, _ = solve_emotion(vis, blocks, mm, w_phys=1.0, w_pos=1.0,
                                w_pair=1.0, pool_of=None)
        if a.no_struct:
            pe_b = {}
        else:
            scorer = fit_struct_scorer(trn)
            sp = struct_pools(vis, blocks, scorer)
            pe_b, _ = solve_emotion(vis, blocks, mm, w_phys=1.0, w_pos=1.0,
                                    w_pair=1.0, pool_of=sp)
        for q in emo_q:
            bro = base_e.loc[(q, fi)] if (q, fi) in base_e.index else None
            rows.append(dict(
                qa_id=q, fold=fi,
                pred_nopool=pe_a.get(q), pred_structpool=pe_b.get(q),
                answer=ans[q],
                base_pred=(bro['pred'] if bro is not None else None),
                base_correct=(int(bro['correct']) if bro is not None else None)))
        ca = sum(1 for r in rows if r['fold'] == fi
                 and r['pred_nopool'] == r['answer'])
        cb = sum(1 for r in rows if r['fold'] == fi
                 and r['pred_structpool'] == r['answer'])
        n = sum(1 for r in rows if r['fold'] == fi)
        print(f'fold {fi}: nopool {ca}/{n}={ca / n:.4f} '
              f'structpool {cb}/{n}={cb / n:.4f}', flush=True)
        pd.DataFrame([r for r in rows if r['fold'] == fi]).to_csv(
            os.path.join(ROOT, 'research', 't2_valid',
                         f'oof_emotion_cpu.fold{fi}.csv'), index=False)

    d = pd.DataFrame(rows)
    n = len(d)
    arms = ['pred_nopool'] + ([] if a.no_struct else ['pred_structpool'])
    for arm in arms:
        c = int((d[arm] == d['answer']).sum())
        print(f'{arm}: {c}/{n} = {c / n:.4f}')
    bb = d[d.base_correct.notna()]
    print(f'baseline rows matched: {len(bb)}/{n}; '
          f'baseline acc on matched: {bb.base_correct.mean():.4f}')
    for arm in arms:
        agree = int((d[arm].astype(str) == d['base_pred'].astype(str)).sum())
        wr = int((((d['base_pred'] != d['answer']) & (d[arm] == d['answer']))).sum())
        rw = int((((d['base_pred'] == d['answer']) & (d[arm] != d['answer']))).sum())
        print(f'{arm} vs baseline: agree {agree}/{n}, W->R {wr}, R->W {rw}, '
              f'net {wr - rw:+d}')
    out = os.path.join(ROOT, 'research', 't2_valid', 'oof_emotion_cpu.csv')
    d.to_csv(out, index=False)
    print('wrote', out)

    # ---- real-test run (fit on all users) ----
    if not a.skip_test:
        run_test_view()


def run_test_view(struct=True):
    """Fit on all training users, solve the real-test emotion questions.

    Returns a DataFrame with pipeline/champion/cpu letters for the 17
    champion-diff emotion rows (also written to test_emotion_cpu.csv).
    Import-safe for apply_patches.py (which uses only the cpu letters).
    """
    tr, te, meta = load_all()
    print('--- real-test emotion run (all-users fit) ---', flush=True)
    gm = fit_group_model(tr)
    print('group model done', flush=True)
    vis = real_test_view(te)
    blocks = infer_blocks(vis, gm)
    print(f'inferred {len(blocks)} test blocks', flush=True)
    mm = fit_manner(tr, meta)
    print('fit_manner done', flush=True)
    mm['pm'] = EP.fit(tr, meta, mm, pool_of=PIPE.true_pool(tr))
    print('emopair done', flush=True)
    pe_a, _ = solve_emotion(vis, blocks, mm, w_phys=1.0, w_pos=1.0,
                            w_pair=1.0, pool_of=None)
    print('solve nopool done', flush=True)
    if struct:
        scorer = fit_struct_scorer(tr)
        sp = struct_pools(vis, blocks, scorer)
        pe_b, _ = solve_emotion(vis, blocks, mm, w_phys=1.0, w_pos=1.0,
                                w_pair=1.0, pool_of=sp)
        print('solve structpool done', flush=True)
    else:
        pe_b = {}
    pipe = pd.read_csv(os.path.join(ROOT, 'submissions/submission_final.csv'))
    pipe_pred = dict(zip(pipe.qa_id, pipe['prediction'].astype(str)))
    champ = pd.read_csv(os.path.join(
        ROOT, 'submissions/submission_097076_332of342_CHAMPION.csv'))
    champ_pred = dict(zip(champ.qa_id, champ['prediction'].astype(str)))
    trows = []
    for q in DIFF17:
        trows.append(dict(qa_id=q, pipeline=pipe_pred[q], champion=champ_pred[q],
                          cpu_nopool=pe_a.get(q), cpu_structpool=pe_b.get(q),
                          match_nopool=int(pe_a.get(q) == champ_pred[q]),
                          match_struct=int(pe_b.get(q) == champ_pred[q])))
    t = pd.DataFrame(trows)
    print(t.to_string(index=False))
    print(f"cpu_nopool matches champion: {int(t.match_nopool.sum())}/17; "
          f"cpu_structpool: {int(t.match_struct.sum())}/17")
    tout = os.path.join(ROOT, 'research', 't2_valid', 'test_emotion_cpu.csv')
    t.to_csv(tout, index=False)
    print('wrote', tout)
    return t


if __name__ == '__main__':
    main()
