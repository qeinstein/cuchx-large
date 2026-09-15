"""Final runner.  ONE code path: `pipeline.solve` is used verbatim for both the
subject-disjoint held-out validation and the real Kaggle test set.

  python champ/final.py validate     -> 5-fold leakage-safe OOF + full report
  python champ/final.py test         -> submission_final.csv + coverage/fallback report

Fallback policy: `solve` returns None for any question whose evidence is entirely absent.
Those are filled from the immutable champion (or `CHAMP_FALLBACK`) on test and counted as
errors on validation.  A silent constant answer is never produced.
"""
import os, sys, json
from collections import defaultdict
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, make_pseudo, real_test_view
import pipeline as P
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = ['single', 'multi', 'combination', 'emotion', 'sequence', 'object_interaction']


# --------------------------------------------------------------------------- reporting
def coverage_report(vis, diag, label):
    print(f'\n--- cache coverage ({label}) ---')
    nclip = vis.true_path.nunique()
    print(f'  clips                           : {nclip}')
    print(f'  engineered/physical features ok : {diag.get("feat_ok", 0)}'
          f'   missing: {diag.get("feat_missing", 0)}')
    print(f'  dense temporal logits ok        : {diag.get("logits_ok", 0)}'
          f'   missing: {diag.get("logits_missing", 0)}')
    print(f'  HARn sequence classifier ok     : {diag.get("harn_clf_ok", 0)}'
          f'   missing: {diag.get("harn_clf_missing", 0)}')
    print(f'  HARn parent session-pool ok     : {diag.get("harn_pool_ok", 0)}'
          f'   missing: {diag.get("harn_pool_missing", 0)}')
    print(f'  session pools satisfiable       : {diag.get("pool_sat", 0)}'
          f'   unsatisfiable: {diag.get("pool_unsat", 0)}')
    print(f'  sequence questions with logits  : {diag.get("seq_ok", 0)}'
          f'   without: {diag.get("seq_no_evidence", 0)}')
    print(f'\n--- fallbacks ({label}) ---')
    print(f'  total fallback predictions      : {diag.get("fallback", 0)}')
    for k in ('harn_no_evidence', 'harn_flat_scores', 'object_flat_prior',
              'seq_no_evidence', 'never_predicted'):
        if diag.get(k):
            print(f'    {k:24s}: {diag[k]}')
    fq = diag.get('fallback_qids', [])
    if fq:
        print('    ids:', ', '.join(f'{q}({why})' for q, why in fq[:20]),
              '...' if len(fq) > 20 else '')


# --------------------------------------------------------------------------- validate
def validate(nfold=5):
    tr, te, meta = load_all()
    P.caches(meta)
    users = sorted(tr.user.dropna().unique())
    rows, diag = [], defaultdict(int)
    diag['fallback_qids'] = []
    for fi, hold in enumerate(folds(users, nfold)):
        ctx = P.fit_all(tr, meta, hold)
        vis, key, aux = make_pseudo(tr, meta, hold, seed=100 + fi)
        pred, blocks, pool_of, diag = P.solve(vis, ctx, 'oof', diag)
        ans = dict(zip(key.qa_id, key.answer))
        for r in vis.itertuples():
            pv = pred.get(r.qa_id)
            rows.append(dict(qa_id=r.qa_id, fold=fi, source=r.source, category=r.category,
                             pred=pv, answer=ans[r.qa_id],
                             fell_back=int(pv is None),
                             correct=int(pv == ans[r.qa_id])))
        print(f'  fold {fi} done ({len(hold)} users)', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, 'champ', 'oof_final.csv'), index=False)
    v7 = pd.read_csv(os.path.join(ROOT, 'oof_v7_final.csv'))[['qa_id', 'correct']] \
        .rename(columns={'correct': 'c7'})
    m = d.merge(v7, on='qa_id', how='left')

    print('\n================ FINAL HELD-OUT VALIDATION (5-fold subject-disjoint) ================')
    print(f'  {"category":22s} {"correct":>12s}   {"acc":>7s}      v7')
    for c in CATS:
        g = m[m.category == c]
        print(f'  {c:22s} {g.correct.sum():5d}/{len(g):5d}   {g.correct.mean():.4f}'
              f'   {int(g.c7.sum()):5d} = {g.c7.mean():.4f}')
    print(f'  {"TOTAL":22s} {m.correct.sum():5d}/{len(m):5d}   {m.correct.mean():.4f}'
          f'   {int(m.c7.sum()):5d} = {m.c7.mean():.4f}')
    w = ((m.correct == 1) & (m.c7 == 0)).sum(); l = ((m.correct == 0) & (m.c7 == 1)).sum()
    print(f'\n  wins vs v7 {w}   losses vs v7 {l}   net {w - l:+d}')
    print('  per-fold:', '  '.join(
        f'f{f}={g.correct.sum()}/{len(g)}={g.correct.mean():.4f}' for f, g in m.groupby('fold')))
    print(f'\n  predictions that fell back (counted as errors): {int(d.fell_back.sum())}')
    coverage_report(pd.DataFrame({'true_path': d.qa_id}), diag, 'validation, all folds pooled')
    return m


# --------------------------------------------------------------------------- test
def run_test():
    tr, te, meta = load_all()
    P.caches(meta)
    ctx = P.fit_all(tr, meta, hold_users=[])
    vis = real_test_view(te)
    pred, blocks, pool_of, diag = P.solve(vis, ctx, 'test')

    fallback_name = os.environ.get('CHAMP_FALLBACK',
                                    'submissions/submission_097076_332of342_CHAMPION.csv')
    fallback_path = os.path.join(ROOT, fallback_name)
    if not os.path.exists(fallback_path):
        raise FileNotFoundError('CHAMP_FALLBACK does not exist: ' + fallback_path)
    v8 = pd.read_csv(fallback_path).set_index('qa_id').prediction
    if list(v8.index) != list(te.qa_id):
        raise ValueError('fallback row order/key mismatch: ' + fallback_path)
    out, nfb = [], 0
    for q in te.qa_id:
        p = pred.get(q)
        if p is None:
            p = v8.loc[q]; nfb += 1
        out.append(p)
    sub = pd.DataFrame({'qa_id': te.qa_id, 'prediction': out})

    # ---- format validation
    cat = dict(zip(te.qa_id, te.category))
    bad = []
    for q, p in zip(sub.qa_id, sub.prediction):
        p = str(p); c = cat[q]
        if not p or any(ch not in 'ABCD' for ch in p) or len(set(p)) != len(p):
            bad.append((q, p, c))
        elif c in ('single', 'emotion', 'combination', 'object_interaction') and len(p) != 1:
            bad.append((q, p, c))
        elif c == 'sequence' and sorted(p) != list('ABCD'):
            bad.append((q, p, c))
        elif c == 'multi' and not (1 <= len(p) <= 4):
            bad.append((q, p, c))
    assert len(sub) == 682 and list(sub.qa_id) == list(te.qa_id), 'row/order mismatch'
    assert not bad, bad[:10]
    outp = os.path.join(ROOT, os.environ.get('CHAMP_OUT','submissions/submission_final.csv'))
    sub.to_csv(outp, index=False)

    print('\n================ REAL TEST INFERENCE ================')
    coverage_report(vis, diag, 'test')
    print(f'\n  blocks inferred: {len(blocks)}  sizes '
          f'{pd.Series([len(b) for b in blocks]).value_counts().to_dict()}')
    print(f'  fallback predictions taken from {fallback_name}: {nfb}')
    print('\n--- prediction distribution by category ---')
    mm = te[['qa_id', 'source', 'category']].merge(sub, on='qa_id')
    for (s, c), g in mm.groupby(['source', 'category']):
        vc = g.prediction.value_counts()
        print(f'  {s:5s} {c:19s} n={len(g):3d} distinct={g.prediction.nunique():2d} '
              f'top={dict(list(vc.items())[:3])}')
    v9 = pd.read_csv(os.path.join(ROOT, 'submissions/submission_v9.csv')).set_index('qa_id').prediction
    dif = mm.assign(chg=[p != v9.loc[q] for q, p in zip(mm.qa_id, mm.prediction)])
    print(f'\n--- diff vs submitted v9 (public 0.85087) ---')
    print(dif.groupby(['source', 'category']).chg.agg(['sum', 'count']).to_string())
    print(f'  total changed vs v9: {int(dif.chg.sum())} / 682')
    print(f'\n  written {outp}')


if __name__ == '__main__':
    if sys.argv[1] == 'validate':
        validate()
    else:
        run_test()
