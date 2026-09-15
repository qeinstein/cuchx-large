"""Flip audit for the 15 test changes between the rank-6 champion and the final assault.

No new model is trained.  The only fitting is the same light heads (HistGradientBoosting /
logistic regression / count tables) the pipeline already uses, re-run under two configs so
that champion and assault predictions can be compared question-by-question on the same folds.

Two mechanisms separate the configs:
  H = HARn DINOv2 late fusion   (CHAMP_W_HDINO 0.0 -> 0.3)   -> HARn single, object
  E = emotion DINOv2 features   (CHAMP_EMO_DINO 0 -> 1)      -> emotion
Everything else (structural pool, session blocks, biconditional, emotion pairwise) is
identical in both, so every flip is attributable to H or E.
"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = ['single', 'multi', 'combination', 'emotion', 'sequence', 'object_interaction']


def run_config(tag, env):
    """5-fold held-out predictions + decision margins under one config."""
    out = os.path.join(ROOT, 'champ', f'audit_{tag}.csv')
    if os.path.exists(out):
        print(f'  [{tag}] reusing {out}')
        return pd.read_csv(out)
    import importlib
    for k, v in env.items():
        os.environ[k] = v
    for m in ('core', 'pool', 'harn', 'emopair', 'pipeline', 'dense', 'decode'):
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    import core, pipeline as P
    from core import load_all, make_pseudo
    from pseudotest import folds
    core.MARGIN.clear()
    tr, te, meta = load_all()
    P.caches(meta)
    users = sorted(tr.user.dropna().unique())
    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        ctx = P.fit_all(tr, meta, hold)
        vis, key, aux = make_pseudo(tr, meta, hold, seed=100 + fi)
        pred, blocks, pool_of, diag = P.solve(vis, ctx, 'oof')
        ans = dict(zip(key.qa_id, key.answer))
        for r in vis.itertuples():
            pv = pred.get(r.qa_id)
            rows.append(dict(qa_id=r.qa_id, fold=fi, source=r.source, category=r.category,
                             pred=pv, answer=ans[r.qa_id],
                             margin=core.MARGIN.get(r.qa_id, np.nan),
                             correct=int(pv == ans[r.qa_id])))
        print(f'  [{tag}] fold {fi} done', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    return d


def main():
    print('=' * 78)
    print('PHASE 1  held-out predictions under the two configs (light heads only, no training)')
    print('=' * 78)
    champ = run_config('champ', {'CHAMP_EMO_DINO': '0', 'CHAMP_W_HDINO': '0.0'})
    assault = run_config('assault', {'CHAMP_EMO_DINO': '1', 'CHAMP_W_HDINO': '0.3'})

    m = champ[['qa_id', 'fold', 'source', 'category', 'pred', 'answer', 'correct', 'margin']] \
        .rename(columns={'pred': 'p_c', 'correct': 'c_c', 'margin': 'mg_c'}) \
        .merge(assault[['qa_id', 'pred', 'correct', 'margin']]
               .rename(columns={'pred': 'p_a', 'correct': 'c_a', 'margin': 'mg_a'}), on='qa_id')
    m['mech'] = np.where(m.category == 'emotion', 'E',
                         np.where(m.source == 'HARn', 'H', '-'))
    m['flip'] = m.p_c.astype(str) != m.p_a.astype(str)

    print('\n' + '=' * 78)
    print('PHASE 2  OOF totals')
    print('=' * 78)
    print(f'  {"category":22s} {"champion":>10s} {"assault":>10s} {"delta":>7s}')
    for c in CATS:
        g = m[m.category == c]
        print(f'  {c:22s} {g.c_c.sum():10d} {g.c_a.sum():10d} {g.c_a.sum()-g.c_c.sum():+7d}')
    print(f'  {"TOTAL":22s} {m.c_c.sum():10d} {m.c_a.sum():10d} '
          f'{m.c_a.sum()-m.c_c.sum():+7d}   of {len(m)}')

    print('\n' + '=' * 78)
    print('PHASE 3  flip audit per mechanism  (questions 1-4)')
    print('=' * 78)
    print(f'  {"mech":6s} {"flips":>6s} {"W->R":>6s} {"R->W":>6s} {"R->R":>6s} {"W->W":>6s}'
          f' {"precision":>10s} {"net/100":>8s}')
    stats = {}
    for mech in ('H', 'E'):
        f = m[(m.mech == mech) & m.flip]
        wr = int(((f.c_c == 0) & (f.c_a == 1)).sum())
        rw = int(((f.c_c == 1) & (f.c_a == 0)).sum())
        rr = int(((f.c_c == 1) & (f.c_a == 1)).sum())
        ww = int(((f.c_c == 0) & (f.c_a == 0)).sum())
        prec = wr / max(1, wr + rw)
        net = 100.0 * (wr - rw) / max(1, len(f))
        stats[mech] = dict(n=len(f), wr=wr, rw=rw, prec=prec, net=net)
        print(f'  {mech:6s} {len(f):6d} {wr:6d} {rw:6d} {rr:6d} {ww:6d} {prec:10.3f} {net:8.1f}')

    print('\n' + '=' * 78)
    print('PHASE 4  margin distribution of successful vs failed flips  (question 5)')
    print('=' * 78)
    for mech in ('H', 'E'):
        f = m[(m.mech == mech) & m.flip].copy()
        # a non-finite margin means no alternative assignment was even feasible for that
        # clip, i.e. maximum confidence.  Map it to the top of the scale, do not drop it.
        f['mg_a'] = f.mg_a.fillna(np.inf)
        BIG = 1e6
        f['mg_s'] = np.where(np.isfinite(f.mg_a), f.mg_a, BIG)
        n_inf = int((~np.isfinite(f.mg_a)).sum())
        print(f'  mech {mech}: {len(f)} flips, of which {n_inf} have no feasible '
              f'alternative (margin = inf, treated as maximum confidence)')
        if not len(f):
            continue
        good = f[(f.c_c == 0) & (f.c_a == 1)].mg_a
        bad = f[(f.c_c == 1) & (f.c_a == 0)].mg_a
        print(f'  mech {mech}:  W->R n={len(good)} median={good.median():.3f} '
              f'q25={good.quantile(.25):.3f} q75={good.quantile(.75):.3f}')
        print(f'           R->W n={len(bad)} median={bad.median():.3f} '
              f'q25={bad.quantile(.25):.3f} q75={bad.quantile(.75):.3f}'
              if len(bad) else '           R->W n=0')
        # precision as a function of a margin floor -> defines the reliable region
        print(f'  {"margin>=":>12s} {"flips":>6s} {"W->R":>5s} {"R->W":>5s} {"prec":>6s} {"net":>5s}')
        qs = [-1e9, 0.0] + list(np.quantile(f.mg_s, [.2, .4, .5, .6, .8]))
        for th in sorted(set(round(q, 4) for q in qs)):
            s = f[f.mg_s >= th]
            wr = int(((s.c_c == 0) & (s.c_a == 1)).sum())
            rw = int(((s.c_c == 1) & (s.c_a == 0)).sum())
            print(f'  {th:12.3f} {len(s):6d} {wr:5d} {rw:5d} '
                  f'{wr/max(1,wr+rw):6.3f} {wr-rw:+5d}')
        stats[mech]['tab'] = f

    print('\n' + '=' * 78)
    print('PHASE 5  the 15 test flips, classified  (question 6)')
    print('=' * 78)
    te = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))[['qa_id', 'source', 'category']]
    a = pd.read_csv(os.path.join(ROOT, 'submissions/submission_090643_regen.csv'))
    b = pd.read_csv(os.path.join(ROOT, 'submissions/submission_final_assault.csv'))
    t = te.merge(a, on='qa_id').merge(b, on='qa_id', suffixes=('_c', '_n'))
    t['flip'] = t.prediction_c != t.prediction_n
    t['mech'] = np.where(t.category == 'emotion', 'E',
                         np.where(t.source == 'HARn', 'H', '-'))
    # test-side margins: re-run the assault config on the real test view and read the
    # margins the solvers recorded (inference only)
    tmp = os.path.join(ROOT, 'champ', 'test_margins.json')
    if os.path.exists(tmp):
        tm = json.load(open(tmp))
    else:
        import importlib
        os.environ['CHAMP_EMO_DINO'] = '1'; os.environ['CHAMP_W_HDINO'] = '0.3'
        for mod in ('core', 'pool', 'harn', 'emopair', 'pipeline'):
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])
        import core, pipeline as P2
        from core import load_all as la2, real_test_view
        core.MARGIN.clear()
        tr2, te2, meta2 = la2()
        P2.caches(meta2)
        ctx2 = P2.fit_all(tr2, meta2, hold_users=[])
        P2.solve(real_test_view(te2), ctx2, 'test')
        tm = {k: (None if not np.isfinite(v) else float(v))
              for k, v in core.MARGIN.items()}
        json.dump(tm, open(tmp, 'w'))
        print(f'  computed {len(tm)} test margins')
    tm = {k: (np.inf if v is None else v) for k, v in tm.items()}
    t['mg'] = [tm.get(q, np.nan) for q in t.qa_id]
    F = t[t.flip].copy()

    # thresholds: the smallest margin floor at which OOF precision >= .60 (A) and >= .50 (B)
    thr = {}
    for mech in ('H', 'E'):
        f = stats[mech].get('tab')
        thr[mech] = (np.inf, np.inf)
        if f is None or not len(f):
            continue
        cand = sorted(set(np.round(f.mg_s, 4)))
        tA = tB = np.inf
        for th in cand:
            s = f[f.mg_s >= th]
            wr = int(((s.c_c == 0) & (s.c_a == 1)).sum())
            rw = int(((s.c_c == 1) & (s.c_a == 0)).sum())
            p = wr / max(1, wr + rw)
            if p >= 0.60 and wr + rw >= 5 and tA == np.inf:
                tA = th
            if p >= 0.50 and wr + rw >= 5 and tB == np.inf:
                tB = th
        thr[mech] = (tA, tB)
        print(f'  mech {mech}: OOF-derived margin floors  A(prec>=.60)={tA:.3f}  '
              f'B(prec>=.50)={tB:.3f}')

    def cls(row):
        tA, tB = thr.get(row.mech, (np.inf, np.inf))
        v = 1e6 if not np.isfinite(row.mg) else row.mg
        if v >= tA:
            return 'A'
        if v >= tB:
            return 'B'
        return 'C'
    F['grade'] = [cls(r) for r in F.itertuples()]
    print()
    print(F[['qa_id', 'source', 'category', 'mech', 'prediction_c', 'prediction_n',
             'mg', 'grade']].to_string(index=False))
    print('\n  grade counts:', F.grade.value_counts().to_dict())

    print('\n' + '=' * 78)
    print('PHASE 6  selective gating evaluated on OOF')
    print('=' * 78)
    for mech in ('H', 'E'):
        f = stats[mech].get('tab')
        if f is None:
            continue
        tA, tB = thr[mech]
        for name, th in (('all flips', -1e9), ('A+B gate', tB), ('A only', tA)):
            s = f[f.mg_s >= th]
            wr = int(((s.c_c == 0) & (s.c_a == 1)).sum())
            rw = int(((s.c_c == 1) & (s.c_a == 0)).sum())
            print(f'  {mech}  {name:10s} kept {len(s):4d} flips  W->R {wr:3d}  R->W {rw:3d}'
                  f'  net {wr-rw:+4d}')
    # OOF totals under each gating policy
    print()
    base = int(m.c_c.sum())
    for name in ('champion', 'all flips', 'A+B gate', 'A only'):
        tot = 0
        for _, r in m.iterrows():
            if not r.flip:
                tot += r.c_c; continue
            tA, tB = thr.get(r.mech, (np.inf, np.inf))
            v = 1e6 if not np.isfinite(r.mg_a) else r.mg_a
            take = (name == 'all flips'
                    or (name == 'A+B gate' and v >= tB)
                    or (name == 'A only' and v >= tA))
            tot += r.c_a if take else r.c_c
        print(f'  OOF total, {name:10s}: {tot}/{len(m)} = {tot/len(m):.4f}  '
              f'({tot-base:+d} vs champion)')
    F.to_csv(os.path.join(ROOT, 'champ', 'test_flip_grades.csv'), index=False)
    json.dump({k: [float(v[0]), float(v[1])] for k, v in thr.items()},
              open(os.path.join(ROOT, 'champ', 'flip_thresholds.json'), 'w'))
    print('\n  wrote champ/test_flip_grades.csv and champ/flip_thresholds.json')


if __name__ == '__main__':
    main()
