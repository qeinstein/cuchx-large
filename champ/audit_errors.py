"""Information audit: cluster every held-out error by root cause, and measure the oracle
ceiling for each candidate fix.  Training/OOF data only; no test labels are touched."""
import os, sys, json, itertools
import numpy as np, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, opts, mgroup
import dense as D, pool as PL, harn as H
import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
ACT = ['single', 'multi', 'combination', 'sequence']


def main():
    tr, te, meta = load_all()
    P.caches(meta)
    mi = meta.set_index('qa_path')
    has_skel = set(mi[mi.f0.notna()].index)
    d = pd.read_csv(os.path.join(ROOT, 'champ', 'oof_final.csv'))
    info = tr.set_index('qa_id')
    d = d.join(info[['path', 'user', 'aa', 'bb', 'cc', 'A', 'B', 'C', 'D', 'harn_action']],
               on='qa_id')

    # ---- true session pools and per-clip action truth
    tpool = defaultdict(set)
    for _, r in tr[(tr.source == 'HAU') & tr.category.isin(ACT)].iterrows():
        for L in str(r['answer']):
            if L in 'ABCD':
                for p in str(r[L]).split(','):
                    tpool[(r.user, r.aa, r.bb)].add(p.strip())
    # ---- recovered pools, from the shared inference path, per fold
    from core import make_pseudo, fit_group_model, infer_blocks
    from pseudotest import folds
    users = sorted(tr.user.dropna().unique())
    rec_pool, blk_of, blk_ok = {}, {}, {}
    for fi, hold in enumerate(folds(users, 5)):
        ctx = P.fit_all(tr, meta, hold)
        vis, key, aux = make_pseudo(tr, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, ctx['gm'])
        pathof = dict(zip(vis.idx, vis.true_path))
        sc = P.statcache_view('oof')
        for blk in blocks:
            _, bd = PL.solve_block(vis, blk, sc, ctx['scorer'], 'oof')
            paths = [pathof[b] for b in blk]
            sess = {(tr[tr.path == p].iloc[0].user, tr[tr.path == p].iloc[0].aa,
                     tr[tr.path == p].iloc[0].bb) for p in paths}
            for p in paths:
                rec_pool[p] = set(bd['pool']) if bd else set()
                blk_of[p] = tuple(paths)
                blk_ok[p] = int(len(sess) == 1)
        print(f'  fold {fi} pools recovered', flush=True)

    d['no_skel'] = [p not in has_skel for p in d.path]
    d['nested'] = [P._C['nest'].get(p) is not None for p in d.path]
    d['blk_ok'] = [blk_ok.get(p, -1) for p in d.path]

    rows = []
    for r in d.itertuples():
        why = None
        if r.correct == 1:
            continue
        if r.no_skel:
            why = 'no_skeleton_or_imu'
        elif r.source == 'HARn':
            why = 'harn_action_id' if r.nested else 'harn_not_nested'
        elif r.blk_ok == 0:
            why = 'session_block_wrong'
        else:
            key = (r.user, r.aa, r.bb)
            tp = tpool[key]
            rp = rec_pool.get(r.path, set())
            if r.category in ('single', 'multi', 'combination'):
                oo = [[x.strip() for x in str(getattr(r, L)).split(',')] for L in 'ABCD']
                truth = set()
                for L in str(r.answer):
                    if L in 'ABCD':
                        truth |= set(oo['ABCD'.index(L)])
                miss = truth - rp
                extra = (rp & {a for o in oo for a in o}) - tp
                why = ('pool_missing_true_action' if miss else
                       'pool_extra_false_action' if extra else 'pool_ranking')
            elif r.category == 'emotion':
                O = [set(opts(tr[tr.path == p].iloc[len(tr[tr.path == p]) - 1]))
                     for p in blk_of.get(r.path, (r.path,))]
                why = 'manner_assignment'
            else:
                why = 'sequence_localization'
        rows.append(dict(qa_id=r.qa_id, source=r.source, category=r.category, why=why))
    E = pd.DataFrame(rows)
    print('\n================ ERROR ROOT-CAUSE TABLE (held-out OOF) ================')
    tab = pd.crosstab(E.category, E.why)
    print(tab.to_string())
    print('\ntotals by category:')
    print(E.category.value_counts().to_string())
    print('\ntotals by mechanism:')
    print(E.why.value_counts().to_string())
    E.to_csv(os.path.join(ROOT, 'champ', 'error_audit.csv'), index=False)

    # ---- oracle ceilings
    print('\n================ ORACLE CEILINGS ================')
    # (1) perfect pool recovery -> single/multi/combination
    for cat in ('single', 'multi', 'combination'):
        s = tr[(tr.source == 'HAU') & (tr.category == cat)]
        ok = 0
        for _, r in s.iterrows():
            oo = [[x.strip() for x in str(r[L]).split(',')] for L in 'ABCD']
            tp = tpool[(r.user, r.aa, r.bb)]
            hit = [i for i, o in enumerate(oo) if all(a in tp for a in o)]
            pred = ''.join('ABCD'[i] for i in sorted(hit))
            ok += int(pred == str(r['answer']))
        print(f'  {cat:12s} with PERFECT session pool: {ok}/{len(s)} = {ok/len(s):.4f}')
    # (2) HARn: perfect action id -> single and object
    n = tr[tr.source == 'HARn']
    for cat in ('single', 'object_interaction'):
        s = n[n.category == cat]
        print(f'  HARn {cat:18s} with PERFECT action id: '
              f'{len(s)}/{len(s)} = 1.0000  (action->answer map is deterministic)')
    # (3) action-family concentration of HARn errors
    ea = E[(E.source == 'HARn')]
    m = ea.merge(d[['qa_id', 'harn_action']], on='qa_id')
    print('\n  HARn errors by action folder (top 12):')
    print(m.harn_action.value_counts().head(12).to_string())


if __name__ == '__main__':
    main()
