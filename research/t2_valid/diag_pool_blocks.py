"""One-off diagnostics (T2 valid): greedy-block counts per OOF fold + struct-pool
letters on REPAIRED test blocks (isolates T's repair effect from dense)."""
import json, os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, 'champ'))
os.environ['CHAMP_POOL_FEATS'] = 'struct'
from core import load_all, make_pseudo, real_test_view, fit_group_model, infer_blocks
from pseudotest import folds
import pool as PL
import pipeline as PIPE
tr, te, meta = load_all()
users = sorted(tr.user.dropna().unique())
for fi, hold in enumerate(folds(users, 5)):
    trn = tr[tr.user.isin([u for u in users if u not in hold])]
    gm = fit_group_model(trn)
    vis, key, aux = make_pseudo(tr, meta, hold, seed=100 + fi)
    blocks = infer_blocks(vis, gm)
    n_greedy = n_comp = 0
    for blk in blocks:
        uni, rows_a, qs, forced = PL.candidate_evidence(vis, list(blk), 'oof', {})
        _free, comps, _fixed = PL._components(qs, uni, forced)
        n_comp += len(comps)
        if any(len(a) > 20 for a, _ in comps):
            n_greedy += 1
    print(f'fold {fi}: {len(blocks)} blocks, {n_comp} comps, greedy {n_greedy}', flush=True)
# repaired-blocks test letters
PL.fit_cooc(tr)
tpool = PIPE.true_pool(tr)
bbu = PIPE.training_blocks(tr, sorted(tr.user.dropna().unique()), tpool)
clf, cols = PL.fit_pool_model(tr, None, bbu, {})
scorer = PL.make_scorer(clf, cols)
print('scorer fit done', flush=True)
vis = real_test_view(te)
blocks = [tuple(int(x) for x in b) for b in json.load(open('champ/test_blocks_repaired.json'))]
pred = {}
for blk in blocks:
    pp, _bd = PL.solve_block(vis, list(blk), {}, scorer, 'test')
    pred.update(pp)
pipe = dict(zip(pd.read_csv('submissions/submission_final.csv').qa_id, pd.read_csv('submissions/submission_final.csv')['prediction'].astype(str)))
champ = dict(zip(pd.read_csv('submissions/submission_097076_332of342_CHAMPION.csv').qa_id, pd.read_csv('submissions/submission_097076_332of342_CHAMPION.csv')['prediction'].astype(str)))
for q in ['test_0137', 'test_0206', 'test_0289']:
    print(f'{q}: pipeline={pipe[q]} champion={champ[q]} struct+repaired={pred.get(q)}', flush=True)
