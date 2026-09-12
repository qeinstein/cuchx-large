"""OOF audit of the direct clip-action presence head."""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all
from pseudotest import folds
import clip_action as CA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGITS = os.environ.get('CHAMP_LOGITS', 'dense_logits_full40.npz')
BASE = os.environ.get(
    'CHAMP_OOF_BASE',
    os.path.join(ROOT, 'research', 'final_video_20260910', 'oof_pipeline_base.csv'))


def statcache(paths):
    # The compact dense OOF audits survive even when the large raw-logit cache
    # has been cleaned.  They contain the clip/action confidence statistics this
    # head needs and were generated strictly out of fold.
    out = {}
    for path in sorted(os.path.join(ROOT, 'research', 'final_video_20260910',
                                    f'dense_oof_f{i}.npy') for i in range(5)):
        if os.path.exists(path):
            out.update(np.load(path, allow_pickle=True).item())
    return out


def run(nfold=5):
    tr, _te, _meta = load_all()
    users = sorted(tr.user.dropna().unique())
    base = pd.read_csv(BASE).set_index('qa_id')
    thresholds = [0.35, 0.45, 0.55, 0.65]
    results = {t: [] for t in thresholds}
    sc = statcache(tr[tr.source == 'HAU'].path.unique())
    for fi, hold in enumerate(folds(users, nfold)):
        trn = tr[~tr.user.isin(hold)]
        clf, cols = CA.fit(trn, sc)
        ev = tr[(tr.user.isin(hold)) & (tr.source == 'HAU') &
                tr.category.isin(['single', 'multi', 'combination'])]
        for r in ev.itertuples():
            stats = sc.get(r.path)
            if stats is None:
                continue
            truth = str(r.answer)
            old = str(base.loc[r.qa_id, 'pred'])
            for threshold in thresholds:
                guess = CA.predict_question(r, stats, clf, cols, threshold)
                results[threshold].append(dict(fold=fi, qa_id=r.qa_id,
                    category=r.category, truth=truth, baseline=old,
                    prediction=guess, new_ok=int(guess == truth),
                    base_ok=int(old == truth)))
        print(f'  fold {fi} complete', flush=True)
    for threshold, rows in results.items():
        d = pd.DataFrame(rows)
        out = os.environ.get('CHAMP_CLIP_ACTION_OOF',
                             os.path.join(ROOT, 'research', 'clip_action_oof.csv'))
        if threshold != thresholds[0]:
            out = out.replace('.csv', f'_t{threshold:.2f}.csv')
        d.to_csv(out, index=False)
        print(f'\nthreshold={threshold:.2f}')
        for cat in ['single', 'multi', 'combination']:
            x = d[d.category == cat]
            print(f'  {cat:12s} new={x.new_ok.sum():4d}/{len(x):4d} '
                  f'base={x.base_ok.sum():4d}/{len(x):4d} delta={x.new_ok.sum()-x.base_ok.sum():+d}')
        print(f'  total delta={d.new_ok.sum()-d.base_ok.sum():+d} '
              f'W2R={((d.base_ok==0)&(d.new_ok==1)).sum()} '
              f'R2W={((d.base_ok==1)&(d.new_ok==0)).sum()} wrote={out}')


if __name__ == '__main__':
    run()
