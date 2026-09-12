"""Audit whether pairwise sequence test predictions are stable across fit folds."""
import os
import re
import sys
import itertools

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seqpair as SP
from core import load_all, opts
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clip_name(path):
    m = re.search(r'LM_test_(\d+)', str(path))
    return 'LM_test_%04d' % int(m.group(1)) if m else None


def main(nfold=5):
    tr, te, meta = load_all()
    segs = SP.build_segment_data(meta)
    users = sorted(tr.user.dropna().unique())
    q = te[te.category == 'sequence'].copy()
    champion = pd.read_csv(os.path.join(ROOT, 'submission_097076_332of342_CHAMPION.csv'))
    champion = dict(zip(champion.qa_id, champion.prediction))
    full_path = os.path.join(
        ROOT, os.environ.get('CHAMP_SEQPAIR_FULL',
                             'research/final_video_20260910/submission_seqpair_protected_v1.csv'))
    full = pd.read_csv(full_path).set_index('qa_id').prediction.to_dict()
    votes = {r.qa_id: [] for r in q.itertuples()}

    for fi, hold in enumerate(folds(users, nfold)):
        fit_users = [u for u in users if u not in hold]
        Xd, y, src, _, _ = SP.build_training_pairs(tr, meta, segs, fit_users)
        clf, cols = SP.fit_pair_model(Xd, y)
        dpri, apri = SP.fit_priors(segs, [k for k in segs if k[0] in fit_users])
        for r in q.itertuples():
            oo = opts(r)
            clip = clip_name(r.path)
            prof = SP.clip_profiles('test', clip) if clip else None
            if prof is None or not all(o in SP.OPT2CLS for o in oo):
                votes[r.qa_id].append(None)
                continue
            cls4 = [SP.OPT2CLS[o] for o in oo]
            perm, _, _ = SP.decode(prof, cls4, clf, cols, dpri, apri)
            votes[r.qa_id].append(''.join('ABCD'[p] for p in perm))
        print('fold', fi, 'pairs', len(Xd), flush=True)

    rows = []
    for r in q.itertuples():
        vv = [v for v in votes[r.qa_id] if v is not None]
        counts = pd.Series(vv).value_counts()
        majority = counts.index[0] if len(counts) else champion[r.qa_id]
        rows.append(dict(qa_id=r.qa_id, champion=champion[r.qa_id],
                         full=full[r.qa_id], majority=majority,
                         votes='|'.join(vv),
                         stability=int(counts.iloc[0]) if len(counts) else 0))
    out = pd.DataFrame(rows)
    path = os.path.join(ROOT, 'research/final_video_20260910',
                        'seqpair_test_stability.csv')
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print('written', path)


if __name__ == '__main__':
    main()
