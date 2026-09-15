"""Apply only stable pairwise-sequence test flips to the frozen champion."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROVEN_SEQUENCE = {'test_0330', 'test_0335', 'test_0358', 'test_0643'}


def main(min_stability=4):
    base_path = os.path.join(ROOT, 'submissions/submission_097076_332of342_CHAMPION.csv')
    stability_path = os.path.join(ROOT, 'research/final_video_20260910',
                                  'seqpair_test_stability.csv')
    out = pd.read_csv(base_path)
    audit = pd.read_csv(stability_path)
    if not set(audit.qa_id).issubset(set(out.qa_id)):
        raise ValueError('stability rows are missing from candidate')

    selected = audit[(audit.stability >= min_stability) &
                     (audit.full == audit.majority) &
                     (audit.majority != audit.champion) &
                     (~audit.qa_id.isin(PROVEN_SEQUENCE))]
    selected = selected[['qa_id', 'champion', 'majority', 'stability']]
    for r in selected.itertuples():
        mask = out.qa_id.eq(r.qa_id)
        if not mask.any() or str(out.loc[mask, 'prediction'].iloc[0]) != str(r.champion):
            raise ValueError('unexpected base value for ' + r.qa_id)
        out.loc[mask, 'prediction'] = r.majority

    path = os.path.join(ROOT, os.environ.get(
        'CHAMP_SEQPAIR_STABLE_OUT',
        'research/final_video_20260910/submission_seqpair_stable_v1.csv'))
    out.to_csv(path, index=False)
    print('minimum stability:', min_stability)
    print('selected flips:', len(selected))
    for r in selected.itertuples():
        print(' ', r.qa_id, r.champion, '->', r.majority,
              f'({r.stability}/5 folds)')
    print('written', path)
    return out


if __name__ == '__main__':
    main(int(os.environ.get('CHAMP_SEQPAIR_MIN_STABILITY', '4')))
