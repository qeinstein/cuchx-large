"""Rerun mechanism W (object<->single consistency) from a pipeline-style base.

Thin CPU-only wrapper around objlab/object_rule.py (CSV-only: training_qa.csv,
test_qa.csv, base predictions). Historical OOF: 7 flips, 7 W->R, 0 R->W, prec 1.000
(objlab/object_rule.py docstring). Expected champion rows: test_0480, test_0525,
test_0528, test_0530 (->B interim; champion D comes from OBJTMPL), test_0533,
test_0534.

Usage: PYTHONHASHSEED=0 python3 research/t2_valid/mech_w_rerun.py --base submission_final.csv
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'objlab'))
sys.path.insert(0, os.path.join(ROOT, 'champ'))

from object_rule import fit, apply_test  # noqa: E402
from core import load_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tr, te, _meta = load_all()
    base = pd.read_csv(args.base)
    base_pred = dict(zip(base.qa_id, base['prediction'].astype(str)))
    champ = pd.read_csv(os.path.join(ROOT, 'submissions/submission_097076_332of342_CHAMPION.csv'))
    champ_pred = dict(zip(champ.qa_id, champ['prediction'].astype(str)))

    model = fit(tr)
    d = apply_test(model, te, base_pred)
    print(f'W overrides from {args.base}: {len(d)}')
    match = 0
    for r in d.itertuples():
        flag = 'MATCH' if str(champ_pred.get(r.qa_id)) == str(r.prediction) else 'DIFF '
        if flag == 'MATCH':
            match += 1
        print(f'  {flag} {r.qa_id}: {r.champ} -> {r.prediction} '
              f'(champ {champ_pred.get(r.qa_id)}) [{r.rule}] {r.why[:100]}')
    print(f'matching champion: {match}/{len(d)}')

    tag = os.path.basename(args.base).replace('.csv', '')
    out_p = args.out or os.path.join(ROOT, 'research', 't2_valid', f'w_from_{tag}.csv')
    pred = dict(base_pred)
    for r in d.itertuples():
        pred[r.qa_id] = str(r.prediction)
    pd.DataFrame([dict(qa_id=q, prediction=pred[q])
                  for q in base.qa_id]).to_csv(out_p, index=False)
    d.to_csv(os.path.join(ROOT, 'research', 't2_valid', f'w_from_{tag}.audit.csv'),
             index=False)
    print('wrote', out_p)


if __name__ == '__main__':
    main()
