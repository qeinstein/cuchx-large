"""Rerun exact-template mechanisms (OBJTMPL + USERTMPL) from a pipeline-style base.

CPU-only, CSV-only. Reuses research/exact_visible_template_audit_20260905.py
(test_audit / signature / to_letters) and research/evaluate_user_template_pairs_20260904.py
(donor-block machinery), with TEST_BLOCKS reconstructed from
champ/test_blocks_repaired.json (the historical test_pool_snapshot.pkl is missing).

Historical gates (faithful):
  OBJTMPL: unanimous semantic template (purity==1.0), object_interaction only,
    support>=1. Expected: test_0524 A->B, test_0530 A->D (directly; the historical
    route went A->B via W2 then B->D via template).
  USERTMPL: exact whole-cohort visible-signature match to user20/user21 (segments
    (0,14),(14,28)); per-trial manner transfer when trial counts match; sequence
    transfer via predict_sequence_from_donor; single/combination via ordered
    semantic containment (73/73 OOF gate). Expected: test_0385 ->A, test_0397 ->D.
    (test_0397 is outside the 55: base already D, expect no-fire.)

Usage: PYTHONHASHSEED=0 python3 research/t2_valid/mech_templates_rerun.py --base submission_final.csv
"""
import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research'))

import exact_visible_template_audit_20260905 as T  # noqa: E402
import evaluate_user_template_pairs_20260904 as E  # noqa: E402


def run_objtmpl(tr, te, base_pred, champ_pred):
    cur = pd.DataFrame([dict(qa_id=q, prediction=p) for q, p in base_pred.items()])
    t = T.test_audit(tr, te, cur)
    g = t[(t.purity == 1.0) & (t.category == 'object_interaction')
          & (t.changed == 1) & t.prediction.notna()]
    out = {}
    print(f'OBJTMPL unanimous object disagreements: {len(g)}')
    for r in g.itertuples():
        flag = 'MATCH' if str(champ_pred.get(r.qa_id)) == str(r.prediction) else 'DIFF '
        print(f'  {flag} {r.qa_id}: {r.current} -> {r.prediction} '
              f'(champ {champ_pred.get(r.qa_id)}) support={r.support} '
              f'semantic={r.semantic}')
        out[r.qa_id] = str(r.prediction)
    return out, g


def run_usertmpl(te, base_pred, champ_pred):
    # E.TEST_BLOCKS / E.TEST_SIG self-reconstruct from the tracked atlas
    # (multi_pair_atlas_20260904/block_atlas_test.csv); no override needed.
    blocks = E.TEST_BLOCKS
    print(f'user-template test blocks: {len(blocks)}')
    out, rows = {}, []
    for lo, hi, donor in ((0, 14, 'user20'), (14, 28, 'user21')):
        donor_blocks = E.USER_BLOCKS[donor]
        for offset, j in enumerate(range(lo, hi)):
            if offset >= len(donor_blocks) or j >= len(blocks):
                continue
            ids = blocks[j]
            db = donor_blocks[offset]
            try:
                tsig = E.TEST_SIG[j]
            except Exception:
                continue
            dsig = E.visible_signature(db)
            sig_ok = (tsig == dsig)
            tg = te[(te.idx.isin(ids)) & (te.category == 'emotion')].sort_values('idx')
            labels = E.source_labels(db)
            if sig_ok and len(tg) == len(labels):
                for i, (_, r) in enumerate(tg.iterrows()):
                    new = E.exact_letter(r, labels[i])
                    if new is None or str(base_pred[r.qa_id]) == new:
                        continue
                    out[r.qa_id] = new
                    rows.append(dict(qa_id=r.qa_id, test_block=j, donor=donor,
                                     old=base_pred[r.qa_id], new=new,
                                     donor_label=labels[i]))
    print(f'USERTMPL emotion transfers: {len(rows)} (signature-exact only)')
    for r in rows:
        flag = 'MATCH' if str(champ_pred.get(r["qa_id"])) == r['new'] else 'DIFF '
        print(f'  {flag} {r["qa_id"]}: {r["old"]} -> {r["new"]} '
              f'(champ {champ_pred.get(r["qa_id"])}) donor={r["donor"]} '
              f'label={r["donor_label"]}')
    return out, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
    tr['user'] = tr.path.str.extract(r'(user\d+)')
    te['idx'] = te.path.str.extract(r'LM_test_(\d+)').astype(int)
    base = pd.read_csv(args.base)
    base_pred = dict(zip(base.qa_id, base['prediction'].astype(str)))
    champ = pd.read_csv(os.path.join(ROOT, 'submissions/submission_097076_332of342_CHAMPION.csv'))
    champ_pred = dict(zip(champ.qa_id, champ['prediction'].astype(str)))

    all_new = {}
    o_obj, g_obj = run_objtmpl(tr, te, base_pred, champ_pred)
    all_new.update(o_obj)
    o_user, g_user = run_usertmpl(te, base_pred, champ_pred)
    for q, v in o_user.items():
        if q in all_new and all_new[q] != v:
            print(f'  CONFLICT {q}: objtmpl={all_new[q]} usertmpl={v} (keep objtmpl)')
        else:
            all_new[q] = v
    n_match = sum(1 for q, v in all_new.items() if str(champ_pred.get(q)) == v)
    print(f'total template flips: {len(all_new)}; matching champion: {n_match}')

    tag = os.path.basename(args.base).replace('.csv', '')
    out_p = args.out or os.path.join(ROOT, 'research', 't2_valid',
                                     f'templates_from_{tag}.csv')
    pred = dict(base_pred)
    pred.update(all_new)
    pd.DataFrame([dict(qa_id=q, prediction=pred[q])
                  for q in base.qa_id]).to_csv(out_p, index=False)
    g_obj.to_csv(os.path.join(ROOT, 'research', 't2_valid',
                              f'templates_from_{tag}.obj_audit.csv'), index=False)
    if len(g_user):
        g_user.to_csv(os.path.join(ROOT, 'research', 't2_valid',
                                   f'templates_from_{tag}.user_audit.csv'), index=False)
    print('wrote', out_p)


if __name__ == '__main__':
    main()
