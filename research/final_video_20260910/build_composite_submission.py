"""Generate composite test submission from champion pipeline + GPU dense + H1.

Uses:
- Champion pipeline with CHAMP_LOGITS=dense_logits_full40.npz for test predictions
  (the GPU-trained TCN produces better per-frame action logits for sequence ordering)
- H1 object mapping hybrid applied post-hoc

The submission is built from the FULL training set (no held-out fold) for test.
"""
import os
import sys
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'champ'))

# Use GPU-trained logits for test prediction
os.environ['CHAMP_LOGITS'] = 'dense_logits_full40.npz'

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import re
import csv

from core import load_all, real_test_view, fit_group_model, opts
import pipeline as P


def norm(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def build_h1_map(tr):
    """Build action->object mapping from ALL training co-occurrence."""
    harn = tr[tr.source == 'HARn']
    by_ut = {}
    for _, r in harn.iterrows():
        by_ut.setdefault((r.user, r.trial), []).append(r)
    
    act2obj = {}
    for (u, t), rs in by_ut.items():
        singles = [r for r in rs if r.category == 'single']
        objects = [r for r in rs if r.category == 'object_interaction']
        if len(singles) == 1 and len(objects) == 1:
            s = singles[0]
            o = objects[0]
            # Use the answer text
            act_text = norm(s[s['answer']])
            obj_text = norm(o[o['answer']])
            act2obj.setdefault(act_text, Counter())[obj_text] += 1
    return act2obj


def main():
    tr, te, meta = load_all()
    P.caches(meta)
    
    # Fit on ALL training data
    all_users = sorted(tr.user.dropna().unique())
    ctx = P.fit_all(tr, meta, hold_users=set(), split_train='test')
    
    # Build test view and solve
    vis = real_test_view(te)
    pred, blk, pool, diag = P.solve(vis, ctx, 'test')
    
    print(f'Test predictions: {len(pred)}')
    print(f'Diagnostics: {dict(diag)}')
    
    # H1 override for test object_interaction rows
    h1_map = build_h1_map(tr)
    test_harn = vis[(vis.source == 'HARn')]
    by_ut_test = {}
    for _, r in test_harn.iterrows():
        by_ut_test.setdefault((r.user, r.trial), []).append(r)
    
    n_h1 = 0
    for (u, t), rs in by_ut_test.items():
        singles = [r for r in rs if r['category'] == 'single']
        objects = [r for r in rs if r['category'] == 'object_interaction']
        if len(singles) == 1 and len(objects) == 1:
            s = singles[0]
            o = objects[0]
            s_pred = pred.get(s.qa_id)
            if s_pred and s_pred in 'ABCD':
                act_text = norm(s[s_pred])
                if act_text in h1_map:
                    best_obj = h1_map[act_text].most_common(1)[0][0]
                    oo = [norm(o[L]) for L in 'ABCD']
                    if best_obj in oo:
                        old = pred.get(o.qa_id)
                        new = 'ABCD'[oo.index(best_obj)]
                        if old != new:
                            pred[o.qa_id] = new
                            n_h1 += 1
    print(f'H1 overrides applied: {n_h1}')
    
    # Load champion submission as base
    champ = pd.read_csv(os.path.join(ROOT, 'submissions/submission_097076_332of342_CHAMPION.csv'))
    print(f'Champion rows: {len(champ)}')
    
    # Build submission
    sub = champ.copy()
    applied = 0
    fallback = 0
    for idx, row in sub.iterrows():
        qa_id = row['qa_id']
        p = pred.get(qa_id)
        if p is not None:
            sub.at[idx, 'prediction'] = p
            applied += 1
        else:
            fallback += 1
    
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       'submission_composite_v1.csv')
    sub.to_csv(out, index=False)
    
    # Verify
    assert len(sub) == 682, f'Expected 682 rows, got {len(sub)}'
    assert sub.qa_id.nunique() == 682
    
    # Diff vs champion
    diffs = []
    for _, (qa, pred_new) in sub[['qa_id', 'prediction']].iterrows():
        champ_pred = champ[champ.qa_id == qa]['prediction'].values[0]
        if str(pred_new) != str(champ_pred):
            diffs.append((qa, champ_pred, pred_new))
    
    print(f'\nApplied: {applied}, Fallback (kept champion): {fallback}')
    print(f'Diffs vs champion: {len(diffs)}')
    for qa, old, new in sorted(diffs)[:20]:
        print(f'  {qa}: {old} -> {new}')
    if len(diffs) > 20:
        print(f'  ... and {len(diffs) - 20} more')
    
    # SHA-256
    with open(out, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f'\nSHA-256: {sha}')
    print(f'Output: {out}')


if __name__ == '__main__':
    main()
