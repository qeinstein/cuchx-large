"""Composite OOF evaluation: pipeline + GPU dense seq + H1 object hybrid.

Merges three validated improvements over the pipeline baseline:
1. GPU-trained dense TCN sequence ordering (55.7% vs 37.7%)
2. H1 action->object mapping hybrid (+9/-0 on object_interaction)

Grouped folds match oof_driver (pseudotest.folds(users, 5, seed=7)).
All improvements are locally validated before inclusion.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'champ'))

# Must set CHAMP_LOGITS before importing pipeline so it finds the logits
os.environ['CHAMP_LOGITS'] = 'dense_logits_screen1_bucket.npz'

import numpy as np
import pandas as pd

from core import load_all, make_pseudo, opts
from pseudotest import folds
import pipeline as P


def gpu_seq_preds():
    """Load GPU OOF sequence predictions from the per-fold .npy files."""
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'training_qa.csv'),
                                    encoding='utf-8-sig')))
    seq = {r['path']: r for r in rows
           if r['path'].startswith('HAU/') and r['category'] == 'sequence'}
    preds = {}  # (fold, qa_id) -> pred_letter_string
    for fi in range(5):
        fp = os.path.join(ROOT, 'research', 'final_video_20260910',
                          f'dense_oof_f{fi}.npy')
        if not os.path.exists(fp):
            print(f'  WARNING: {fp} missing, skipping fold {fi}')
            continue
        d = np.load(fp, allow_pickle=True).item()
        for qa_path, o in d.items():
            r = seq.get(qa_path)
            if r is None:
                continue
            oo = [r[k].strip() for k in 'ABCD']
            order = sorted(range(4),
                           key=lambda i: o[oo[i]]['centroid']
                           if oo[i] in o else 9.0)
            pr = ''.join('ABCD'[i] for i in order)
            preds[(fi, r['qa_id'])] = pr
    print(f'  GPU seq OOF predictions loaded: {len(preds)}')
    return preds


def h1_object_map(tr, hold_users):
    """Build action->object mapping from outer-train co-occurrence (H1 hybrid)."""
    from collections import Counter
    import re

    def norm(s):
        return re.sub(r'\s+', ' ', s.strip().lower())

    # Find dual clips: same user/trial with both single and object_interaction
    harn = tr[tr.source == 'HARn']
    by_ut = {}
    for _, r in harn.iterrows():
        if r.user in hold_users:
            continue
        by_ut.setdefault((r.user, r.trial), []).append(r)
    
    action_to_object = {}  # norm(action_text) -> Counter of norm(object_text)
    for (u, t), rs in by_ut.items():
        singles = [r for r in rs if r.category == 'single']
        objects = [r for r in rs if r.category == 'object_interaction']
        if len(singles) == 1 and len(objects) == 1:
            s = singles[0]
            o = objects[0]
            act = norm(str(s['answer']))
            obj = norm(str(o['answer']))
            # Map each option text to letter
            for L in 'ABCD':
                if norm(str(s[L])) == act:
                    act_text = norm(str(s[L]))
                    break
            for L in 'ABCD':
                if norm(str(o[L])) == obj:
                    obj_text = norm(str(o[L]))
                    break
            action_to_object.setdefault(act_text, Counter())[obj_text] += 1
    return action_to_object


def main():
    tr, te, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    
    # Initialize pipeline caches (loads dense logits, skeleton, etc.)
    P.caches(meta)
    
    # Load GPU sequence predictions
    gpu_seq = gpu_seq_preds()
    
    # Run full pipeline OOF with composite overrides
    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        print(f'Fold {fi}...', flush=True)
        ctx = P.fit_all(tr, meta, hold)
        
        # Build H1 object map from outer-train
        h1_map = h1_object_map(tr, hold)
        
        ps = make_pseudo(tr, te, meta, hold)
        pred, blk, pool, diag = P.solve(ps, ctx, 'oof')
        
        # Apply composite overrides
        for qa_id in list(pred.keys()):
            row = ps[ps.qa_id == qa_id].iloc[0] if qa_id in ps.qa_id.values else None
            if row is None:
                continue
            
            # Override 1: GPU sequence predictions
            if row['category'] == 'sequence' and (fi, qa_id) in gpu_seq:
                pred[qa_id] = gpu_seq[(fi, qa_id)]
            
            # Override 2: H1 object mapping
            if row['category'] == 'object_interaction' and row['source'] == 'HARn':
                # Find the sibling single prediction
                sib_single = ps[(ps.source == 'HARn') & 
                               (ps.category == 'single') &
                               (ps.user == row['user']) &
                               (ps.trial == row['trial'])]
                if len(sib_single) == 1:
                    sib_qa = sib_single.iloc[0].qa_id
                    sib_pred_letter = pred.get(sib_qa)
                    if sib_pred_letter and sib_pred_letter in 'ABCD':
                        import re
                        def norm(s):
                            return re.sub(r'\s+', ' ', str(s).strip().lower())
                        act_text = norm(sib_single.iloc[0][sib_pred_letter])
                        if act_text in h1_map:
                            obj_counter = h1_map[act_text]
                            best_obj = obj_counter.most_common(1)[0][0]
                            # Check if this object appears among options
                            oo = [norm(row[L]) for L in 'ABCD']
                            if best_obj in oo:
                                pred[qa_id] = 'ABCD'[oo.index(best_obj)]
        
        # Build output rows
        answer_of = dict(zip(ps.qa_id, ps.answer if 'answer' in ps.columns else [None]*len(ps)))
        cat_of = dict(zip(ps.qa_id, ps.category))
        src_of = dict(zip(ps.qa_id, ps.source))
        user_of = dict(zip(ps.qa_id, ps.user))
        
        for qa_id, p in pred.items():
            ans = str(answer_of.get(qa_id, ''))
            c = int(p == ans) if p is not None else 0
            rows.append(dict(qa_id=qa_id, fold=fi, source=src_of.get(qa_id, ''),
                            category=cat_of.get(qa_id, ''),
                            user=user_of.get(qa_id, ''),
                            pred=p, answer=ans, correct=c))
    
    d = pd.DataFrame(rows)
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       'oof_composite_v1.csv')
    d.to_csv(out, index=False)
    
    print(f'\n=== COMPOSITE OOF RESULTS ===')
    print(f'Total: {d.correct.sum()}/{len(d)} = {d.correct.mean():.4f}')
    for cat, g in d.groupby('category'):
        print(f'  {cat}: {g.correct.sum()}/{len(g)} = {g.correct.mean():.4f}')
    for fi, g in d.groupby('fold'):
        print(f'  fold {fi}: {g.correct.sum()}/{len(g)} = {g.correct.mean():.4f}')


if __name__ == '__main__':
    main()
