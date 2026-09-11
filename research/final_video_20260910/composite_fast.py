"""Fast composite OOF: post-hoc override of pipeline baseline.

Takes the already-computed oof_pipeline_base.csv and applies two validated
overrides without re-running the full pipeline:
1. GPU dense TCN sequence predictions (replaces pipeline seq predictions)
2. H1 object mapping hybrid (overrides some object_interaction predictions)

This is faster and exactly equivalent to running the full composite pipeline
since both overrides are downstream of the pipeline's action/pool predictions.
"""
import os
import csv
import re
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def norm(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def gpu_seq_preds():
    """Load GPU OOF sequence predictions from the per-fold .npy files."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'training_qa.csv'),
                                    encoding='utf-8-sig')))
    seq = {r['path']: r for r in rows
           if r['path'].startswith('HAU/') and r['category'] == 'sequence'}
    preds = {}
    for fi in range(5):
        fp = os.path.join(ROOT, 'research', 'final_video_20260910',
                          f'dense_oof_f{fi}.npy')
        if not os.path.exists(fp):
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
    return preds


def h1_overrides():
    """Load pre-computed H1 hybrid overrides."""
    h1 = pd.read_csv(os.path.join(ROOT, 'research', 'final_video_20260910',
                                  'oof_hybrid_H1.csv'))
    return h1


def main():
    base = pd.read_csv(os.path.join(ROOT, 'research', 'final_video_20260910',
                                    'oof_pipeline_base.csv'))
    base['correct'] = base['correct'].astype(int)
    
    gpu_seq = gpu_seq_preds()
    h1 = h1_overrides()
    h1_preds = dict(zip(zip(h1.qa_id, h1.fold.astype(int)),
                        zip(h1.pred, h1.correct.astype(int))))
    
    # Apply overrides
    n_seq_changed = 0
    n_h1_changed = 0
    composite = base.copy()
    
    for idx, row in composite.iterrows():
        qa_id = row['qa_id']
        fi = int(row['fold'])
        
        # Override 1: GPU sequence
        if row['category'] == 'sequence' and (fi, qa_id) in gpu_seq:
            new_pred = gpu_seq[(fi, qa_id)]
            if new_pred != row['pred']:
                n_seq_changed += 1
            composite.at[idx, 'pred'] = new_pred
            composite.at[idx, 'correct'] = int(new_pred == row['answer'])
        
        # Override 2: H1 object mapping
        if row['category'] == 'object_interaction' and (qa_id, fi) in h1_preds:
            h1_pred, h1_correct = h1_preds[(qa_id, fi)]
            if str(h1_pred) != str(row['pred']) and str(h1_pred) in 'ABCD':
                n_h1_changed += 1
                composite.at[idx, 'pred'] = h1_pred
                composite.at[idx, 'correct'] = int(h1_pred == row['answer'])
    
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       'oof_composite_v1.csv')
    composite.to_csv(out, index=False)
    
    print(f'=== COMPOSITE OOF (post-hoc overrides) ===')
    print(f'Seq predictions changed: {n_seq_changed}')
    print(f'H1 obj predictions changed: {n_h1_changed}')
    print()
    print(f'Pipeline base: {base.correct.sum()}/{len(base)} = {base.correct.mean():.4f}')
    print(f'Composite:     {composite.correct.sum()}/{len(composite)} = {composite.correct.mean():.4f}')
    print(f'Delta: +{composite.correct.sum() - base.correct.sum()}')
    print()
    
    for cat in sorted(composite.category.unique()):
        b = base[base.category == cat]
        c = composite[composite.category == cat]
        delta = c.correct.sum() - b.correct.sum()
        sym = '+' if delta >= 0 else ''
        print(f'  {cat:25s}: {b.correct.sum()}/{len(b)} -> {c.correct.sum()}/{len(c)} ({sym}{delta})')
    
    print()
    for fi in sorted(composite.fold.unique()):
        b = base[base.fold == fi]
        c = composite[composite.fold == fi]
        delta = c.correct.sum() - b.correct.sum()
        print(f'  fold {fi}: {b.correct.sum()}/{len(b)} -> {c.correct.sum()}/{len(c)} (+{delta})')
    
    # Bootstrap CI for net gain
    rng = np.random.default_rng(42)
    gains = (composite.correct.values - base.correct.values).astype(float)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(gains), len(gains))
        boots.append(gains[idx].sum())
    boots = np.array(boots)
    print(f'\n  Bootstrap 95% CI for net gain: [{np.percentile(boots, 2.5):.0f}, {np.percentile(boots, 97.5):.0f}]')
    print(f'  P(net >= 0): {(boots >= 0).mean():.4f}')
    print(f'  Expected net gain: {np.mean(boots):.1f}')


if __name__ == '__main__':
    main()
