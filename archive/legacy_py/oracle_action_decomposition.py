"""Track A: Action Oracle Decomposition Benchmark.

Measures downstream QA accuracy for action categories:
- Single (1,238 samples)
- Combination (790 samples)
- Multi (809 samples)
- Sequence (308 samples)
- Total Action QA (3,145 samples)

Across 8 distinct levels of latent oracle information:
1. Oracle Action Presence / Set only
2. Oracle Action Count only
3. Oracle Action Set + Count
4. Oracle Action Set + Pairwise Ordering
5. Oracle Full Action Ordering
6. Oracle Concurrent-Action Relationships
7. Oracle Temporal Boundaries
8. Oracle Full Trajectory
"""

import pandas as pd
import numpy as np
from itertools import permutations

train = pd.read_csv('training_qa.csv')

# Extract clip ground-truth
clips = {}
for p, g in train.groupby('path'):
    info = {
        'path': p,
        'source': g.iloc[0]['source'],
        'single_act': None,
        'multi_acts': set(),
        'comb_acts': set(),
        'seq_order': [],
        'all_acts': set()
    }
    s = g[g.category == 'single']
    if len(s) and s.iloc[0]['answer'] in 'ABCD':
        act = str(s.iloc[0][s.iloc[0]['answer']]).strip().lower()
        info['single_act'] = act
        info['all_acts'].add(act)
    m = g[g.category == 'multi']
    if len(m):
        for l in str(m.iloc[0]['answer']):
            if l in 'ABCD':
                act = str(m.iloc[0][l]).strip().lower()
                info['multi_acts'].add(act)
                info['all_acts'].add(act)
    c = g[g.category == 'combination']
    if len(c) and c.iloc[0]['answer'] in 'ABCD':
        for w in str(c.iloc[0][c.iloc[0]['answer']]).split(','):
            act = w.strip().lower()
            info['comb_acts'].add(act)
            info['all_acts'].add(act)
    seq = g[g.category == 'sequence']
    if len(seq):
        ans = str(seq.iloc[0]['answer']).strip().upper()
        if len(ans) == 4 and set(ans).issubset(set('ABCD')):
            ordered = [str(seq.iloc[0][l]).strip().lower() for l in ans]
            info['seq_order'] = ordered
            for a in ordered:
                info['all_acts'].add(a)
    clips[p] = info

def evaluate_oracle_level(level_name):
    # Evaluates Single, Combination, Multi, Sequence
    action_cats = ['single', 'combination', 'multi', 'sequence']
    cat_correct = {c: 0 for c in action_cats}
    cat_total = {c: 0 for c in action_cats}
    
    for _, row in train[train.category.isin(action_cats)].iterrows():
        p = row['path']
        cat = row['category']
        ans = str(row['answer']).strip()
        info = clips[p]
        all_acts = info['all_acts']
        seq_order = info['seq_order']
        n_acts = len(all_acts)
        
        # Options map
        opts = {l: str(row[l]).strip().lower() for l in 'ABCD'}
        
        pred = 'A'
        
        # Level 1: Action Presence / Set Only
        if level_name == '1. Action Presence / Set Only':
            if cat == 'single':
                # Which option is in all_acts?
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = m[0] if len(m) == 1 else (m[0] if m else 'A')
            elif cat == 'combination':
                # Which combo option has highest intersection with all_acts?
                scores = []
                for l in 'ABCD':
                    acts = [w.strip().lower() for w in opts[l].split(',')]
                    inter = sum(1 for a in acts if a in all_acts)
                    scores.append(inter / len(acts))
                pred = 'ABCD'[int(np.argmax(scores))]
            elif cat == 'multi':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = ''.join(m) if m else 'A'
            elif cat == 'sequence':
                # Without ordering, best is random or neutral ABCD
                pred = 'ABCD'
                
        # Level 2: Action Count Only
        elif level_name == '2. Action Count Only':
            if cat == 'single':
                pred = 'A'
            elif cat == 'combination':
                # Pick combo option whose number of items is closest to count
                scores = []
                for l in 'ABCD':
                    acts = [w.strip().lower() for w in opts[l].split(',')]
                    scores.append(-abs(len(acts) - n_acts))
                pred = 'ABCD'[int(np.argmax(scores))]
            elif cat == 'multi':
                # Multi answer length closest to count
                # Guess first n_acts letters
                k = min(4, max(1, n_acts))
                pred = 'ABCD'[:k]
            elif cat == 'sequence':
                pred = 'ABCD'
                
        # Level 3: Action Set + Count
        elif level_name == '3. Action Set + Count':
            if cat == 'single':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = m[0] if len(m) == 1 else (m[0] if m else 'A')
            elif cat == 'combination':
                scores = []
                for l in 'ABCD':
                    acts = [w.strip().lower() for w in opts[l].split(',')]
                    inter = sum(1 for a in acts if a in all_acts)
                    frac = inter / len(acts)
                    sz_diff = -abs(len(acts) - n_acts)
                    scores.append((frac, sz_diff))
                best_idx = int(np.lexsort(([s[1] for s in scores], [s[0] for s in scores]))[-1])
                pred = 'ABCD'[best_idx]
            elif cat == 'multi':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = ''.join(m) if m else 'A'
            elif cat == 'sequence':
                pred = 'ABCD'
                
        # Level 4: Action Set + Pairwise Ordering
        elif level_name == '4. Action Set + Pairwise Ordering':
            if cat == 'single':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = m[0] if len(m) == 1 else (m[0] if m else 'A')
            elif cat == 'combination':
                scores = []
                for l in 'ABCD':
                    acts = [w.strip().lower() for w in opts[l].split(',')]
                    inter = sum(1 for a in acts if a in all_acts)
                    scores.append(inter / len(acts))
                pred = 'ABCD'[int(np.argmax(scores))]
            elif cat == 'multi':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = ''.join(m) if m else 'A'
            elif cat == 'sequence':
                if seq_order:
                    # Sort options by pairwise ordering in seq_order
                    ranks = []
                    for l in 'ABCD':
                        act = opts[l]
                        idx = seq_order.index(act) if act in seq_order else 99
                        ranks.append((idx, l))
                    ranks.sort()
                    pred = ''.join([x[1] for x in ranks])
                else:
                    pred = 'ABCD'
                    
        # Level 5: Full Action Ordering
        elif level_name == '5. Full Action Ordering':
            if cat == 'single':
                # If sequence order known, single is one of them
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = m[0] if len(m) == 1 else (m[0] if m else 'A')
            elif cat == 'combination':
                scores = []
                for l in 'ABCD':
                    acts = [w.strip().lower() for w in opts[l].split(',')]
                    inter = sum(1 for a in acts if a in all_acts)
                    scores.append(inter / len(acts))
                pred = 'ABCD'[int(np.argmax(scores))]
            elif cat == 'multi':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = ''.join(m) if m else 'A'
            elif cat == 'sequence':
                if seq_order:
                    ranks = sorted(['A', 'B', 'C', 'D'], key=lambda l: seq_order.index(opts[l]) if opts[l] in seq_order else 99)
                    pred = ''.join(ranks)
                else:
                    pred = 'ABCD'
                    
        # Level 6: Concurrent-Action Relationships
        elif level_name == '6. Concurrent-Action Relationships':
            # Adds exact multi-action concurrent subset knowledge for combination
            if cat == 'single':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = m[0] if len(m) == 1 else (m[0] if m else 'A')
            elif cat == 'combination':
                # Exact match with ground truth combination actions!
                c_acts = info['comb_acts']
                scores = []
                for l in 'ABCD':
                    acts = set(w.strip().lower() for w in opts[l].split(','))
                    scores.append(len(acts.intersection(c_acts)) / max(1, len(acts.union(c_acts))))
                pred = 'ABCD'[int(np.argmax(scores))]
            elif cat == 'multi':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = ''.join(m) if m else 'A'
            elif cat == 'sequence':
                if seq_order:
                    ranks = sorted(['A', 'B', 'C', 'D'], key=lambda l: seq_order.index(opts[l]) if opts[l] in seq_order else 99)
                    pred = ''.join(ranks)
                else:
                    pred = 'ABCD'
                    
        # Level 7: Temporal Boundaries
        elif level_name == '7. Temporal Boundaries':
            # Temporal boundaries uniquely identify action transitions and isolate single focal action
            if cat == 'single':
                # Single focal action
                if info['single_act']:
                    m = [l for l in 'ABCD' if opts[l] == info['single_act']]
                    pred = m[0] if m else 'A'
                else:
                    m = [l for l in 'ABCD' if opts[l] in all_acts]
                    pred = m[0] if m else 'A'
            elif cat == 'combination':
                c_acts = info['comb_acts']
                scores = []
                for l in 'ABCD':
                    acts = set(w.strip().lower() for w in opts[l].split(','))
                    scores.append(len(acts.intersection(c_acts)) / max(1, len(acts.union(c_acts))))
                pred = 'ABCD'[int(np.argmax(scores))]
            elif cat == 'multi':
                m = [l for l in 'ABCD' if opts[l] in all_acts]
                pred = ''.join(m) if m else 'A'
            elif cat == 'sequence':
                if seq_order:
                    ranks = sorted(['A', 'B', 'C', 'D'], key=lambda l: seq_order.index(opts[l]) if opts[l] in seq_order else 99)
                    pred = ''.join(ranks)
                else:
                    pred = 'ABCD'
                    
        # Level 8: Full Trajectory
        elif level_name == '8. Full Trajectory':
            if cat == 'single':
                if info['single_act']:
                    m = [l for l in 'ABCD' if opts[l] == info['single_act']]
                    pred = m[0] if m else 'A'
                else:
                    m = [l for l in 'ABCD' if opts[l] in all_acts]
                    pred = m[0] if m else 'A'
            elif cat == 'combination':
                c_acts = info['comb_acts']
                scores = []
                for l in 'ABCD':
                    acts = set(w.strip().lower() for w in opts[l].split(','))
                    scores.append(len(acts.intersection(c_acts)) / max(1, len(acts.union(c_acts))))
                pred = 'ABCD'[int(np.argmax(scores))]
            elif cat == 'multi':
                # Exact multi acts
                m_acts = info['multi_acts']
                m = [l for l in 'ABCD' if opts[l] in m_acts]
                pred = ''.join(m) if m else 'A'
            elif cat == 'sequence':
                if seq_order:
                    ranks = sorted(['A', 'B', 'C', 'D'], key=lambda l: seq_order.index(opts[l]) if opts[l] in seq_order else 99)
                    pred = ''.join(ranks)
                else:
                    pred = 'ABCD'
                    
        if pred == ans:
            cat_correct[cat] += 1
        cat_total[cat] += 1
        
    tot_c = sum(cat_correct.values())
    tot_n = sum(cat_total.values())
    return {
        'level': level_name,
        'overall': tot_c / tot_n * 100,
        'single': cat_correct['single'] / cat_total['single'] * 100,
        'combination': cat_correct['combination'] / cat_total['combination'] * 100,
        'multi': cat_correct['multi'] / cat_total['multi'] * 100,
        'sequence': cat_correct['sequence'] / cat_total['sequence'] * 100
    }

levels = [
    '1. Action Presence / Set Only',
    '2. Action Count Only',
    '3. Action Set + Count',
    '4. Action Set + Pairwise Ordering',
    '5. Full Action Ordering',
    '6. Concurrent-Action Relationships',
    '7. Temporal Boundaries',
    '8. Full Trajectory'
]

results = []
for lvl in levels:
    res = evaluate_oracle_level(lvl)
    results.append(res)

df_res = pd.DataFrame(results)
print('================================================================================')
print('                 TRACK A: ACTION ORACLE DECOMPOSITION TABLE')
print('================================================================================')
print(f"{'Oracle Information Level':35s} | {'Overall':8s} | {'Single':7s} | {'Combin.':8s} | {'Multi':7s} | {'Sequence':8s}")
print('-' * 88)
for _, r in df_res.iterrows():
    print(f"{r['level']:35s} | {r['overall']:6.2f}% | {r['single']:5.2f}% | {r['combination']:6.2f}% | {r['multi']:5.2f}% | {r['sequence']:6.2f}%")
