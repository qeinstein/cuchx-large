"""
adversarial_audit_330_champion_20260907.py
Adversarial test-wide audit across all 682 test rows of the 330 champion.
Identifies:
  1. Internal contradictions across sibling questions (Single vs Multi vs Sequence vs Comb)
  2. Sequence pairwise total order violations
  3. Emotion triad / manner speed order inversions
  4. Cross-modal sensor vs visual vs dense logit disagreements
  5. Top 50 most suspicious / fragile champion predictions
"""

import sys, os, math, itertools
import numpy as np, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, "champ")
from core import load_all, real_test_view, fit_group_model, infer_blocks, mgroup, PHYS

print("Loading test data, metadata, and features...")
tr, te, meta = load_all()
sub = pd.read_csv("submissions/submission_096491_330of342_CHAMPION.csv").set_index("qa_id")
vis = real_test_view(te)
meta_idx = meta.set_index("qa_path")
feats = pd.read_csv("champ/feats.csv").set_index("unit_dir")

# 1. Map clips to their questions
clip_to_rows = defaultdict(list)
for _, r in te.iterrows():
    p = r.path
    # extract clip identifier e.g. LM_test_0001
    clip_id = None
    for part in p.split("/"):
        if part.startswith("LM_test_"):
            clip_id = part
            break
    if clip_id is None:
        clip_id = p
    clip_to_rows[clip_id].append(r)

print(f"Total test questions: {len(te)}, Total unique clips: {len(clip_to_rows)}")

# 2. Check Single vs Multi consistency on the same clip
print("\n=== 1. AUDITING SINGLE vs MULTI SUBSET CONSISTENCY ===")
single_multi_conflicts = []
for clip_id, rows in clip_to_rows.items():
    s_rows = [r for r in rows if r.category == 'single']
    m_rows = [r for r in rows if r.category == 'multi']
    if s_rows and m_rows:
        for sr in s_rows:
            s_pred = sub.loc[sr.qa_id, 'prediction']
            s_action = str(sr.get(s_pred)).strip()
            for mr in m_rows:
                m_pred = sub.loc[mr.qa_id, 'prediction']
                # m_pred can be multiple letters e.g. 'CD'
                m_actions = [str(mr.get(L)).strip() for L in m_pred if pd.notna(mr.get(L))]
                # If s_action is present as an option in mr, is it included in m_actions?
                mr_opts = {L: str(mr.get(L)).strip() for L in "ABCD" if pd.notna(mr.get(L))}
                if s_action in mr_opts.values():
                    # Option exists in multi!
                    s_letter_in_mr = [L for L, name in mr_opts.items() if name == s_action][0]
                    if s_letter_in_mr not in m_pred:
                        single_multi_conflicts.append({
                            'clip': clip_id, 'single_qid': sr.qa_id, 'single_ans': f"{s_pred}:{s_action}",
                            'multi_qid': mr.qa_id, 'multi_ans': f"{m_pred}:{[mr_opts[L] for L in m_pred]}",
                            'missing_in_multi': f"{s_letter_in_mr}:{s_action}"
                        })

print(f"Total Single vs Multi conflicts: {len(single_multi_conflicts)}")
for c in single_multi_conflicts:
    print(f"  Clip {c['clip']}: Single {c['single_qid']}={c['single_ans']} but Multi {c['multi_qid']}={c['multi_ans']} is MISSING {c['missing_in_multi']}")

# 3. Check Sequence Question Internal Consistency
print("\n=== 2. AUDITING SEQUENCE TOTAL-ORDER CONSISTENCY ===")
seq_rows = te[te.category == 'sequence'].copy()
print(f"Total sequence questions: {len(seq_rows)}")

# Group sequence questions by session/block if possible
# In test set, sequence questions are in blocks. Let's find pairs of sequence questions on the same session
seq_conflicts = []
for i in range(len(seq_rows)):
    for j in range(i + 1, len(seq_rows)):
        r1, r2 = seq_rows.iloc[i], seq_rows.iloc[j]
        # Check if they share the same video or same unit_dir
        # Extract clip IDs
        c1 = [p for p in r1.path.split('/') if p.startswith('LM_test_')][0]
        c2 = [p for p in r2.path.split('/') if p.startswith('LM_test_')][0]
        p1 = sub.loc[r1.qa_id, 'prediction']
        p2 = sub.loc[r2.qa_id, 'prediction']
        opts1 = {L: str(r1.get(L)).strip() for L in "ABCD" if pd.notna(r1.get(L))}
        opts2 = {L: str(r2.get(L)).strip() for L in "ABCD" if pd.notna(r2.get(L))}
        
        # Check overlap of actions
        inter = set(opts1.values()) & set(opts2.values())
        if len(inter) >= 2:
            # Check if relative order of overlapping actions agrees
            order1 = [opts1[L] for L in p1 if opts1[L] in inter]
            order2 = [opts2[L] for L in p2 if opts2[L] in inter]
            if order1 != order2:
                seq_conflicts.append({
                    'q1': r1.qa_id, 'clip1': c1, 'pred1': p1, 'order1': order1,
                    'q2': r2.qa_id, 'clip2': c2, 'pred2': p2, 'order2': order2,
                    'actions': inter
                })

print(f"Total sequence pairwise order conflicts: {len(seq_conflicts)}")
for sc in seq_conflicts:
    print(f"  Conflict between {sc['q1']} ({sc['clip1']}) and {sc['q2']} ({sc['clip2']}):")
    print(f"     Order in {sc['q1']}: {' < '.join(sc['order1'])}")
    print(f"     Order in {sc['q2']}: {' < '.join(sc['order2'])}")

# 4. Check 3-Clip Emotion Blocks Speed Inversions
print("\n=== 3. AUDITING 3-CLIP EMOTION BLOCKS FOR MANNER SPEED INVERSIONS ===")
blocks = infer_blocks(vis, fit_group_model(tr))

# Training speed table
h = tr[(tr.source == "HAU") & (tr.category == "emotion")].copy()
h["manner"] = h.apply(lambda r: str(r.get(r.answer)).strip(), axis=1)
manner_speeds = {}
for m, g in h.groupby("manner"):
    spds = [feats.loc[meta_idx.loc[p, "unit_dir"], "sk_v_mean"] for p in g.path
            if p in meta_idx.index and pd.notna(meta_idx.loc[p, "unit_dir"]) and meta_idx.loc[p, "unit_dir"] in feats.index
            and np.isfinite(feats.loc[meta_idx.loc[p, "unit_dir"], "sk_v_mean"])]
    if spds:
        manner_speeds[m] = np.median(spds)

e_te = te[te.category == "emotion"].copy()
te_by_clip = {}
for _, r in e_te.iterrows():
    cid = [p for p in r.path.split('/') if p.startswith('LM_test_')][0]
    te_by_clip[cid] = r

speed_inversions = []
for bi, b in enumerate(blocks):
    if len(b) == 3:
        clips = [vis[vis.idx == idx].iloc[0].true_path for idx in b]
        rows = [te_by_clip.get(c) for c in clips if c in te_by_clip]
        if len(rows) == 3:
            preds = [sub.loc[r.qa_id, 'prediction'] for r in rows]
            manners = [str(r.get(p)).strip() for r, p in zip(rows, preds)]
            
            # Get physical speeds
            clip_spds = []
            for c in clips:
                udir = meta_idx.loc[c, 'unit_dir'] if c in meta_idx.index else None
                spd = float(feats.loc[udir, 'sk_v_mean']) if (pd.notna(udir) and udir in feats.index) else np.nan
                clip_spds.append(spd)
                
            # Expected speed of predicted manners
            exp_spds = [manner_speeds.get(m, np.nan) for m in manners]
            
            # Check if clip speeds and expected speeds correlate
            if all(np.isfinite(clip_spds)) and all(np.isfinite(exp_spds)):
                # Pairwise speed comparisons
                for i0, i1 in [(0, 1), (1, 2), (0, 2)]:
                    diff_phys = clip_spds[i1] - clip_spds[i0]
                    diff_manner = exp_spds[i1] - exp_spds[i0]
                    if abs(diff_phys) > 0.005 and abs(diff_manner) > 0.005:
                        if np.sign(diff_phys) != np.sign(diff_manner):
                            speed_inversions.append({
                                'block': bi, 'clips': (clips[i0], clips[i1]),
                                'qids': (rows[i0].qa_id, rows[i1].qa_id),
                                'manners': (manners[i0], manners[i1]),
                                'phys_spds': (clip_spds[i0], clip_spds[i1]),
                                'manner_spds': (exp_spds[i0], exp_spds[i1]),
                                'severity': abs(diff_phys)
                            })

print(f"Total speed inversions in 3-clip emotion blocks: {len(speed_inversions)}")
df_inv = pd.DataFrame(speed_inversions)
if len(df_inv) > 0:
    df_inv = df_inv.sort_values('severity', ascending=False)
    print(df_inv.head(10).to_string(index=False))

# 5. HARn Single Low-Margin and Sensor-Class Disagreements
print("\n=== 4. AUDITING ALL HARN SINGLE QUESTIONS ===")
harn_single = te[(te.source == 'HARn') & (te.category == 'single')].copy()
print(f"Total HARn single questions: {len(harn_single)}")

harn_audit = []
for _, r in harn_single.iterrows():
    qid = r.qa_id
    cid = [p for p in r.path.split('/') if p.startswith('LM_test_')][0]
    p_champ = sub.loc[qid, 'prediction']
    ans_text = str(r.get(p_champ)).strip()
    
    # Get sensor features
    udir = meta_idx.loc[cid, 'unit_dir'] if cid in meta_idx.index else None
    f = feats.loc[udir] if (pd.notna(udir) and udir in feats.index) else {}
    
    imu_std = float(f.get('imu_acc_std', np.nan))
    imu_mean = float(f.get('imu_acc_mean', np.nan))
    sk_spd = float(f.get('sk_v_mean', np.nan))
    sk_cad = float(f.get('sk_cad_hz', np.nan))
    sk_hip = float(f.get('sk_hip_path', np.nan))
    
    opts_dict = {L: str(r.get(L)).strip() for L in "ABCD" if pd.notna(r.get(L))}
    
    harn_audit.append({
        'qid': qid, 'clip': cid, 'champ': f"{p_champ}:{ans_text}",
        'opts': opts_dict,
        'imu_std': round(imu_std, 4), 'sk_spd': round(sk_spd, 4),
        'sk_cad': round(sk_cad, 2), 'sk_hip': round(sk_hip, 4)
    })

df_harn = pd.DataFrame(harn_audit)
df_harn.to_csv("scratch/harn_single_audit.csv", index=False)
print(f"Saved HARn single audit ({len(df_harn)} rows) to scratch/harn_single_audit.csv")

# Look for physical contradictions in HARn single
# e.g., high motion (imu_std > 0.3) labeled as stationary actions (peeling fruit, reading documents, turning pages, drinking water)
stationary_actions = {'peeling fruit', 'reading documents', 'turning pages', 'drinking water', 'eating food', 'listen to music', 'check the time'}
high_motion_actions = {'jog in place', 'do squats', 'walk', 'sweep the floor', 'mop the floor', 'jump'}

print("\n--- Physical Contradiction Probes in HARn Single ---")
for _, r in df_harn.iterrows():
    ans_lower = r['champ'].split(':')[-1].lower()
    imu = r['imu_std']
    # Case A: Stationary answer but high IMU
    for act in stationary_actions:
        if act in ans_lower and imu > 0.25:
            print(f"SUSPICIOUS (High IMU for stationary): {r['qid']} ({r['clip']}) ans={r['champ']} with imu={imu}! Opts={r['opts']}")
    # Case B: High motion answer but near-zero IMU
    for act in high_motion_actions:
        if act in ans_lower and imu < 0.05 and np.isfinite(imu):
            print(f"SUSPICIOUS (Low IMU for active): {r['qid']} ({r['clip']}) ans={r['champ']} with imu={imu}! Opts={r['opts']}")
