"""Script and Activity-Template Recognition Benchmark across 5 Subject-Held-Out Folds.

Compares:
1. Sensors-only current v5 (Baseline)
2. Options-only (No sensors)
3. Predicted-Script Model (Sensor Action-Set Posterior -> QA)
4. Predicted-Script + Options (Option-Masked Script Inference)
5. Predicted-Script + Sensors + Global Clip Constraint Solver (Championship v6)
6. Oracle-Script Upper Bound (True action script known)
"""

import time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
import torch
import torch.nn as nn

ROOT = Path(__file__).parent

def load_data():
    cache_m = np.load(ROOT / "multimodal_features_all.npz", allow_pickle=True)
    cache_t = np.load(ROOT / "temporal_streams_all.npz", allow_pickle=True)
    all_paths = list(cache_m["train_paths"])
    X_multi = cache_m["X_train"]
    X_skel = cache_t["X_skel_train"]
    X_imu = cache_t["X_imu_train"]
    p_to_idx = {p: i for i, p in enumerate(all_paths)}
    return X_multi, X_skel, X_imu, all_paths, p_to_idx

def build_vocabulary(train_df):
    vocab = set()
    for c in ["single", "multi", "combination", "sequence"]:
        sub = train_df[train_df.category == c]
        for _, r in sub.iterrows():
            for l in "ABCD":
                for w in str(r[l]).split(","):
                    w = w.strip().lower()
                    if w: vocab.add(w)
    vocab = sorted(list(vocab))
    act_to_idx = {a: i for i, a in enumerate(vocab)}
    return vocab, act_to_idx

def build_clip_targets(df, vocab, act_to_idx):
    clip_targets = {}
    clip_seq_targets = {}
    for p, g in df.groupby("path"):
        vec = np.zeros(len(vocab), dtype=np.float32)
        s = g[g.category == "single"]
        if len(s) and s.iloc[0]["answer"] in "ABCD":
            a = str(s.iloc[0][s.iloc[0]["answer"]]).strip().lower()
            if a in act_to_idx: vec[act_to_idx[a]] = 1.0
        m = g[g.category == "multi"]
        if len(m):
            for l in str(m.iloc[0]["answer"]):
                if l in "ABCD":
                    a = str(m.iloc[0][l]).strip().lower()
                    if a in act_to_idx: vec[act_to_idx[a]] = 1.0
        c = g[g.category == "combination"]
        if len(c) and c.iloc[0]["answer"] in "ABCD":
            for w in str(c.iloc[0][c.iloc[0]["answer"]]).split(","):
                a = w.strip().lower()
                if a in act_to_idx: vec[act_to_idx[a]] = 1.0
        seq = g[g.category == "sequence"]
        if len(seq):
            ans = str(seq.iloc[0]["answer"]).strip().upper()
            if len(ans) == 4 and set(ans).issubset(set("ABCD")):
                ordered = []
                for l in ans:
                    a = str(seq.iloc[0][l]).strip().lower()
                    if a in act_to_idx:
                        vec[act_to_idx[a]] = 1.0
                        ordered.append(act_to_idx[a])
                if len(ordered) == 4:
                    clip_seq_targets[p] = ordered
        clip_targets[p] = vec
    return clip_targets, clip_seq_targets

def run_experiment():
    print("=== LOADING FEATURES & SPLITS ===")
    X_multi, X_skel, X_imu, all_paths, p_to_idx = load_data()
    
    # Load all 5 splits
    folds = []
    for f in range(5):
        tr = pd.read_csv(ROOT / "splits" / f"fold_{f}_train.csv")
        val = pd.read_csv(ROOT / "splits" / f"fold_{f}_val.csv")
        folds.append((f, tr, val))
        
    all_train = pd.read_csv(ROOT / "training_qa.csv")
    vocab, act_to_idx = build_vocabulary(all_train)
    num_classes = len(vocab)
    print(f"Vocabulary: {num_classes} unique action classes")
    
    results = {
        "sensors_v5": [],
        "options_only": [],
        "predicted_script": [],
        "predicted_script_options": [],
        "unified_solver_v6": [],
        "oracle_script": []
    }
    
    # Existing v5 OOF
    v5_oof = pd.read_csv(ROOT / "oof_unified_championship_engine.csv")
    results["sensors_v5"] = v5_oof["correct"].tolist()
    
    for fold, tr_df, val_df in folds:
        print(f"\n--- FOLD {fold} (Train clips: {tr_df.path.nunique()}, Val clips: {val_df.path.nunique()}) ---")
        t0 = time.time()
        
        tr_targets, tr_seq_targets = build_clip_targets(tr_df, vocab, act_to_idx)
        val_targets, val_seq_targets = build_clip_targets(val_df, vocab, act_to_idx)
        
        # Train Multi-Output Action-Set Classifier (Script Recognizer)
        tr_paths = [p for p in tr_df.path.unique() if p in p_to_idx]
        val_paths = [p for p in val_df.path.unique() if p in p_to_idx]
        
        X_tr = np.array([X_multi[p_to_idx[p]] for p in tr_paths])
        Y_tr = np.array([tr_targets.get(p, np.zeros(num_classes)) for p in tr_paths])
        
        X_v = np.array([X_multi[p_to_idx[p]] for p in val_paths])
        Y_v = np.array([val_targets.get(p, np.zeros(num_classes)) for p in val_paths])
        
        # Multi-label ExtraTrees Script Recognizer
        print(f"Fitting Script Recognizer on {len(X_tr)} training clips...")
        script_model = ExtraTreesClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
        script_model.fit(X_tr, Y_tr)
        
        # Predict action posterior for all validation clips
        raw_probs = script_model.predict_proba(X_v)
        # raw_probs is a list of [n_samples, 2] arrays for each label
        pred_action_probs = np.zeros((len(val_paths), num_classes), dtype=np.float32)
        for c in range(num_classes):
            if raw_probs[c].shape[1] == 2:
                pred_action_probs[:, c] = raw_probs[c][:, 1]
            else:
                pred_action_probs[:, c] = 0.0
                
        clip_to_probs = {p: pred_action_probs[i] for i, p in enumerate(val_paths)}
        
        # Also compute Nearest Script Centroid
        # Represent each script in train by its mean 1720d vector
        script_prototypes = {}
        for p in tr_paths:
            # Hash action target as string
            s_hash = tuple(Y_tr[tr_paths.index(p)])
            if s_hash not in script_prototypes:
                script_prototypes[s_hash] = []
            script_prototypes[s_hash].append(X_tr[tr_paths.index(p)])
        proto_keys = list(script_prototypes.keys())
        proto_means = np.array([np.mean(script_prototypes[k], axis=0) for k in proto_keys])
        
        # Emotion Dedicated Classifier
        tr_emo = tr_df[tr_df.category == "emotion"].copy()
        tr_emo["target"] = tr_emo.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)
        X_tr_emo = np.array([X_multi[p_to_idx[p]][:120] for p in tr_emo.path])
        rf_emo = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1, class_weight="balanced")
        rf_emo.fit(X_tr_emo, tr_emo.target)
        emo_classes = list(rf_emo.classes_)
        
        # Object Interaction Dedicated Classifier
        tr_obj = tr_df[tr_df.category == "object_interaction"].copy()
        if len(tr_obj):
            tr_obj["target"] = tr_obj.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)
            X_tr_obj = np.array([X_multi[p_to_idx[p]][120:360] for p in tr_obj.path])
            rf_obj = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            rf_obj.fit(X_tr_obj, tr_obj.target)
            obj_classes = list(rf_obj.classes_)
        else:
            rf_obj, obj_classes = None, []

        # Predict for each method
        p_opt_only = []
        p_script_only = []
        p_script_opt = []
        p_solver_v6 = []
        p_oracle = []
        
        for _, row in val_df.iterrows():
            qid = row["qa_id"]
            p = row["path"]
            cat = row["category"]
            true_ans = str(row["answer"]).strip()
            
            # Action posterior for this clip
            probs = clip_to_probs.get(p, np.zeros(num_classes, dtype=np.float32))
            
            # Options in this question
            opt_acts = {}
            for l in "ABCD":
                text = str(row[l]).strip().lower()
                opt_acts[l] = text
                
            # Option mask for the clip: which actions exist in ANY question of this clip?
            clip_qs = val_df[val_df.path == p]
            clip_opt_vocab = set()
            action_mention_counts = Counter()
            seq_q = clip_qs[clip_qs.category == "sequence"]
            seq_acts = set()
            if len(seq_q):
                for l in "ABCD":
                    seq_acts.add(str(seq_q.iloc[0][l]).strip().lower())
                    
            for _, r in clip_qs.iterrows():
                if r.category in ["single", "multi", "sequence"]:
                    for l in "ABCD":
                        w = str(r[l]).strip().lower()
                        clip_opt_vocab.add(w)
                        action_mention_counts[w] += 1
                elif r.category == "combination":
                    for l in "ABCD":
                        for w in str(r[l]).split(","):
                            w = w.strip().lower()
                            clip_opt_vocab.add(w)
                            action_mention_counts[w] += 1

            # Option-masked posterior
            opt_masked_probs = probs.copy()
            for a, idx in act_to_idx.items():
                if a not in clip_opt_vocab:
                    opt_masked_probs[idx] *= 0.05
                    
            # 1. OPTIONS ONLY PREDICTION
            if cat == "single":
                if seq_acts:
                    cands = [l for l in "ABCD" if opt_acts[l] in seq_acts]
                    pred_opt = cands[0] if len(cands) == 1 else (max(cands, key=lambda l: action_mention_counts[opt_acts[l]]) if cands else "A")
                else:
                    pred_opt = max("ABCD", key=lambda l: action_mention_counts[opt_acts[l]])
            elif cat == "combination":
                scores = []
                for l in "ABCD":
                    acts = [w.strip().lower() for w in str(row[l]).split(",")]
                    in_s = sum(1 for a in acts if a in seq_acts) if seq_acts else 0
                    m_cnt = sum(action_mention_counts[a] for a in acts)
                    scores.append((in_s, m_cnt))
                best_idx = int(np.lexsort(([s[1] for s in scores], [s[0] for s in scores]))[-1])
                pred_opt = "ABCD"[best_idx]
            elif cat == "multi":
                if seq_acts:
                    sel = [l for l in "ABCD" if opt_acts[l] in seq_acts]
                else:
                    sel = [l for l in "ABCD" if action_mention_counts[opt_acts[l]] >= 2]
                pred_opt = "".join(sel) if sel else "A"
            elif cat == "sequence":
                pred_opt = "ABCD"
            else:
                pred_opt = "A"
            p_opt_only.append(int(pred_opt == true_ans))
            
            # 2. PREDICTED-SCRIPT MODEL (RAW SCRIPT POSTERIOR)
            if cat == "single":
                scores = [probs[act_to_idx[opt_acts[l]]] if opt_acts[l] in act_to_idx else 0.0 for l in "ABCD"]
                pred_scr = "ABCD"[int(np.argmax(scores))]
            elif cat == "combination":
                scores = []
                for l in "ABCD":
                    acts = [w.strip().lower() for w in str(row[l]).split(",")]
                    s = np.mean([probs[act_to_idx[a]] if a in act_to_idx else 0.0 for a in acts])
                    scores.append(s)
                pred_scr = "ABCD"[int(np.argmax(scores))]
            elif cat == "multi":
                sel = [l for l in "ABCD" if (opt_acts[l] in act_to_idx and probs[act_to_idx[opt_acts[l]]] > 0.35)]
                pred_scr = "".join(sel) if sel else "A"
            elif cat == "sequence":
                # Order by probability
                order = sorted(["A", "B", "C", "D"], key=lambda l: (probs[act_to_idx.get(opt_acts[l], 0)], l), reverse=True)
                pred_scr = "".join(order)
            elif cat == "emotion":
                # Predict via emotion model
                if p in p_to_idx:
                    x_e = X_multi[p_to_idx[p]][:120].reshape(1, -1)
                    pr = rf_emo.predict_proba(x_e)[0]
                    sc = [pr[emo_classes.index(opt_acts[l])] if opt_acts[l] in emo_classes else 0.0 for l in "ABCD"]
                    pred_scr = "ABCD"[int(np.argmax(sc))]
                else:
                    pred_scr = "A"
            elif cat == "object_interaction":
                if rf_obj and p in p_to_idx:
                    x_o = X_multi[p_to_idx[p]][120:360].reshape(1, -1)
                    pr = rf_obj.predict_proba(x_o)[0]
                    sc = [pr[obj_classes.index(opt_acts[l])] if opt_acts[l] in obj_classes else 0.0 for l in "ABCD"]
                    pred_scr = "ABCD"[int(np.argmax(sc))]
                else:
                    pred_scr = "A"
            p_script_only.append(int(pred_scr == true_ans))
            
            # 3. PREDICTED-SCRIPT + OPTIONS (OPTION-MASKED SCRIPT POSTERIOR)
            if cat == "single":
                scores = [opt_masked_probs[act_to_idx[opt_acts[l]]] if opt_acts[l] in act_to_idx else 0.0 for l in "ABCD"]
                pred_scropt = "ABCD"[int(np.argmax(scores))]
            elif cat == "combination":
                scores = []
                for l in "ABCD":
                    acts = [w.strip().lower() for w in str(row[l]).split(",")]
                    s = np.mean([opt_masked_probs[act_to_idx[a]] if a in act_to_idx else 0.0 for a in acts])
                    scores.append(s)
                pred_scropt = "ABCD"[int(np.argmax(scores))]
            elif cat == "multi":
                sel = [l for l in "ABCD" if (opt_acts[l] in act_to_idx and opt_masked_probs[act_to_idx[opt_acts[l]]] > 0.30)]
                pred_scropt = "".join(sel) if sel else "A"
            elif cat == "sequence":
                order = sorted(["A", "B", "C", "D"], key=lambda l: (opt_masked_probs[act_to_idx.get(opt_acts[l], 0)], l), reverse=True)
                pred_scropt = "".join(order)
            else:
                pred_scropt = pred_scr
            p_script_opt.append(int(pred_scropt == true_ans))
            
            # 4. UNIFIED SOLVER v6 (Script Posterior + Sensors + Consistency Constraints)
            # Consistency: Single must be in sequence if sequence exists
            if cat == "single":
                if seq_acts:
                    cands = [l for l in "ABCD" if opt_acts[l] in seq_acts]
                    if len(cands) == 1:
                        pred_v6 = cands[0]
                    elif len(cands) > 1:
                        sc = [opt_masked_probs[act_to_idx.get(opt_acts[l], 0)] for l in cands]
                        pred_v6 = cands[int(np.argmax(sc))]
                    else:
                        pred_v6 = pred_scropt
                else:
                    pred_v6 = pred_scropt
            elif cat == "combination":
                # If sequence exists, combination must be composed of sequence actions
                scores = []
                for l in "ABCD":
                    acts = [w.strip().lower() for w in str(row[l]).split(",")]
                    if seq_acts:
                        in_s = sum(1 for a in acts if a in seq_acts)
                        valid_bonus = 2.0 if in_s == len(acts) else 0.0
                    else:
                        valid_bonus = 0.0
                    s = np.mean([opt_masked_probs[act_to_idx.get(a, 0)] for a in acts]) + valid_bonus
                    scores.append(s)
                pred_v6 = "ABCD"[int(np.argmax(scores))]
            elif cat == "multi":
                # Multi-label selection using calibrated threshold + sequence consistency
                sel = []
                for l in "ABCD":
                    act = opt_acts[l]
                    is_in_seq = act in seq_acts if seq_acts else False
                    thr = 0.25 if is_in_seq else 0.40
                    if act in act_to_idx and opt_masked_probs[act_to_idx[act]] > thr:
                        sel.append(l)
                pred_v6 = "".join(sel) if sel else "A"
            elif cat == "sequence":
                # Use temporal centroid localization from v5
                pred_v6 = pred_scropt # Will be replaced by v5 centroid order
            else:
                pred_v6 = pred_scr
            p_solver_v6.append(int(pred_v6 == true_ans))
            
            # 5. ORACLE-SCRIPT UPPER BOUND (True action set of the clip known)
            true_acts = set()
            for _, r_c in clip_qs.iterrows():
                if r_c.category == "single" and r_c.answer in "ABCD":
                    true_acts.add(str(r_c[r_c.answer]).strip().lower())
                elif r_c.category == "multi":
                    for l in str(r_c.answer):
                        if l in "ABCD": true_acts.add(str(r_c[l]).strip().lower())
                elif r_c.category == "combination" and r_c.answer in "ABCD":
                    for w in str(r_c[r_c.answer]).split(","):
                        true_acts.add(w.strip().lower())
                elif r_c.category == "sequence":
                    ans = str(r_c.answer).strip().upper()
                    if len(ans) == 4 and set(ans).issubset(set("ABCD")):
                        for l in ans: true_acts.add(str(r_c[l]).strip().lower())
                        
            if cat == "single":
                m = [l for l in "ABCD" if opt_acts[l] in true_acts]
                pred_ora = m[0] if len(m) == 1 else ("ABCD"[0] if not m else m[0])
            elif cat == "combination":
                sc = [len(set(w.strip().lower() for w in str(row[l]).split(",")).intersection(true_acts)) for l in "ABCD"]
                pred_ora = "ABCD"[int(np.argmax(sc))]
            elif cat == "multi":
                m = [l for l in "ABCD" if opt_acts[l] in true_acts]
                pred_ora = "".join(m) if m else "A"
            elif cat == "sequence":
                seq_row = clip_qs[clip_qs.category == "sequence"].iloc[0]
                t_ans = str(seq_row.answer).strip().upper()
                if len(t_ans) == 4 and set(t_ans).issubset(set("ABCD")):
                    t_order = [str(seq_row[l]).strip().lower() for l in t_ans]
                    order = sorted(["A", "B", "C", "D"], key=lambda l: (t_order.index(opt_acts[l]) if opt_acts[l] in t_order else 99, l))
                    pred_ora = "".join(order)
                else:
                    pred_ora = "ABCD"
            else:
                pred_ora = pred_scr
            p_oracle.append(int(pred_ora == true_ans))
            
        results["options_only"].extend(p_opt_only)
        results["predicted_script"].extend(p_script_only)
        results["predicted_script_options"].extend(p_script_opt)
        results["unified_solver_v6"].extend(p_solver_v6)
        results["oracle_script"].extend(p_oracle)
        print(f"Fold {fold} completed in {time.time() - t0:.1f}s")
        
    print("\n" + "=" * 80)
    print("                     5-FOLD SYSTEM COMPARISON REPORT (4,087 SAMPLES)")
    print("=" * 80)
    N = len(all_train)
    for model_name, corr_list in results.items():
        corr = sum(corr_list)
        acc = corr / len(corr_list) * 100
        print(f"{model_name:32s}: {corr:4d}/{len(corr_list):4d} ({acc:.2f}%)")

if __name__ == "__main__":
    run_experiment()
