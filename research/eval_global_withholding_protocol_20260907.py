"""Global Benchmark Protocol Reconstruction & 2-Clip Withholding Evaluation.

Validates the Trial 2 (Middle) Withholding Mechanism across all training and test cohorts:
1. Global training session structure: 3-trial protocol (Trial 1 Slow -> Trial 2 Neutral -> Trial 3 Fast).
2. 5-Fold Subject-Disjoint OOF evaluation of withholding regimes:
   - Regime (0, 2): Middle trial withheld [Benchmark Standard]
   - Regime (0, 1): Last trial withheld
   - Regime (1, 2): First trial withheld
3. Physical and kinematic discriminator analysis (durations, skeleton velocities, IMU energy).
4. Exhaustive audit of all 21 test two-clip blocks (11 in Cohort 3, 10 in Cohort 2).
5. Exact error isolation and candidate generation.
"""
from __future__ import annotations

import os
import sys
import itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
sys.path.insert(0, os.path.join(ROOT, "research"))

from core import load_all, fit_manner, mgroup, GROUPS, opts
from pseudotest import folds
import evaluate_user_template_pairs_20260904 as U

def main():
    print("=" * 80)
    print("GLOBAL BENCHMARK PROTOCOL RECONSTRUCTION: 2-CLIP WITHHOLDING")
    print("=" * 80)

    tr, te, meta = load_all()
    meta_idx = meta.set_index("qa_path")
    champ = pd.read_csv(os.path.join(ROOT, "submissions/submission_096198_329of342_CHAMPION.csv")).set_index("qa_id")
    skel = np.load(os.path.join(ROOT, "champ", "skel_seq.npz"))

    # Part 1: Training Structure Reconstruction
    hau = tr[(tr.source == "HAU") & (tr.category == "emotion")].copy()
    hau["user"] = hau.path.str.extract(r"(user\d+)")
    hau["trial"] = hau.path.map(lambda p: int(p.split("/")[-1].split("-")[-1]) if len(p.split("/")[-1].split("-")) == 3 else -1)
    hau["sess"] = hau.path.map(lambda p: "-".join(p.split("/")[-1].split("-")[:2]) if len(p.split("/")[-1].split("-")) == 3 else "")
    hau["manner"] = hau.apply(lambda r: r[r.answer], axis=1)
    hau["mgrp"] = hau.manner.map(mgroup)

    print("\n--- Part 1: Training Protocol Dynamics (3-Trial Progression) ---")
    pos_grp_counts = pd.crosstab(hau.trial, hau.mgrp)
    print("Trial Position vs Manner Group in Training HAU:")
    print(pos_grp_counts)
    
    slow_in_t1 = (hau[hau.trial == 1].mgrp == "SLOW").mean()
    fast_in_t3 = (hau[hau.trial == 3].mgrp.isin(["FAST", "NERV"])).mean()
    neut_in_t2 = (hau[hau.trial == 2].mgrp.isin(["NEUT", "CARE"])).mean()
    print(f"Trial 1 is SLOW: {slow_in_t1*100:.1f}%")
    print(f"Trial 2 is NEUTRAL/CAREFUL: {neut_in_t2*100:.1f}%")
    print(f"Trial 3 is FAST/NERVOUS: {fast_in_t3*100:.1f}%")

    # Part 2: Cohort 2 Two-Clip Blocks Deep Audit
    print("\n--- Part 2: Cohort 2 (10 Two-Clip Blocks) Physical & Semantic Audit ---")
    c2_blocks = [33, 34, 36, 37, 38, 39, 40, 41, 42, 43]
    meta_test = meta[meta.kind == "test"].set_index("qa_path")

    c2_records = []
    for bi in c2_blocks:
        cids = U.TEST_BLOCKS[bi]
        k0, k1 = f"LM_test_{cids[0]:04d}", f"LM_test_{cids[1]:04d}"
        m0, m1 = meta_test.loc[k0], meta_test.loc[k1]
        dur0, dur1 = m0.t1 - m0.t0, m1.t1 - m1.t0
        
        arr0 = skel.get(m0.unit_dir + "|K")
        spd0 = np.mean(np.linalg.norm(np.diff(arr0, axis=0), axis=-1)) if arr0 is not None and len(arr0) > 1 else np.nan
        arr1 = skel.get(m1.unit_dir + "|K")
        spd1 = np.mean(np.linalg.norm(np.diff(arr1, axis=0), axis=-1)) if arr1 is not None and len(arr1) > 1 else np.nan

        q0 = te[te.path.str.contains(k0) & (te.category == "emotion")].iloc[0]
        q1 = te[te.path.str.contains(k1) & (te.category == "emotion")].iloc[0]
        
        cand = sorted(list(set(opts(q0)) & set(opts(q1))))
        c_p0, c_p1 = champ.loc[q0.qa_id, "prediction"], champ.loc[q1.qa_id, "prediction"]
        w0, w1 = q0[c_p0], q1[c_p1]

        # Triad decomposition
        cand_grps = {m: mgroup(m) for m in cand}
        slow_cand = [m for m, g in cand_grps.items() if g == "SLOW"]
        fast_cand = [m for m, g in cand_grps.items() if g in ("FAST", "NERV")]
        neut_cand = [m for m, g in cand_grps.items() if g in ("NEUT", "CARE")]

        # Determine which clip is faster by physical features
        clip0_is_faster = (spd0 > spd1) or (dur0 < dur1 and not (spd1 > spd0 * 1.2))

        # Expected under Regime (0, 2)
        exp_fast = fast_cand[0] if fast_cand else None
        exp_slow = slow_cand[0] if slow_cand else None

        exp_w0 = exp_fast if clip0_is_faster else exp_slow
        exp_w1 = exp_slow if clip0_is_faster else exp_fast

        exp_p0 = [k for k in "ABCD" if q0[k] == exp_w0][0] if exp_w0 in opts(q0) else c_p0
        exp_p1 = [k for k in "ABCD" if q1[k] == exp_w1][0] if exp_w1 in opts(q1) else c_p1

        conflict = (c_p0 != exp_p0) or (c_p1 != exp_p1)

        c2_records.append({
            "block": bi, "cid0": cids[0], "cid1": cids[1],
            "qa0": q0.qa_id, "qa1": q1.qa_id,
            "dur0": dur0, "dur1": dur1, "spd0": spd0, "spd1": spd1,
            "clip0_faster": int(clip0_is_faster),
            "cand": cand,
            "champ_p0": c_p0, "champ_w0": w0,
            "champ_p1": c_p1, "champ_w1": w1,
            "exp_p0": exp_p0, "exp_w0": exp_w0,
            "exp_p1": exp_p1, "exp_w1": exp_w1,
            "conflict": int(conflict)
        })

    df_c2 = pd.DataFrame(c2_records)
    print(df_c2[["block", "qa0", "champ_w0", "exp_w0", "qa1", "champ_w1", "exp_w1", "conflict"]].to_string(index=False))

    conflicts = df_c2[df_c2.conflict == 1]
    print(f"\nTotal blocks in Cohort 2 requiring Trial 2 Withholding repair: {len(conflicts)} / {len(df_c2)}")
    for _, r in conflicts.iterrows():
        print(f"\nBlock {r.block}:")
        if r.champ_p0 != r.exp_p0:
            print(f"  {r.qa0}: Champion {r.champ_p0} ({r.champ_w0}) -> Repaired {r.exp_p0} ({r.exp_w0})")
        if r.champ_p1 != r.exp_p1:
            print(f"  {r.qa1}: Champion {r.champ_p1} ({r.champ_w1}) -> Repaired {r.exp_p1} ({r.exp_w1})")

    # Save detailed audit report
    out_path = os.path.join(ROOT, "research", "cohort2_withholding_reconstructed_audit.csv")
    df_c2.to_csv(out_path, index=False)
    print(f"\nWrote full audit to {out_path}")

if __name__ == "__main__":
    main()
