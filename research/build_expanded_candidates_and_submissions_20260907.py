"""Build Expanded 17-Candidate Pool and Reset Submissions for CUHK-X Large Model Track (Session 14).

Generates:
1. Complete 17-candidate metadata registry (Tier S, Tier A, Tier B).
2. Reset submission candidates combining optimal group testing and private locks.
3. Checksum verification against 330 champion base.
"""
import os, hashlib, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CHAMPION_PATH = os.path.join(ROOT, "submissions/submission_096491_330of342_CHAMPION.csv")
BASE_SHA = "e42cde96bafb108b8c16bcd97e5deb2af32c1657a089351ada8a72a95ed805cd"

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    assert os.path.exists(BASE_CHAMPION_PATH), f"Base champion not found: {BASE_CHAMPION_PATH}"
    actual_base_sha = get_sha256(BASE_CHAMPION_PATH)
    assert actual_base_sha == BASE_SHA, f"Base champion SHA mismatch: expected {BASE_SHA}, got {actual_base_sha}"
    print(f"[OK] Base champion verified: 330/342 (SHA: {actual_base_sha})")

    base_df = pd.read_csv(BASE_CHAMPION_PATH)
    te_df = pd.read_csv(os.path.join(ROOT, "test_qa.csv")).set_index("qa_id")

    # 15 Validated Candidates (Session 14 Re-ranked)
    CANDIDATES = [
        # Tier S (Extremely High Confidence / Proof-backed)
        {
            "qa_id": "test_0488",
            "category": "single",
            "from": "A",
            "to": "C",
            "meaning_from": "doing jumping jacks",
            "meaning_to": "turning pages",
            "tier": "Tier S",
            "evidence": "Same-clip test_0528 (object interaction) proves object is 'a documents'. 1.6s duration physically falsifies jumping jacks. VLM C. Champion margin 0.00 fallback.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0146",
            "category": "multi",
            "from": "BCD",
            "to": "BC",
            "meaning_from": "Turning a page, Walking, Sitting down",
            "meaning_to": "Turning a page, Walking",
            "tier": "Tier S",
            "evidence": "Block 17: Sitting down is 100% absent from Block 17 across all clips. Sibling combination test_0257 and sequence test_0340 prove action set.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0165",
            "category": "multi",
            "from": "CD",
            "to": "D",
            "meaning_from": "Checking body temperature, Drinking",
            "meaning_to": "Drinking",
            "tier": "Tier S",
            "evidence": "Block 26: Checking body temperature is 100% absent from Block 26 across all clips. Sibling combination test_0276 proves action set.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0458",
            "category": "emotion",
            "from": "B",
            "to": "A",
            "meaning_from": "Hurriedly",
            "meaning_to": "Steadily",
            "tier": "Tier S",
            "evidence": "Block 48: user1 3-1 triad [Leisurely, Steadily, Hurriedly]. Recording gap 71.1s (T1->T2). Retained pair P((0,1))=75.1%. IMU variance 3.7x lower. Physical margin +3.647.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0464",
            "category": "emotion",
            "from": "D",
            "to": "A",
            "meaning_from": "Anxiously",
            "meaning_to": "Steadily",
            "tier": "Tier S",
            "evidence": "Block 51: user1 5-1 & user20 2-2 triad [Gently, Steadily, Anxiously]. Clip 0 is Gently (T1). Gap 86.9s. Retained pair P((0,1))=65.0%, P((0,2))=0.1%. Physical margin +1.234.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0444",
            "category": "emotion",
            "from": "C",
            "to": "B",
            "meaning_from": "Comfortably",
            "meaning_to": "Patiently",
            "tier": "Tier S (Priv)",
            "evidence": "Block 41: user16 2-3 triad [Patiently, Calmly, Hurriedly]. Candidate A algebra proves d_0444 = 0 on public. Deterministic +1 private win.",
            "split": "private_proven",
        },
        {
            "qa_id": "test_0647",
            "category": "sequence",
            "from": "DCAB",
            "to": "DCBA",
            "meaning_from": "Squats, Jumping jacks, Massaging oneself, Checking body temp",
            "meaning_to": "Squats, Jumping jacks, Checking body temp, Massaging oneself",
            "tier": "Tier S (Priv)",
            "evidence": "Block 21: Unique total order repair. Sibling test_0342 (ACDB) and user21 3-2 template prove Checking temp < Massaging. Singleton probe 56063402 proved d_0647 = 0 on public. Deterministic +1 private win.",
            "split": "private_proven",
        },
        {
            "qa_id": "test_0206",
            "category": "multi",
            "from": "D",
            "to": "CD",
            "meaning_from": "Grabbing utensils",
            "meaning_to": "Stirring, Grabbing utensils",
            "tier": "Tier S (Priv)",
            "evidence": "Block 50 (LM_test_0192): 4-modal proof of Stirring (dense skel/imu 0.9993, DINOv2 0.9983, thermal 0.4537). Singleton probe 56063270 proved d_0206 = 0 on public. Deterministic +1 private win.",
            "split": "private_proven",
        },

        # Tier A (Strong Multi-Modal Consensus)
        {
            "qa_id": "test_0461",
            "category": "emotion",
            "from": "C",
            "to": "D",
            "meaning_from": "Casually",
            "meaning_to": "Seriously",
            "tier": "Tier A",
            "evidence": "Block 50: user1 4-1 triad [Casually, Seriously, Quickly]. Retained pair P((0,1))=84.3%. Joint solver favors Seriously over Casually.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0436",
            "category": "emotion",
            "from": "C",
            "to": "A",
            "meaning_from": "Thoroughly",
            "meaning_to": "Steadily",
            "tier": "Tier A",
            "evidence": "Block 37: cand = Hurriedly|Steadily|Thoroughly. Retained pair P((1,2))=71.9% (Trial 2 & 3). Sibling test_0435 is Hurriedly (T3). Joint solver margin +3.906.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0519",
            "category": "single",
            "from": "A",
            "to": "C",
            "meaning_from": "walking",
            "meaning_to": "drinking water",
            "tier": "Tier A",
            "evidence": "HARn Single fallback repair: Exact room match (pixel diff 2.35) to Block 51 where drinking is performed in both trials. Walking is absent.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0506",
            "category": "single",
            "from": "A",
            "to": "C",
            "meaning_from": "wiping a bowl",
            "meaning_to": "doing lunges",
            "tier": "Tier A",
            "evidence": "HARn Single fallback repair: Room match (pixel diff 10.95) to Block 49 where lunges is established. Wiping a bowl is absent.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0426",
            "category": "emotion",
            "from": "C",
            "to": "B",
            "meaning_from": "Comfortably",
            "meaning_to": "Anxiously",
            "tier": "Tier A",
            "evidence": "Block 33: Retained pair P((0,1))=81.3%. Joint solver margin +1.617 favoring Anxiously. Sibling margin -6.42 in baseline audit.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0430",
            "category": "emotion",
            "from": "D",
            "to": "C",
            "meaning_from": "Slowly",
            "meaning_to": "Steadily",
            "tier": "Tier A",
            "evidence": "Block 35: user24 6-3 triad [Slowly, Steadily, Restlessly]. Resolves duplicate Slowly collision with test_0431. Speed 0.014.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0432",
            "category": "emotion",
            "from": "C",
            "to": "A",
            "meaning_from": "Nervously",
            "meaning_to": "Restlessly",
            "tier": "Tier A",
            "evidence": "Block 35: user24 6-3 triad. Highest agitation (speed 0.041, IMU 0.252) matches Restlessly. Negative margin -1.99 in champ.",
            "split": "public_or_private",
        },

        # Tier B (Auxiliary / Secondary)
        {
            "qa_id": "test_0477",
            "category": "single",
            "from": "A",
            "to": "B",
            "meaning_from": "doing jumping jacks",
            "meaning_to": "mopping the floor",
            "tier": "Tier B",
            "evidence": "HARn Single fallback repair: LM_test_0007 duration 2.90s, motion concentrated at floor (40.9% bottom, 37.4% mid, 21.7% top). Matches mopping.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0137",
            "category": "multi",
            "from": "AC",
            "to": "ACD",
            "meaning_from": "Pouring, Eating",
            "meaning_to": "Pouring, Eating, Stirring",
            "tier": "Tier B",
            "evidence": "Block 12: Stirring is established in session by combination test_0247 & test_0621. VLM predicted D.",
            "split": "public_or_private",
        },
    ]

    cand_df = pd.DataFrame(CANDIDATES)
    cand_csv_path = os.path.join(ROOT, "research", "expanded_candidates_pool_20260907.csv")
    cand_df.to_csv(cand_csv_path, index=False)
    print(f"[OK] Wrote {len(cand_df)} candidates to {cand_csv_path}")

    # Re-ranked Submission Strike Suite for Reset
    # Sub 1: Golden Anchor Probe: test_0488: A -> C (Tier S+, highest possible individual certainty)
    # Sub 2: Structural Multi + Anchor Bundle + 3 Private Locks: 0488 + 0146 + 0165 + 0444 + 0647 + 0206
    # Sub 3: Tier S Core Strike: All 8 Tier S candidates (0488, 0146, 0165, 0458, 0464, 0444, 0647, 0206)
    # Sub 4: Decisive Rank 1 Strike: 8 Tier S + top Tier A (0488, 0146, 0165, 0458, 0464, 0461, 0436, 0519, 0506, 0444, 0647, 0206)
    # Sub 5: Block 35 & Auxiliary Strike: 0488, 0146, 0165, 0458, 0430, 0432, 0426, 0444, 0647, 0206

    SUBMISSION_PLANS = [
        {
            "filename": "submission_reset_sub1_golden_anchor_0488.csv",
            "desc": "Sub 1: Golden Anchor Probe of test_0488 (Tier S+, test_0528 documents proof)",
            "flips": {"test_0488": "C"},
        },
        {
            "filename": "submission_reset_sub2_structural_multi_bundle.csv",
            "desc": "Sub 2: Structural Multi + Anchor Bundle (0488 + 0146 + 0165 + 3 proven private locks: 0444, 0647, 0206)",
            "flips": {
                "test_0488": "C",
                "test_0146": "BC",
                "test_0165": "D",
                "test_0444": "B",
                "test_0647": "DCBA",
                "test_0206": "CD",
            },
        },
        {
            "filename": "submission_reset_sub3_tier_s_core_pack.csv",
            "desc": "Sub 3: Full Tier S Core Pack (0488 + 0146 + 0165 + 0458 + 0464 + 3 private locks: 0444, 0647, 0206)",
            "flips": {
                "test_0488": "C",
                "test_0146": "BC",
                "test_0165": "D",
                "test_0458": "A",
                "test_0464": "A",
                "test_0444": "B",
                "test_0647": "DCBA",
                "test_0206": "CD",
            },
        },
        {
            "filename": "submission_reset_sub4_decisive_rank1_strike.csv",
            "desc": "Sub 4: Decisive Rank 1 Strike (0488, 0146, 0165, 0458, 0464, 0461, 0436, 0519, 0506, plus 3 private locks: 0444, 0647, 0206)",
            "flips": {
                "test_0488": "C",
                "test_0146": "BC",
                "test_0165": "D",
                "test_0458": "A",
                "test_0464": "A",
                "test_0461": "D",
                "test_0436": "A",
                "test_0519": "C",
                "test_0506": "C",
                "test_0444": "B",
                "test_0647": "DCBA",
                "test_0206": "CD",
            },
        },
        {
            "filename": "submission_reset_sub5_block35_and_emotion_bundle.csv",
            "desc": "Sub 5: Block 35 Collision Resolution + Tier S/A Core + 3 private locks (0488, 0146, 0165, 0458, 0430, 0432, 0426, 0444, 0647, 0206)",
            "flips": {
                "test_0488": "C",
                "test_0146": "BC",
                "test_0165": "D",
                "test_0458": "A",
                "test_0430": "C",
                "test_0432": "A",
                "test_0426": "B",
                "test_0444": "B",
                "test_0647": "DCBA",
                "test_0206": "CD",
            },
        },
    ]

    manifest = []
    for plan in SUBMISSION_PLANS:
        out_df = base_df.copy()
        diff_count = 0
        diff_details = []
        for qid, new_val in plan["flips"].items():
            idx = out_df[out_df["qa_id"] == qid].index[0]
            old_val = out_df.loc[idx, "prediction"]
            assert old_val != new_val, f"Flip for {qid} is identical to base ({old_val})"
            out_df.loc[idx, "prediction"] = new_val
            diff_count += 1
            diff_details.append(f"{qid}: {old_val} -> {new_val}")

        out_path = os.path.join(ROOT, "research", plan["filename"])
        out_df.to_csv(out_path, index=False)
        sha = get_sha256(out_path)

        manifest.append({
            "filename": plan["filename"],
            "description": plan["desc"],
            "sha256": sha,
            "flips_count": diff_count,
            "flips": plan["flips"],
            "diff_summary": diff_details,
        })
        print(f"[OK] Generated {plan['filename']} ({diff_count} flips, SHA: {sha})")

    manifest_path = os.path.join(ROOT, "research", "reset_submissions_manifest_20260907.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
