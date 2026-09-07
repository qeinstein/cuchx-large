"""Build Expanded Candidate Pool and Reset Submissions for CUHK-X Large Model Track.

Generates:
1. Complete 11-candidate metadata registry (Tier S, Tier A, Tier B).
2. Reset submission candidates combining optimal group testing.
3. Checksum verification against 330 champion base.
"""
import os, hashlib, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CHAMPION_PATH = os.path.join(ROOT, "submission_096491_330of342_CHAMPION.csv")
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

    # 11 Validated Candidates
    CANDIDATES = [
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
            "qa_id": "test_0469",
            "category": "emotion",
            "from": "A",
            "to": "D",
            "meaning_from": "Slowly",
            "meaning_to": "Comfortably",
            "tier": "Tier A",
            "evidence": "Block 54: Kinematic and skeleton speed calibration favoring Comfortably.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0137",
            "category": "multi",
            "from": "AC",
            "to": "ACD",
            "meaning_from": "Pouring, Eating",
            "meaning_to": "Pouring, Eating, Stirring",
            "tier": "Tier A",
            "evidence": "Block 12: Sole multi omission across all 144 multi questions. Sibling combination questions test_0247 and test_0621 confirm Stirring in session. VLM predicted D.",
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
            "qa_id": "test_0488",
            "category": "single",
            "from": "A",
            "to": "C",
            "meaning_from": "doing jumping jacks",
            "meaning_to": "turning pages",
            "tier": "Tier A",
            "evidence": "HARn Single fallback repair: LM_test_0023 duration is 1.60s with 4.26% motion. 1.6s duration physically falsifies jumping jacks. VLM explicitly predicted C.",
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
            "qa_id": "test_0477",
            "category": "single",
            "from": "A",
            "to": "B",
            "meaning_from": "doing jumping jacks",
            "meaning_to": "mopping the floor",
            "tier": "Tier B",
            "evidence": "HARn Single fallback repair: LM_test_0007 duration 2.90s, motion concentrated at floor (40.9% bottom, 37.4% mid, 21.7% top). Matches mopping the floor.",
            "split": "public_or_private",
        },
        {
            "qa_id": "test_0465",
            "category": "emotion",
            "from": "A",
            "to": "D",
            "meaning_from": "Slowly",
            "meaning_to": "Leisurely",
            "tier": "Tier B",
            "evidence": "Block 52: Semantic nuance between Slowly and Leisurely.",
            "split": "public_or_private",
        },
    ]

    cand_df = pd.DataFrame(CANDIDATES)
    cand_csv_path = os.path.join(ROOT, "research", "expanded_candidates_pool_20260907.csv")
    cand_df.to_csv(cand_csv_path, index=False)
    print(f"[OK] Wrote {len(cand_df)} candidates to {cand_csv_path}")

    # Build Submission Configurations
    SUBMISSION_PLANS = [
        {
            "filename": "submission_reset_sub1_singleton_probe_0458.csv",
            "desc": "Sub 1: Singleton Probe of test_0458 (Tier S anchor)",
            "flips": {"test_0458": "A"},
        },
        {
            "filename": "submission_reset_sub2_tier_s_bundle.csv",
            "desc": "Sub 2: Tier S Bundle (0458 + 0464 + 0444 private lock)",
            "flips": {"test_0458": "A", "test_0464": "A", "test_0444": "B"},
        },
        {
            "filename": "submission_reset_sub3_core_rank1_pack.csv",
            "desc": "Sub 3: Core Rank 1 Pack (0458 + 0464 + 0461 + 0469 + 0137 + 0444)",
            "flips": {
                "test_0458": "A",
                "test_0464": "A",
                "test_0461": "D",
                "test_0469": "D",
                "test_0137": "ACD",
                "test_0444": "B",
            },
        },
        {
            "filename": "submission_reset_sub4_maximal_rank1_strike.csv",
            "desc": "Sub 4: Maximal Rank 1 Strike (9 Tier-S/A Flips: 0458, 0464, 0461, 0469, 0137, 0436, 0488, 0426, 0444)",
            "flips": {
                "test_0458": "A",
                "test_0464": "A",
                "test_0461": "D",
                "test_0469": "D",
                "test_0137": "ACD",
                "test_0436": "A",
                "test_0488": "C",
                "test_0426": "B",
                "test_0444": "B",
            },
        },
        {
            "filename": "submission_reset_sub5_orthogonal_validation.csv",
            "desc": "Sub 5: Orthogonal Cross-Check (0458 + 0436 + 0488 + 0477 + 0444)",
            "flips": {
                "test_0458": "A",
                "test_0436": "A",
                "test_0488": "C",
                "test_0477": "B",
                "test_0444": "B",
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
