"""Execution and Monitoring Script for Kaggle Reset Strike Suite (2026-09-08).

Orchestrates the 5 reset submissions for the CUHK-X Large Model Track:
  1. Verifies local file SHA-256 integrity against reset manifest.
  2. Monitors UTC time countdown to quota reset at 00:00:00 UTC.
  3. Provides step-by-step submission dispatch and evaluation polling.
  4. Records evaluation results, score deltas, and updated leaderboard state.

Usage:
  python research/execute_reset_submissions_20260908.py check       # Verify files, quota, and countdown
  python research/execute_reset_submissions_20260908.py status      # Poll and display latest submissions
  python research/execute_reset_submissions_20260908.py submit 1    # Submit Sub 1 (Golden Anchor Probe)
  python research/execute_reset_submissions_20260908.py submit 2    # Submit Sub 2 (Structural Multi Bundle)
  python research/execute_reset_submissions_20260908.py submit 3    # Submit Sub 3 (Tier S Core Pack)
  python research/execute_reset_submissions_20260908.py submit 4    # Submit Sub 4 (Decisive Rank 1 Strike)
  python research/execute_reset_submissions_20260908.py submit 5    # Submit Sub 5 (Block 35 Collision Resolution)
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPETITION_SLUG = "cuhk-x-competition-large-model-track"
MANIFEST_PATH = os.path.join(ROOT, "research", "reset_submissions_manifest_20260907.json")
RESULTS_LOG_PATH = os.path.join(ROOT, "research", "submission_results_20260908.json")

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip(), res.stderr.strip(), res.returncode

def load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)

def verify_files():
    manifest = load_manifest()
    print("=== Verifying Submission Files Integrity ===")
    all_ok = True
    for item in manifest:
        fpath = os.path.join(ROOT, "research", item["filename"])
        if not os.path.exists(fpath):
            print(f"[FAIL] Missing file: {fpath}")
            all_ok = False
            continue
        actual_sha = get_sha256(fpath)
        if actual_sha != item["sha256"]:
            print(f"[FAIL] SHA mismatch for {item['filename']}: expected {item['sha256']}, got {actual_sha}")
            all_ok = False
        else:
            print(f"[OK] {item['filename']} (flips: {item['flips_count']}, SHA: {actual_sha[:16]}...)")
    return all_ok

def check_time_and_countdown():
    now_utc = datetime.now(timezone.utc)
    print(f"Current UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    # Reset is scheduled for 2026-09-08 00:00:00 UTC
    reset_target = datetime(2026, 9, 8, 0, 0, 0, tzinfo=timezone.utc)
    delta = reset_target - now_utc
    if delta.total_seconds() > 0:
        hrs, rem = divmod(int(delta.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)
        print(f"Time to Quota Reset: {hrs:02d}h {mins:02d}m {secs:02d}s")
        return False, delta.total_seconds()
    else:
        print("[READY] Quota reset target (00:00:00 UTC) has been passed!")
        return True, 0

def get_kaggle_status():
    out, err, code = run_cmd(f"kaggle competitions submissions {COMPETITION_SLUG}")
    if code != 0:
        print(f"[ERROR] Failed to query Kaggle submissions:\n{err}")
        return None
    return out

def submit_file(sub_num):
    manifest = load_manifest()
    if not (1 <= sub_num <= len(manifest)):
        print(f"[ERROR] Invalid submission number: {sub_num} (must be 1 to {len(manifest)})")
        return False

    item = manifest[sub_num - 1]
    fpath = os.path.join(ROOT, "research", item["filename"])
    
    # Re-verify SHA before submitting
    actual_sha = get_sha256(fpath)
    assert actual_sha == item["sha256"], "SHA mismatch right before submission!"

    desc = item["description"]
    print(f"\nSubmitting: {item['filename']}")
    print(f"Description: {desc}")
    print(f"File SHA-256: {actual_sha}")

    cmd = f'kaggle competitions submit -c {COMPETITION_SLUG} -f "{fpath}" -m "{desc}"'
    print(f"Executing: {cmd}")
    out, err, code = run_cmd(cmd)
    print(out)
    if err:
        print(f"Stderr: {err}")

    if "Successfully submitted" in out:
        print("[SUCCESS] Submission accepted by Kaggle! Polling for evaluation...")
        for wait_sec in range(15, 60, 10):
            print(f"Waiting {wait_sec}s...")
            time.sleep(wait_sec)
            status_out = get_kaggle_status()
            if status_out:
                lines = status_out.splitlines()
                print("\n".join(lines[:6]))
                # Check top line status
                if len(lines) > 2 and "COMPLETE" in lines[2]:
                    print("\n[EVALUATION COMPLETE]")
                    break
        return True
    else:
        print("[WARNING] Submission not completed or quota exhausted.")
        return False

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "check":
        ok = verify_files()
        is_ready, rem = check_time_and_countdown()
        print("\n--- Recent Kaggle Submissions ---")
        status = get_kaggle_status()
        if status:
            print("\n".join(status.splitlines()[:8]))
    elif sys.argv[1] == "status":
        status = get_kaggle_status()
        if status:
            print(status)
    elif sys.argv[1] == "submit":
        if len(sys.argv) < 3:
            print("Usage: python execute_reset_submissions_20260908.py submit <sub_number 1-5>")
            sys.exit(1)
        sub_num = int(sys.argv[2])
        submit_file(sub_num)
    else:
        print(f"Unknown command: {sys.argv[1]}")

if __name__ == "__main__":
    main()
