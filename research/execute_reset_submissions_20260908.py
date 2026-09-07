"""Execution and Monitoring Script for Optimized Adaptive Reset Strike Suite (2026-09-08).

Orchestrates the reset submissions for the CUHK-X Large Model Track:
  1. Verifies local file SHA-256 integrity against reset suite manifest.
  2. Monitors UTC time countdown to quota reset at 00:00:00 UTC.
  3. Dispatches submissions one by one (DO NOT AUTO-SUBMIT).
  4. Interfaces with adaptive_reset_decoder_20260908.py to determine next best move.

Usage:
  python research/execute_reset_submissions_20260908.py check                    # Verify files, quota, countdown
  python research/execute_reset_submissions_20260908.py status                   # Poll and display latest Kaggle submissions
  python research/execute_reset_submissions_20260908.py submit sub1_core_bundle  # Submit Sub 1 (Core 4-Candidate Bundle)
  python research/execute_reset_submissions_20260908.py submit sub2_structural   # Submit Sub 2 (Non-Emotion 3-Candidate Pack)
  python research/execute_reset_submissions_20260908.py submit <name>            # Submit specific candidate file from manifest
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
MANIFEST_PATH = os.path.join(ROOT, "research", "reset_suite_manifest_20260908.json")

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
    for name, item in manifest.items():
        fpath = os.path.join(ROOT, item["file"])
        if not os.path.exists(fpath):
            print(f"[FAIL] Missing file: {fpath}")
            all_ok = False
            continue
        actual_sha = get_sha256(fpath)
        if actual_sha != item["sha256"]:
            print(f"[FAIL] SHA mismatch for {item['file']}: expected {item['sha256']}, got {actual_sha}")
            all_ok = False
        else:
            print(f"[OK] {name:32s} (flips: {item['num_flips']}, SHA: {actual_sha[:16]}...)")
    return all_ok

def check_time_and_countdown():
    now_utc = datetime.now(timezone.utc)
    print(f"Current UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    reset_target = datetime(2026, 9, 8, 0, 0, 0, tzinfo=timezone.utc)
    delta = reset_target - now_utc
    if delta.total_seconds() > 0:
        hrs, rem = divmod(int(delta.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)
        print(f"Time to Quota Reset (00:00:00 UTC): {hrs:02d}h {mins:02d}m {secs:02d}s")
        return False, delta.total_seconds()
    else:
        print("[READY] Quota reset target (00:00:00 UTC) has been passed! Submissions active.")
        return True, 0

def get_kaggle_status():
    out, err, code = run_cmd(f"kaggle competitions submissions {COMPETITION_SLUG}")
    if code != 0:
        print(f"[ERROR] Failed to query Kaggle submissions:\n{err}")
        return None
    return out

def submit_by_name(target_name):
    manifest = load_manifest()
    matched_key = None
    for k in manifest.keys():
        if target_name.lower() in k.lower():
            matched_key = k
            break
            
    if not matched_key:
        print(f"[ERROR] Submission key '{target_name}' not found. Available keys: {list(manifest.keys())}")
        return False

    item = manifest[matched_key]
    fpath = os.path.join(ROOT, item["file"])
    actual_sha = get_sha256(fpath)
    assert actual_sha == item["sha256"], "SHA mismatch right before submission!"

    desc = f"{matched_key}: {', '.join(item['diffs'])}"
    print(f"\nSubmitting Key: {matched_key}")
    print(f"File: {item['file']}")
    print(f"Description: {desc}")
    print(f"SHA-256: {actual_sha}")

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
                if len(lines) > 2 and "COMPLETE" in lines[2]:
                    print("\n[EVALUATION COMPLETE]")
                    print("\n==================================================")
                    print("NEXT STEP: Run adaptive decoder with observed score delta:")
                    print("  python research/adaptive_reset_decoder_20260908.py --d1 <delta1>")
                    print("==================================================")
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
            print("Usage: python execute_reset_submissions_20260908.py submit <key_name>")
            sys.exit(1)
        submit_by_name(sys.argv[2])
    else:
        print(f"Unknown command: {sys.argv[1]}")

if __name__ == '__main__':
    main()
