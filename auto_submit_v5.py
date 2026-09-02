"""Automated submission runner for submission_v5.csv upon daily quota reset."""

import subprocess
import time


def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.stdout


def main():
    print("=== Submitting submission_v5.csv to Kaggle ===")
    submit_cmd = (
        'kaggle competitions submit -c cuhk-x-competition-large-model-track '
        '-f submission_v5.csv -m "v5 Pipeline: 1688d unified multimodal action engine '
        '+ combination-multi invariance proofs"'
    )
    out = run_cmd(submit_cmd)

    if "Successfully submitted" in out:
        print("Submission accepted! Waiting 15s for evaluation...")
        time.sleep(15)
        check_cmd = "kaggle competitions submissions cuhk-x-competition-large-model-track"
        run_cmd(check_cmd)
        print("\nChecking leaderboard rank...")
        run_cmd("kaggle competitions list --search cuhk-x")
    else:
        print("Submission not completed or quota exhausted:")
        print(out)


if __name__ == "__main__":
    main()
