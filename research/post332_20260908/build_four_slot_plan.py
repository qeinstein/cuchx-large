"""Build the exact four-slot adaptive suite relative to the verified 332 champion.

This script is offline and never calls Kaggle or submits a file.  It uses the public
constraint that the five rows changed from the 330 champion to the 332 champion have
total signed contribution +2.  A reversion's leaderboard delta is therefore the
negative of its signed contribution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/post332_20260908"
BASE = ROOT / "submissions/submission_097076_332of342_CHAMPION.csv"
OLD = ROOT / "submissions/submission_096491_330of342_CHAMPION.csv"
BASE_SHA = "25e79e1dae1149bdad81d081d1fad3a94db4e1eb88df7f00e91276e6d5668d56"
REVERSION_ORDER = ["test_0488", "test_0477", "test_0506"]
OMITTED = ["test_0519", "test_0501"]
NEW = {"test_0526": "C"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_variant(name: str, base_rows: list[dict[str, str]], changes: dict[str, str], qa: dict[str, dict[str, str]]) -> dict:
    predictions = {row["qa_id"]: row["prediction"] for row in base_rows}
    predictions.update(changes)
    for qid, value in predictions.items():
        assert qid in qa and all(char in "ABCD" and qa[qid].get(char) for char in value), (qid, value)
    path = OUT / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qa_id", "prediction"])
        writer.writeheader()
        writer.writerows({"qa_id": qid, "prediction": predictions[qid]} for qid in predictions)
    diff = [
        {"qa_id": qid, "from": row["prediction"], "to": predictions[qid]}
        for qid, row in ((row["qa_id"], row) for row in base_rows)
        if row["prediction"] != predictions[qid]
    ]
    diff_path = OUT / name.replace(".csv", ".vs332.diff.csv")
    with diff_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qa_id", "from", "to"])
        writer.writeheader()
        writer.writerows(diff)
    return {
        "file": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "changes": changes,
        "diff_vs_332": diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe", nargs=3, type=int, metavar=("E0488", "E0477", "E0506"),
                        help="optional observed reversion deltas; prints the branch")
    args = parser.parse_args()

    base_rows = read(BASE)
    old_rows = read(OLD)
    qa = {row["qa_id"]: row for row in read(ROOT / "test_qa.csv")}
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
    assert len(base_rows) == 682 and len(old_rows) == 682
    base = {row["qa_id"]: row["prediction"] for row in base_rows}
    old = {row["qa_id"]: row["prediction"] for row in old_rows}

    configs = {
        "S1_REVERT_0488.csv": {"test_0488": old["test_0488"]},
        "S2_REVERT_0477.csv": {"test_0477": old["test_0477"]},
        "S3_REVERT_0506.csv": {"test_0506": old["test_0506"]},
        "S4_IF_0488_WIN__REVERT_0488_PLUS_0526.csv": {"test_0488": old["test_0488"], **NEW},
        "S4_IF_0477_WIN__REVERT_0477_PLUS_0526.csv": {"test_0477": old["test_0477"], **NEW},
        "S4_IF_0506_WIN__REVERT_0506_PLUS_0526.csv": {"test_0506": old["test_0506"], **NEW},
        "S4_IF_REMAINING_GAIN__REVERT_0519_0501_PLUS_0526.csv": {
            "test_0519": old["test_0519"], "test_0501": old["test_0501"], **NEW
        },
        "S4_IF_REMAINING_NONPOSITIVE__0526_ONLY.csv": dict(NEW),
    }
    manifest = [write_variant(name, base_rows, changes, qa) for name, changes in configs.items()]
    plan = {
        "reference": {"score": "332/342", "sha256": BASE_SHA},
        "constraint": "d0488+d0477+d0506+d0519+d0501=+2; reversion delta e=-d",
        "slot_order": [
            "S1_REVERT_0488.csv",
            "S2_REVERT_0477.csv",
            "S3_REVERT_0506.csv",
            "adaptive S4 branch below",
        ],
        "s4_logic": {
            "if_any_early_reversion_delta_is_+1": "submit the matching S4_IF_*_WIN file; at most one can be +1 under the constraint",
            "else": "compute E_remaining=-2-(E0488+E0477+E0506); use S4_IF_REMAINING_GAIN when E_remaining>=0, otherwise S4_IF_REMAINING_NONPOSITIVE",
            "x": "test_0526 D->C; x is independent of the five-row sum and is learned from S4",
        },
        "files": manifest,
    }
    (OUT / "four_slot_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    if args.observe is not None:
        observed = dict(zip(REVERSION_ORDER, args.observe))
        if any(value not in (-1, 0, 1) for value in args.observe):
            raise SystemExit("each observed delta must be -1, 0, or +1")
        early = [qid for qid, value in observed.items() if value == 1]
        remaining = -2 - sum(args.observe)
        if len(early) > 1 or remaining not in (-2, -1, 0, 1):
            raise SystemExit("observations contradict the five-row signed-sum constraint")
        if early:
            branch = f"S4_IF_{early[0].removeprefix('test_')}_WIN__REVERT_{early[0].removeprefix('test_')}__PLUS_0526.csv".replace("__PLUS", "_PLUS")
        else:
            branch = (
                "S4_IF_REMAINING_GAIN__REVERT_0519_0501_PLUS_0526.csv"
                if remaining >= 0 else "S4_IF_REMAINING_NONPOSITIVE__0526_ONLY.csv"
            )
        print(json.dumps({"observed": observed, "remaining_reversion_delta": -2 - sum(args.observe), "next_file": branch}, indent=2))


if __name__ == "__main__":
    main()
