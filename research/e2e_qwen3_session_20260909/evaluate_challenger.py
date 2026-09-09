"""Decision-level audit for a pulled end-to-end validation artifact.

This reports exact model-vs-baseline wins and losses on shared QA ids.  It deliberately does
not promote a challenger from aggregate accuracy alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def truth(row: dict[str, str]) -> str:
    return row.get("truth", row.get("answer", ""))


def prediction(row: dict[str, str]) -> str:
    return row.get("prediction", row.get("pred", ""))


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)

    def metrics(group):
        n = len(group)
        model_correct = sum(bool(row["model_correct"]) for row in group)
        baseline_correct = sum(bool(row["baseline_correct"]) for row in group)
        wins = sum(bool(row["model_correct"]) and not bool(row["baseline_correct"]) for row in group)
        losses = sum(not bool(row["model_correct"]) and bool(row["baseline_correct"]) for row in group)
        disagreements = sum(row["model_prediction"] != row["baseline_prediction"] for row in group)
        return {
            "n": n,
            "model_correct": model_correct,
            "model_accuracy": model_correct / n if n else None,
            "baseline_correct": baseline_correct,
            "baseline_accuracy": baseline_correct / n if n else None,
            "r_to_w_wins": wins,
            "w_to_r_losses": losses,
            "net": wins - losses,
            "disagreements": disagreements,
            "flip_precision": wins / (wins + losses) if wins + losses else None,
            "oracle_union_correct": baseline_correct + wins,
        }

    return {
        "overall": metrics(rows),
        "per_category": {
            category: metrics(group) for category, group in sorted(by_category.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("champ/oof_e2e_baseline_20260909.csv"),
    )
    parser.add_argument("--regime")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_rows = read_rows(args.predictions)
    if args.regime is not None:
        model_rows = [row for row in model_rows if row.get("regime") == args.regime]
    baseline_rows = read_rows(args.baseline)
    model_by_id = {row["qa_id"]: row for row in model_rows}
    baseline_by_id = {row["qa_id"]: row for row in baseline_rows}
    shared = sorted(model_by_id.keys() & baseline_by_id.keys())
    audited = []
    for qa_id in shared:
        model = model_by_id[qa_id]
        baseline = baseline_by_id[qa_id]
        expected = truth(model)
        baseline_expected = truth(baseline)
        if expected != baseline_expected:
            raise ValueError(f"truth mismatch for {qa_id}: {expected} vs {baseline_expected}")
        model_prediction = prediction(model)
        baseline_prediction = prediction(baseline)
        audited.append(
            {
                "qa_id": qa_id,
                "user": model.get("user", ""),
                "category": model.get("category", baseline.get("category", "")),
                "truth": expected,
                "model_prediction": model_prediction,
                "baseline_prediction": baseline_prediction,
                "model_correct": model_prediction == expected,
                "baseline_correct": baseline_prediction == expected,
            }
        )

    result = {
        "predictions": str(args.predictions),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "baseline": str(args.baseline),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "regime": args.regime,
        "model_rows": len(model_rows),
        "baseline_rows": len(baseline_rows),
        "shared_rows": len(shared),
        "model_only": len(model_by_id.keys() - baseline_by_id.keys()),
        "baseline_only": len(baseline_by_id.keys() - model_by_id.keys()),
        **summarize(audited),
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
