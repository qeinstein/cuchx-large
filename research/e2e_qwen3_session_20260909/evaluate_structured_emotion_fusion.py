"""Fuse session-VLM emotion votes with the structured baseline on complete triples.

The gate is intentionally narrow: full three-clip sessions only.  Pair-thinned validation
showed that VLM votes should not be used when a session member is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def absolute_answer(row, letter: str) -> str | None:
    if letter not in "ABCD":
        return None
    return str(getattr(row, letter)).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", nargs="+", required=True)
    parser.add_argument("--baseline", default=str(ROOT / "champ/oof_e2e_baseline_20260909.csv"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-weight", type=float, default=1.0)
    parser.add_argument("--qwen-weight", type=float, default=1.0)
    args = parser.parse_args()

    qa = pd.read_csv(ROOT / "training_qa.csv")
    qa = qa[(qa.source == "HAU") & (qa.category == "emotion")].copy()
    qa["user"] = qa.path.str.extract(r"(user\d+)")[0]
    qa["trial"] = qa.path.str.extract(r"/([^/]+)$")[0]
    qa[["aa", "bb", "clip"]] = qa.trial.str.extract(r"^(\d+)-(\d+)-(\d+)").astype(int)
    qrow = {row.qa_id: row for row in qa.itertuples()}

    baseline_path = Path(args.baseline)
    baseline_frame = pd.read_csv(baseline_path)
    pred_col = "pred" if "pred" in baseline_frame else "prediction"
    truth_col = "answer" if "answer" in baseline_frame else "truth"
    baseline = dict(zip(baseline_frame.qa_id, baseline_frame[pred_col]))
    truth = dict(zip(baseline_frame.qa_id, baseline_frame[truth_col]))

    qwen = []
    qwen_paths = [Path(path) for path in args.qwen]
    for path in qwen_paths:
        frame = pd.read_csv(path)
        if "regime" in frame:
            frame = frame[frame.regime == "full"]
        frame = frame[(frame.category == "emotion") & (frame.parsed == 1)]
        qwen.append(dict(zip(frame.qa_id, frame.prediction)))

    shared = set(baseline) & set(truth) & set(qrow)
    for model in qwen:
        shared &= set(model)
    data = qa[qa.qa_id.isin(shared)].copy()

    output = dict(baseline)
    audit = []
    for block_key, group in data.groupby(["user", "aa", "bb"], sort=True):
        # Complete sessions only.  This is the deployment gate established before fitting.
        rows = [qrow[qid] for qid in sorted(group.qa_id)]
        if len(rows) != 3 or len({row.clip for row in rows}) != 3:
            continue
        common = set.intersection(*[{absolute_answer(row, letter) for letter in "ABCD"} for row in rows])
        if len(common) < 3:
            continue
        candidates = sorted(common)
        choices = []
        for assignment in itertools.permutations(candidates, 3):
            base_votes = sum(
                value == absolute_answer(row, baseline[row.qa_id])
                for row, value in zip(rows, assignment)
            )
            model_votes = [
                sum(value == absolute_answer(row, model[row.qa_id]) for row, value in zip(rows, assignment))
                for model in qwen
            ]
            score = args.baseline_weight * base_votes + args.qwen_weight * sum(model_votes)
            choices.append((score, base_votes, tuple(model_votes), assignment))
        choices.sort(reverse=True)
        best = choices[0]
        margin = best[0] - choices[1][0]
        for row, value in zip(rows, best[3]):
            new = next(letter for letter in "ABCD" if absolute_answer(row, letter) == value)
            old = baseline[row.qa_id]
            output[row.qa_id] = new
            audit.append(
                {
                    "qa_id": row.qa_id,
                    "user": row.user,
                    "block": "/".join(map(str, block_key)),
                    "truth": truth[row.qa_id],
                    "old": old,
                    "new": new,
                    "changed": int(old != new),
                    "old_correct": int(old == truth[row.qa_id]),
                    "new_correct": int(new == truth[row.qa_id]),
                    "block_margin": margin,
                    "baseline_absolute": absolute_answer(row, old),
                    "new_absolute": value,
                    **{
                        f"qwen_{i}_letter": model[row.qa_id]
                        for i, model in enumerate(qwen)
                    },
                }
            )

    audit_frame = pd.DataFrame(audit)
    changed = audit_frame[audit_frame.changed == 1]
    summary = {
        "baseline": str(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "qwen": [{"path": str(path), "sha256": sha256(path)} for path in qwen_paths],
        "baseline_weight": args.baseline_weight,
        "qwen_weight_each": args.qwen_weight,
        "eligible_rows": len(audit_frame),
        "eligible_baseline_correct": int(audit_frame.old_correct.sum()),
        "eligible_fused_correct": int(audit_frame.new_correct.sum()),
        "net": int(audit_frame.new_correct.sum() - audit_frame.old_correct.sum()),
        "changed_rows": len(changed),
        "w_to_r": int(((changed.old_correct == 0) & (changed.new_correct == 1)).sum()),
        "r_to_w": int(((changed.old_correct == 1) & (changed.new_correct == 0)).sum()),
        "both_wrong": int(((changed.old_correct == 0) & (changed.new_correct == 0)).sum()),
        "per_user": {
            user: {
                "n": len(group),
                "baseline": int(group.old_correct.sum()),
                "fused": int(group.new_correct.sum()),
                "net": int(group.new_correct.sum() - group.old_correct.sum()),
            }
            for user, group in audit_frame.groupby("user")
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(out_path, index=False)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
