"""Decision-level audit for the exact-segment VideoMAE challenger.

This never selects test rows.  It screens one held-subject fold against the strongest
available OOF incumbents and emits fixed top-k margin diagnostics for replication on a
second fold if (and only if) the challenger has positive net disagreements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def transition(frame: pd.DataFrame) -> dict[str, int]:
    model_ok = frame.model_ok
    base_ok = frame.base_ok
    changed = frame.prediction.ne(frame.base_prediction)
    return {
        "shared": int(len(model_ok)),
        "model_correct": int(model_ok.sum()),
        "base_correct": int(base_ok.sum()),
        "prediction_disagreements": int(changed.sum()),
        "wins_model_correct_base_wrong": int((changed & model_ok & ~base_ok).sum()),
        "losses_model_wrong_base_correct": int((changed & ~model_ok & base_ok).sum()),
        "changed_both_wrong": int((changed & ~model_ok & ~base_ok).sum()),
        "net": int(model_ok.sum() - base_ok.sum()),
    }


def topk_margin(frame: pd.DataFrame, margin: str, model_ok: str, base_ok: str) -> list[dict]:
    disagree = frame[frame.prediction != frame.base_prediction].sort_values(margin, ascending=False)
    rows = []
    for k in (1, 2, 3, 5, 8, 10, 15, 20):
        part = disagree.head(k)
        if len(part) < k:
            continue
        wins = int((part[model_ok] & ~part[base_ok]).sum())
        losses = int((~part[model_ok] & part[base_ok]).sum())
        both_wrong = int((~part[model_ok] & ~part[base_ok]).sum())
        rows.append({
            "k": k,
            "minimum_margin": float(part[margin].min()),
            "wins": wins,
            "losses": losses,
            "both_wrong": both_wrong,
            "net": wins - losses,
            "precision": float(wins / k),
        })
    return rows


def option_margin(value: str) -> float:
    scores = sorted(json.loads(value), reverse=True)
    return float(scores[0] - scores[1]) if len(scores) >= 2 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    seq_path = args.output_dir / "videomae_action_timeline_fold0_predictions.csv"
    harn_path = args.output_dir / "videomae_action_fold0_mc_predictions.csv"
    seq = pd.read_csv(seq_path)
    harn = pd.read_csv(harn_path)

    mechanism = pd.read_csv(args.repo / "seqlab" / "audit_run18.csv").rename(
        columns={"qa": "qa_id", "ans": "truth", "new": "base_prediction"}
    )
    seq_joint = seq.merge(
        mechanism[["qa_id", "truth", "base_prediction"]],
        on=["qa_id", "truth"], validate="one_to_one",
    )
    seq_joint["model_ok"] = seq_joint.prediction.eq(seq_joint.truth)
    seq_joint["base_ok"] = seq_joint.base_prediction.eq(seq_joint.truth)

    baseline = pd.read_csv(args.repo / "champ" / "oof_e2e_baseline_20260909.csv")
    baseline = baseline[(baseline.source == "HARn") & (baseline.category == "single")].rename(
        columns={"pred": "base_prediction", "answer": "truth"}
    )
    harn_joint = harn.merge(
        baseline[["qa_id", "truth", "base_prediction"]],
        on=["qa_id", "truth"], validate="one_to_one",
    )
    harn_joint["margin"] = harn_joint.option_logits.map(option_margin)
    harn_joint["model_ok"] = harn_joint.prediction.eq(harn_joint.truth)
    harn_joint["base_ok"] = harn_joint.base_prediction.eq(harn_joint.truth)

    report = {
        "protocol": "subject-disjoint fold-0; descriptive gates require fold-1 replication",
        "sequence": {
            **transition(seq_joint),
            "topk_model_margin": topk_margin(seq_joint, "margin", "model_ok", "base_ok"),
        },
        "harn_single": {
            **transition(harn_joint),
            "topk_model_margin": topk_margin(harn_joint, "margin", "model_ok", "base_ok"),
        },
    }
    seq_joint.to_csv(args.output_dir / "videomae_action_timeline_fold0_joint_audit.csv", index=False)
    harn_joint.to_csv(args.output_dir / "videomae_action_fold0_mc_joint_audit.csv", index=False)
    summary_path = args.output_dir / "videomae_action_timeline_fold0_joint_audit.summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
