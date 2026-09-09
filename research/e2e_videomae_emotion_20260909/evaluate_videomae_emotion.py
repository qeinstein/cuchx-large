"""Subject-disjoint decision audit for VideoMAE emotion assignments."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


FAST = {"Quickly", "quickly", "Rapidly", "Hastily", "Hasitly", "Hurriedly", "Swiftly",
        "Briskly", "Urgently", "Frantically", "Impatiently", "Eagerly", "Forcefully"}
SLOW = {"Slowly", "Leisurely", "Unhurriedly", "Calmly", "Peacefully", "Relaxedly", "Lazily",
        "Gently", "Softly", "Quietly", "Comfortably", "Soothingly", "Patiently", "Lightly",
        "Contently", "Absentmindedly"}
CARE = {"Carefully", "Cautiously", "Meticulously", "Precisely", "Thoroughly", "Deliberately",
        "Methodically", "Attentively", "Intently", "Diligently", "Earnestly", "Neatly",
        "Orderly", "Seriously", "Serioiusly"}
NERV = {"Nervously", "Anxiously", "Tensely", "Tensly", "Restlessly"}
GROUPS = ["SLOW", "CARE", "NEUT", "NERV", "FAST"]
G2I = {value: index for index, value in enumerate(GROUPS)}


def manner_group(value: str) -> str:
    return "FAST" if value in FAST else "SLOW" if value in SLOW else \
        "CARE" if value in CARE else "NERV" if value in NERV else "NEUT"


def prepare_qa(path: Path) -> pd.DataFrame:
    qa = pd.read_csv(path)
    qa = qa[(qa.source == "HAU") & (qa.category == "emotion")].copy()
    parts = qa.path.str.extract(r"HAU/(user\d+)/(\d+-\d+)-(\d+)$")
    qa["user"], qa["session"], qa["clip"] = parts[0], parts[0] + "/" + parts[1], parts[2].astype(int)
    return qa


def candidates(rows: pd.DataFrame) -> list[str]:
    return sorted(set.intersection(*[
        {str(row[letter]).strip() for letter in "ABCD"} for _, row in rows.iterrows()
    ]))


def truth_manner(row: pd.Series) -> str:
    return str(row[str(row.answer)]).strip()


def position_prior(qa: pd.DataFrame, held_users: set[str]) -> dict[tuple[str, int, int], float]:
    exact, totals, grouped, group_totals = Counter(), Counter(), Counter(), Counter()
    train = qa[~qa.user.isin(held_users)]
    valid = []
    for _, rows in train.groupby("session"):
        rows = rows.sort_values("clip").drop_duplicates("path")
        common = set(candidates(rows))
        if len(rows) >= 2 and len(common) >= len(rows) and {truth_manner(r) for _, r in rows.iterrows()} <= common:
            valid.append(rows)
            k = len(rows)
            for pos, (_, row) in enumerate(rows.iterrows()):
                manner, group = truth_manner(row), manner_group(truth_manner(row))
                exact[(manner, pos, k)] += 1
                totals[(manner, k)] += 1
                grouped[(group, pos, k)] += 1
                group_totals[(group, k)] += 1
    result = {}
    manners = {truth_manner(row) for rows in valid for _, row in rows.iterrows()}
    for manner in manners:
        group = manner_group(manner)
        for k in (2, 3):
            for pos in range(k):
                back = (grouped[(group, pos, k)] + 1.0) / (group_totals[(group, k)] + k)
                result[(manner, pos, k)] = (exact[(manner, pos, k)] + 3.0 * back) / (totals[(manner, k)] + 3.0)
    return result


def corrected_decode(pred: pd.DataFrame, qa: pd.DataFrame) -> pd.DataFrame:
    held_users = set(pred.user.dropna().unique())
    prior = position_prior(qa, held_users)
    qrows = qa.set_index("qa_id")
    output = []
    for (_, session), frame in pred.groupby(["regime", "session"], sort=False):
        frame = frame.copy()
        rows = qrows.loc[frame.qa_id].sort_values("clip")
        frame = frame.set_index("qa_id").loc[rows.index].reset_index()
        common, k = candidates(rows), len(rows)
        logits = np.stack(frame.group_logits.map(json.loads))
        logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        scored = []
        for assignment in itertools.permutations(common, k):
            if any(assignment[i] not in {str(rows.iloc[i][letter]).strip() for letter in "ABCD"}
                   for i in range(k)):
                continue
            score = sum(logp[i, G2I[manner_group(assignment[i])]] for i in range(k))
            score += 0.30 * sum(math.log(max(prior.get((assignment[i], i, k), 1 / k), 1e-6))
                                for i in range(k))
            scored.append((float(score), assignment))
        scored.sort(reverse=True)
        best = scored[0][1]
        margin = scored[0][0] - scored[1][0]
        for i, (_, row) in enumerate(rows.iterrows()):
            letter = next(letter for letter in "ABCD" if str(row[letter]).strip() == best[i])
            item = frame.iloc[i].to_dict()
            item["corrected_prediction"] = letter
            item["corrected_assignment_margin"] = margin
            output.append(item)
    return pd.DataFrame(output)


def evaluate(frame: pd.DataFrame, baseline: pd.DataFrame,
             prediction: str, margin: str) -> tuple[pd.DataFrame, dict]:
    joint = frame.merge(
        baseline[["qa_id", "truth", "base_prediction"]],
        on=["qa_id", "truth"], validate="one_to_one",
    )
    joint["model_ok"] = joint[prediction].eq(joint.truth)
    joint["base_ok"] = joint.base_prediction.eq(joint.truth)
    changed = joint[prediction].ne(joint.base_prediction)
    wins = changed & joint.model_ok & ~joint.base_ok
    losses = changed & ~joint.model_ok & joint.base_ok
    both_wrong = changed & ~joint.model_ok & ~joint.base_ok
    report = {
        "shared": len(joint),
        "model_correct": int(joint.model_ok.sum()),
        "base_correct": int(joint.base_ok.sum()),
        "prediction_disagreements": int(changed.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "both_wrong": int(both_wrong.sum()),
        "net": int(joint.model_ok.sum() - joint.base_ok.sum()),
        "fixed_complete_session_gate_margin_ge_0_5": {},
        "topk_assignment_margin": [],
    }
    ranked = joint[changed].sort_values(margin, ascending=False)
    gated = ranked[ranked[margin] >= 0.5]
    gated_wins = int((gated.model_ok & ~gated.base_ok).sum())
    gated_losses = int((~gated.model_ok & gated.base_ok).sum())
    report["fixed_complete_session_gate_margin_ge_0_5"] = {
        "changes": len(gated),
        "wins": gated_wins,
        "losses": gated_losses,
        "both_wrong": int((~gated.model_ok & ~gated.base_ok).sum()),
        "net": gated_wins - gated_losses,
    }
    for k in (1, 2, 3, 5, 8, 10, 15, 20):
        part = ranked.head(k)
        if len(part) < k:
            continue
        w = int((part.model_ok & ~part.base_ok).sum())
        loss = int((~part.model_ok & part.base_ok).sum())
        report["topk_assignment_margin"].append({
            "k": k,
            "minimum_margin": float(part[margin].min()),
            "wins": w,
            "losses": loss,
            "both_wrong": int((~part.model_ok & ~part.base_ok).sum()),
            "net": w - loss,
            "precision": float(w / k),
        })
    return joint, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    qa = prepare_qa(args.repo / "training_qa.csv")
    pred = corrected_decode(pred, qa)
    base = pd.read_csv(args.repo / "champ" / "oof_e2e_baseline_20260909.csv")
    base = base[(base.source == "HAU") & (base.category == "emotion")].rename(
        columns={"answer": "truth", "pred": "base_prediction"}
    )
    summary = {
        "protocol": "subject-disjoint fold-0; any margin gate requires fold-1 replication",
        "regimes": {},
    }
    output_dir = args.predictions.parent
    prefix = args.predictions.stem.removesuffix("_predictions")
    for regime, frame in pred.groupby("regime"):
        joint, remote = evaluate(frame, base, "prediction", "assignment_margin")
        _, corrected = evaluate(
            frame, base, "corrected_prediction", "corrected_assignment_margin"
        )
        summary["regimes"][regime] = {
            "remote_candidate_count_decode": remote,
            "corrected_observed_clip_position_decode": corrected,
        }
        joint.to_csv(output_dir / f"{prefix}_{regime}_joint_audit.csv", index=False)
    path = output_dir / f"{prefix}_joint_audit.summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
