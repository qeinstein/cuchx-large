"""Evaluate per-question CTC likelihoods with the verified session total-order constraint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seqlab"))
from fast import Block, PERMS  # noqa: E402


def pair_score(pred: str, truth: str) -> tuple[int, int]:
    pp = {letter: i for i, letter in enumerate(pred)}
    tt = {letter: i for i, letter in enumerate(truth)}
    correct = total = 0
    for i, a in enumerate("ABCD"):
        for b in "ABCD"[i + 1 :]:
            correct += (pp[a] < pp[b]) == (tt[a] < tt[b])
            total += 1
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--output")
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    qa = pd.read_csv(ROOT / "training_qa.csv")
    qa = qa[qa.qa_id.isin(pred.qa_id)].copy()
    qa["user"] = qa.path.str.extract(r"(user\d+)")[0]
    qa["trial"] = qa.path.str.extract(r"/([^/]+)$")[0]
    qa[["aa", "bb"]] = qa.trial.str.extract(r"^(\d+)-(\d+)").astype(int)
    data = qa.merge(pred, on="qa_id", suffixes=("_qa", ""), validate="one_to_one")
    direct = int(data.correct.sum())
    direct_pc = direct_pn = 0
    for row in data.itertuples():
        pc, pn = pair_score(row.prediction, row.truth)
        direct_pc += pc
        direct_pn += pn

    joint_rows = []
    for block_key, group in data.groupby(["user", "aa", "bb"], sort=True):
        group = group.sort_values("qa_id")
        opts = [[str(getattr(row, letter)).strip() for letter in "ABCD"] for row in group.itertuples()]
        union = sorted(set().union(*[set(x) for x in opts]))
        block = Block(union, opts)
        terms = []
        for row in group.itertuples():
            raw = json.loads(row.permutation_scores)
            terms.append((1.0, np.array([float(raw[p]) for p in PERMS])))
        labels, score, all_scores = block.best([], terms)
        second = np.partition(all_scores, -2)[-2] if len(all_scores) > 1 else -np.inf
        for row, label in zip(group.itertuples(), labels):
            pc, pn = pair_score(label, row.truth)
            joint_rows.append(
                dict(
                    qa_id=row.qa_id,
                    block="/".join(map(str, block_key)),
                    truth=row.truth,
                    direct=row.prediction,
                    joint=label,
                    direct_correct=int(row.prediction == row.truth),
                    joint_correct=int(label == row.truth),
                    pair_correct=pc,
                    pair_total=pn,
                    block_margin=float(score - second),
                )
            )
    joint = pd.DataFrame(joint_rows)
    jc = int(joint.joint_correct.sum())
    jpc = int(joint.pair_correct.sum())
    jpn = int(joint.pair_total.sum())
    changed = joint[joint.direct != joint.joint]
    summary = {
        "n": len(data),
        "direct_exact": direct,
        "direct_accuracy": direct / len(data),
        "direct_pair_accuracy": direct_pc / direct_pn,
        "joint_exact": jc,
        "joint_accuracy": jc / len(joint),
        "joint_pair_accuracy": jpc / jpn,
        "joint_changes": len(changed),
        "joint_w_to_r": int(((changed.direct_correct == 0) & (changed.joint_correct == 1)).sum()),
        "joint_r_to_w": int(((changed.direct_correct == 1) & (changed.joint_correct == 0)).sum()),
        "historical_mechanism_s_fold0_exact": 47,
    }
    print(json.dumps(summary, indent=2))
    output = Path(args.output) if args.output else Path(args.predictions).with_name("ctc_temporal_joint_audit.csv")
    joint.to_csv(output, index=False)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
