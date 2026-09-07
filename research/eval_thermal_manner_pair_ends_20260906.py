"""Paired subject-disjoint audit of thermal manner evidence against champion S."""
from __future__ import annotations

import os
import sys

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
# The restored feature table intentionally has no DINO columns. Excluding that inert block
# avoids thousands of all-NaN diagnostics without changing the baseline decisions.
os.environ.setdefault("CHAMP_EMO_DINO", "0")
from core import (  # noqa: E402
    fit_group_model,
    fit_manner,
    infer_blocks,
    load_all,
    make_pseudo,
    solve_emotion,
)
from pseudotest import folds, thin_to_pairs  # noqa: E402


WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)


def transition(base: str, new: str, truth: str) -> str:
    if base == new:
        return "same"
    if base != truth and new == truth:
        return "W->R"
    if base == truth and new != truth:
        return "R->W"
    return "W->W"


def main() -> None:
    os.environ["CHAMP_EMO_R3D"] = "1"
    os.environ["CHAMP_EMO_R3D_CACHE"] = os.path.join(
        ROOT, "research", "thermal_mnv3_clip_features_20260906.npz"
    )
    tr, _, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    rows = []
    for fold, held in enumerate(folds(users, 5)):
        training = tr[~tr.user.isin(held)]
        evaluation = thin_to_pairs(tr, held, 0.38, seed=fold, policy="ends")
        visible, key, _ = make_pseudo(evaluation, meta, held, seed=100 + fold)
        blocks = infer_blocks(visible, fit_group_model(training))
        manner = fit_manner(training, meta)
        truth = dict(zip(key.qa_id, key.answer))
        emotional = visible[visible.category == "emotion"]
        predictions = {}
        for weight in WEIGHTS:
            os.environ["CHAMP_EMO_R3D"] = str(weight)
            predictions[weight], _ = solve_emotion(
                visible, blocks, manner, w_phys=1.0, w_pos=1.0, w_pair=1.0
            )
        for row in emotional.itertuples():
            record = dict(qa_id=row.qa_id, fold=fold, truth=truth[row.qa_id])
            for weight in WEIGHTS:
                record[f"pred_{weight:g}"] = predictions[weight].get(row.qa_id)
            rows.append(record)
        print("fold", fold, "done", flush=True)

    out = pd.DataFrame(rows)
    base = out["pred_0"]
    for weight in WEIGHTS:
        pred = out[f"pred_{weight:g}"]
        changed = pred != base
        shifts = [
            transition(b, n, t)
            for b, n, t in zip(base[changed], pred[changed], out.loc[changed, "truth"])
        ]
        print(
            "weight", weight, "correct", int((pred == out.truth).sum()), "/", len(out),
            "changed", int(changed.sum()), pd.Series(shifts).value_counts().to_dict(),
            "fold_net", {
                int(fold): int(
                    ((pred[g.index] == out.loc[g.index, "truth"]).sum())
                    - ((base[g.index] == out.loc[g.index, "truth"]).sum())
                )
                for fold, g in out.groupby("fold")
            },
        )
    out.to_csv(
        os.path.join(ROOT, "research", "thermal_manner_pair_ends_oof_20260906.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
