"""Test whether anonymised HAU test cohorts are known training subjects.

Uses only action-invariant skeleton morphology (median bone-length ratios).  Model choice is
made by leave-one-session-out training-user identification.  The already established exact
visible cohort matches for user20, user21 and user1 are used only as a test-side calibration;
the unknown 16-block cohort is never manually labelled.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
import evaluate_user_template_pairs_20260904 as U  # noqa: E402


BONES = [
    (5, 6), (11, 12),             # shoulder/hip widths
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12),             # left/right torso
]


def descriptor(k: np.ndarray) -> np.ndarray | None:
    if len(k) < 4:
        return None
    lengths = np.stack([np.linalg.norm(k[:, a] - k[:, b], axis=1) for a, b in BONES], 1)
    med = np.nanmedian(lengths, axis=0)
    if not np.all(np.isfinite(med)) or np.median(med) <= 1e-6:
        return None
    log_len = np.log(np.maximum(med, 1e-6))
    # Ratios remove camera scale.  Bilateral asymmetries and absolute log scale are retained
    # as separate candidate feature sets rather than mixed into the invariant baseline.
    ratios = log_len - log_len.mean()
    asym = np.array([
        log_len[2] - log_len[4], log_len[3] - log_len[5],
        log_len[6] - log_len[8], log_len[7] - log_len[9],
        log_len[10] - log_len[11],
    ])
    return np.concatenate([ratios, asym, [log_len.mean()]])


def robust_scale(x: np.ndarray):
    center = np.nanmedian(x, axis=0)
    scale = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    scale[scale < 1e-5] = 1.0
    return center, scale


def centroid(rows: pd.DataFrame, cols):
    return np.nanmedian(np.stack(rows.desc.to_list())[:, cols], axis=0)


def distances(target: np.ndarray, references: dict[str, np.ndarray]):
    return sorted((float(np.mean(np.square(target - value))), user)
                  for user, value in references.items())


def main():
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    z = np.load(os.path.join(ROOT, "champ", "skel_seq.npz"))
    rows = []
    for r in meta.itertuples():
        if r.kind not in ("train_hau", "test"):
            continue
        if r.kind == "test" and int(str(r.qa_path).split("_")[-1]) < 65:
            continue
        key = r.unit_dir + "|K"
        if key not in z:
            continue
        value = descriptor(z[key])
        if value is None:
            continue
        if r.kind == "train_hau":
            bits = str(r.trial).split("-")
            block = "-".join(bits[:2])
            idx = -1
        else:
            block = "test"
            idx = int(str(r.qa_path).split("_")[-1])
        rows.append(dict(kind=r.kind, qa_path=r.qa_path, user=r.user,
                         block=block, idx=idx, desc=value))
    d = pd.DataFrame(rows)
    train = d[d.kind == "train_hau"].copy()
    test = d[d.kind == "test"].copy()

    all_x = np.stack(train.desc.to_list())
    center, scale = robust_scale(all_x)
    train["desc"] = [(x - center) / scale for x in train.desc]
    test["desc"] = [(x - center) / scale for x in test.desc]

    feature_sets = {
        "ratios": np.arange(12),
        "ratios_asym": np.arange(17),
        "ratios_asym_scale": np.arange(18),
    }
    validation_rows = []
    for name, cols in feature_sets.items():
        # Cohort-sized validation: each user's alternating sessions are identified from the
        # other half, with every competing user's full reference available.
        for phase in (0, 1):
            refs = {}
            targets = {}
            for user, group in train.groupby("user"):
                blocks = sorted(group.block.unique(), key=lambda x: tuple(map(int, x.split("-"))))
                target_blocks = set(blocks[phase::2])
                target_rows = group[group.block.isin(target_blocks)]
                ref_rows = group[~group.block.isin(target_blocks)]
                if len(target_rows) and len(ref_rows):
                    targets[user] = centroid(target_rows, cols)
                    refs[user] = centroid(ref_rows, cols)
            for user, target in targets.items():
                ranked = distances(target, refs)
                rank = [u for _, u in ranked].index(user) + 1
                validation_rows.append(dict(
                    feature_set=name, protocol="half_cohort", phase=phase,
                    user=user, truth=user, prediction=ranked[0][1], rank=rank,
                    correct=int(rank == 1), d1=ranked[0][0], d2=ranked[1][0],
                    margin=ranked[1][0] - ranked[0][0],
                ))

        # Harder single-session identification, retained as a robustness diagnostic.
        for (user, block), target_rows in train.groupby(["user", "block"]):
            refs = {}
            for other, group in train.groupby("user"):
                ref_rows = group if other != user else group[group.block != block]
                if len(ref_rows):
                    refs[other] = centroid(ref_rows, cols)
            ranked = distances(centroid(target_rows, cols), refs)
            rank = [u for _, u in ranked].index(user) + 1
            validation_rows.append(dict(
                feature_set=name, protocol="one_session", phase=-1,
                user=user, truth=user, prediction=ranked[0][1], rank=rank,
                correct=int(rank == 1), d1=ranked[0][0], d2=ranked[1][0],
                margin=ranked[1][0] - ranked[0][0],
            ))
    val = pd.DataFrame(validation_rows)
    vp = os.path.join(ROOT, "research", "hau_cohort_identity_validation_20260906.csv")
    val.to_csv(vp, index=False)

    summary = (val.groupby(["feature_set", "protocol"])
               .agg(n=("correct", "size"), correct=("correct", "sum"),
                    accuracy=("correct", "mean"), mean_rank=("rank", "mean"),
                    median_rank=("rank", "median"), median_margin=("margin", "median"))
               .reset_index())
    # Select only by training validation, never by agreement with known test cohorts.
    half = summary[summary.protocol == "half_cohort"].sort_values(
        ["accuracy", "mean_rank", "median_margin"], ascending=[False, True, False]
    )
    selected = str(half.iloc[0].feature_set)
    cols = feature_sets[selected]
    refs = {user: centroid(group, cols) for user, group in train.groupby("user")}

    cohort_specs = [
        (0, 14, "user20"),
        (14, 28, "user21"),
        (28, 44, None),
        (44, 55, "user1"),
    ]
    lengths = {user: len(blocks) for user, blocks in U.USER_BLOCKS.items()}
    cohort_rows = []
    for cohort, (lo, hi, visible_match) in enumerate(cohort_specs):
        ids = {int(i) for block in U.TEST_BLOCKS[lo:hi] for i in block}
        target_rows = test[test.idx.isin(ids)]
        target = centroid(target_rows, cols)
        ranked = distances(target, refs)
        eligible = {u for u, n in lengths.items() if n == hi - lo}
        constrained = [(dist, user) for dist, user in ranked if user in eligible]
        for rank, (dist, user) in enumerate(ranked, 1):
            cohort_rows.append(dict(
                cohort=cohort, lo=lo, hi=hi, n_blocks=hi-lo,
                n_clips=len(target_rows), feature_set=selected,
                visible_exact_match=visible_match or "", candidate=user,
                all_user_rank=rank, eligible=int(user in eligible), distance=dist,
                length_rank=([u for _, u in constrained].index(user) + 1
                             if user in eligible else np.nan),
            ))
    cohorts = pd.DataFrame(cohort_rows)
    cp = os.path.join(ROOT, "research", "hau_cohort_identity_test_20260906.csv")
    cohorts.to_csv(cp, index=False)

    print("validation")
    print(summary.to_string(index=False))
    print("selected from training only:", selected)
    print("\ntest cohort top-5 (all users)")
    print(cohorts[cohorts.all_user_rank <= 5].to_string(index=False))
    print("\nlength-constrained rankings")
    print(cohorts[cohorts.eligible == 1].sort_values(["cohort", "length_rank"]).to_string(index=False))
    print("wrote", vp, cp)


if __name__ == "__main__":
    main()
