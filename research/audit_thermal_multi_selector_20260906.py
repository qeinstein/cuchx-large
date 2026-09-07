"""Nested audit of thermal evidence on the exact Multi pool-omission candidate set."""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_DIR = os.path.join(ROOT, "research", "multi_pair_atlas_20260904")


def add_thermal(atlas: pd.DataFrame) -> pd.DataFrame:
    visual = pd.read_csv(
        os.path.join(ROOT, "research", "thermal_action_sequence_oof_20260906.csv")
    )
    visual = visual[visual.category == "multi"]
    qa = pd.read_csv(os.path.join(ROOT, "training_qa.csv")).set_index("qa_id")
    scores = {}
    for row in visual.itertuples():
        values = json.loads(row.score)
        source = qa.loc[row.qa_id]
        for letter, value in zip("ABCD", values):
            scores[(row.qa_id, str(source[letter]).strip())] = float(value)
    out = atlas.copy()
    out["thermal_probability_max"] = [
        scores.get((row.qa_id, row.action), np.nan) for row in out.itertuples()
    ]
    out["one_action_add"] = (
        out.best_x_removed_actions.fillna("").eq("")
        & out.best_x_added_actions.fillna("").ne("")
        & out.best_x_added_actions.fillna("").map(lambda value: str(value).count("||") == 0)
    )
    out["eligible"] = (
        out.action_in_this_multi_options.eq(1)
        & out.champion_pool_member.eq(0)
        & out.best_pool_containing_action.notna()
        & out.block_regime.eq("pair")
        & out.one_action_add
        & out.thermal_probability_max.notna()
    )
    out["positive"] = out.ground_truth_multi_member.eq(1).astype(int)
    return out


def metrics(fired: pd.DataFrame, questions: pd.DataFrame) -> dict:
    selected = fired.sort_values(
        ["selector_score", "selected_minus_best_x_score"], ascending=[False, True]
    ).drop_duplicates("qa_id")
    q = questions.drop_duplicates("qa_id").set_index("qa_id")
    before = q.champion_multi_correct.astype(int)
    after = before.copy()
    for row in selected.itertuples():
        after.loc[row.qa_id] = int(row.best_x_multi_correct)
    wr = int(((before == 0) & (after == 1)).sum())
    rw = int(((before == 1) & (after == 0)).sum())
    pos = int(fired.positive.sum())
    return dict(
        action_n=len(fired), action_positive=pos,
        action_precision=pos / max(1, len(fired)),
        questions_changed=len(selected), wr=wr, rw=rw, net=wr - rw,
    )


def choose_threshold(scores: np.ndarray, labels: np.ndarray, minimum: int = 3):
    choices = []
    for threshold in sorted(set(np.round(scores, 8)), reverse=True):
        selected = scores >= threshold
        n = int(selected.sum())
        if n < minimum:
            continue
        positive = int(labels[selected].sum())
        choices.append((positive / n, n, 2 * positive - n, float(threshold)))
    strong = [row for row in choices if row[0] >= 0.8]
    if strong:
        return max(strong, key=lambda row: (row[1], row[2], row[0], row[3])), "precision>=0.8"
    if choices:
        return max(choices, key=lambda row: (row[2], row[0], row[1], row[3])), "max_net"
    return None, "none"


NUMERIC = [
    "thermal_probability_max",
    "selected_minus_best_x_score",
    "pool_probability",
    "dense_skel_imu_probability_max",
    "dense_dino_probability_max",
    "best_pool_containing_action_rank",
    "co_max",
    "napp",
]


def matrix(frame: pd.DataFrame) -> np.ndarray:
    x = frame[NUMERIC].replace([np.inf, -np.inf], np.nan).to_numpy(float).copy()
    # Probabilities span many orders of magnitude. Their log transforms make the linear
    # selector sensitive to evidence ratios rather than arbitrary absolute increments.
    for column in (2, 3, 4):
        x[:, column] = np.log10(np.clip(x[:, column], 1e-8, 1.0))
    return x


def nested_logistic(candidates: pd.DataFrame, questions: pd.DataFrame):
    fold_rows, all_fired = [], []
    for held in sorted(candidates.fold.unique()):
        training = candidates[candidates.fold != held]
        validation = candidates[candidates.fold == held]
        inner_score, inner_label = [], []
        for inner in sorted(training.fold.unique()):
            fit = training[training.fold != inner]
            tune = training[training.fold == inner]
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=0),
            )
            model.fit(matrix(fit), fit.positive)
            inner_score.extend(model.predict_proba(matrix(tune))[:, 1])
            inner_label.extend(tune.positive)
        chosen, reason = choose_threshold(np.asarray(inner_score), np.asarray(inner_label))
        threshold = chosen[-1] if chosen else np.inf
        final = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=0),
        )
        final.fit(matrix(training), training.positive)
        score = final.predict_proba(matrix(validation))[:, 1]
        fired = validation.assign(selector_score=score)
        fired = fired[fired.selector_score >= threshold]
        fold_metric = metrics(fired, questions[questions.fold == held])
        fold_rows.append(dict(
            method="nested_logistic", fold=int(held), threshold=threshold,
            selection=reason, train_n=(chosen[1] if chosen else 0),
            train_precision=(chosen[0] if chosen else np.nan), **fold_metric,
        ))
        all_fired.append(fired.assign(held_fold=held))
    fired = pd.concat(all_fired, ignore_index=True)
    return pd.DataFrame(fold_rows), fired


def nested_grid(candidates: pd.DataFrame, questions: pd.DataFrame):
    fold_rows, all_fired = [], []
    grid = list(itertools.product(
        (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        (2.0, 4.0, 6.0, 8.0, 12.0),
        (0.0, 0.002, 0.01, 0.05, 0.1),
    ))
    for held in sorted(candidates.fold.unique()):
        training = candidates[candidates.fold != held]
        validation = candidates[candidates.fold == held]
        choices = []
        for thermal, gap, pool in grid:
            selected = training[
                (training.thermal_probability_max >= thermal)
                & (training.selected_minus_best_x_score <= gap)
                & (training.pool_probability >= pool)
            ]
            if len(selected) < 3:
                continue
            precision = selected.positive.mean()
            net = int(2 * selected.positive.sum() - len(selected))
            choices.append((precision, len(selected), net, thermal, gap, pool))
        strong = [row for row in choices if row[0] >= 0.8]
        if strong:
            chosen = max(strong, key=lambda row: (row[1], row[2], row[0], row[3], -row[4]))
            reason = "precision>=0.8"
        elif choices:
            chosen = max(choices, key=lambda row: (row[2], row[0], row[1]))
            reason = "max_net"
        else:
            chosen, reason = None, "none"
        if chosen:
            precision, count, _, thermal, gap, pool = chosen
            fired = validation[
                (validation.thermal_probability_max >= thermal)
                & (validation.selected_minus_best_x_score <= gap)
                & (validation.pool_probability >= pool)
            ].assign(selector_score=lambda z: z.thermal_probability_max)
        else:
            precision, count = np.nan, 0
            thermal, gap, pool = np.inf, -np.inf, np.inf
            fired = validation.iloc[0:0].assign(selector_score=np.nan)
        fold_metric = metrics(fired, questions[questions.fold == held])
        fold_rows.append(dict(
            method="nested_grid", fold=int(held), thermal=thermal, gap=gap, pool=pool,
            selection=reason, train_n=count, train_precision=precision, **fold_metric,
        ))
        all_fired.append(fired.assign(held_fold=held))
    fired = pd.concat(all_fired, ignore_index=True)
    return pd.DataFrame(fold_rows), fired


def main() -> None:
    atlas = add_thermal(pd.read_csv(os.path.join(ATLAS_DIR, "action_atlas_oof_ends.csv")))
    candidates = atlas[atlas.eligible].copy()
    questions = atlas[atlas.block_regime.isin(["pair", "triple", "1-clip"])].copy()
    print("eligible", len(candidates), "positives", int(candidates.positive.sum()))
    tables, ledgers = [], []
    for function in (nested_grid, nested_logistic):
        table, fired = function(candidates, questions)
        tables.append(table)
        ledgers.append(fired.assign(method=table.method.iloc[0]))
        print(table.to_string(index=False))
        print("pooled", function.__name__, metrics(fired, questions))
    pd.concat(tables, ignore_index=True).to_csv(
        os.path.join(ROOT, "research", "thermal_multi_selector_nested_20260906.csv"),
        index=False,
    )
    pd.concat(ledgers, ignore_index=True).to_csv(
        os.path.join(ROOT, "research", "thermal_multi_selector_fired_20260906.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
