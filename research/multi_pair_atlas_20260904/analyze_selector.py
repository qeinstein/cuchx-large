"""Nested selector audit for champion-vs-best-X Multi pool choices.

The atlas is deliberately built before this file is run.  This script never regenerates a
submission and never reads test labels.  It evaluates a small, predeclared family of
interpretable pair-block gates, selecting each held-out fold's thresholds from the other
four subject-disjoint folds.  It reports both action decisions and exact Multi questions.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ATLAS = HERE / "action_atlas_oof_ends.csv"
TEST_ATLAS = HERE / "action_atlas_test.csv"
CHAMPION = ROOT / "submission_093859_SUBMITTED.csv"


def prep(d: pd.DataFrame, oof: bool) -> pd.DataFrame:
    d = d.copy()
    d["one_action_add"] = (
        d.best_x_removed_actions.fillna("").eq("")
        & d.best_x_added_actions.fillna("").ne("")
        & d.best_x_added_actions.fillna("").map(lambda s: str(s).count("||") == 0)
    )
    d["eligible_pair"] = (
        d.action_in_this_multi_options.eq(1)
        & d.champion_pool_member.eq(0)
        & d.best_pool_containing_action.notna()
        & d.block_regime.eq("pair")
    )
    if oof:
        d["action_positive"] = d.ground_truth_multi_member.eq(1).astype(int)
        d["question_wins"] = ((d.champion_multi_correct.eq(0)) & d.best_x_multi_correct.eq(1)).astype(int)
        d["question_losses"] = ((d.champion_multi_correct.eq(1)) & d.best_x_multi_correct.eq(0)).astype(int)
    return d


def rule_mask(d: pd.DataFrame, params: dict) -> pd.Series:
    m = d.eligible_pair & d.one_action_add
    if params.get("gap") is not None:
        m &= d.selected_minus_best_x_score <= params["gap"]
    if params.get("dino") is not None:
        m &= d.dense_dino_probability_max >= params["dino"]
    if params.get("pool") is not None:
        m &= d.pool_probability >= params["pool"]
    return m


def choose_nested(train: pd.DataFrame, min_n: int = 3):
    """Choose a rule from training folds only.

    Selection priority is explicit: among gates with >=80% action precision and at least
    ``min_n`` action candidates, retain the gate with the most training candidates, then
    largest net.  If no gate reaches that bar, maximize training net subject to min_n.  A
    held-out fold is never consulted while choosing its rule.
    """
    grid = []
    for gap, dino, pool in itertools.product(
        [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        [0.03, 0.05, 0.10, 0.20, 0.30, 0.50],
        [0.03, 0.05, 0.10, 0.20],
    ):
        params = {"gap": gap, "dino": dino, "pool": pool}
        s = train[rule_mask(train, params)]
        n = len(s)
        pos = int(s.action_positive.sum())
        if n < min_n:
            continue
        grid.append((pos / n, n, pos, 2 * pos - n, params))
    if not grid:
        return None, "no_rule"
    strong = [x for x in grid if x[0] >= 0.80]
    if strong:
        return max(strong, key=lambda x: (x[1], x[3], x[0], -x[4]["gap"])), "precision>=0.80"
    return max(grid, key=lambda x: (x[3], x[2], x[0], -x[1])), "max_net_fallback"


def choose_threshold_for_model(train: pd.DataFrame, scores: np.ndarray, min_n: int = 3):
    """Choose a score threshold under the same 0.80/minimum-n training gate."""
    z = train.copy()
    z["selector_score"] = scores
    rows = []
    for th in sorted(set(np.round(scores, 8)), reverse=True):
        s = z[z.selector_score >= th]
        n = len(s)
        if n < min_n:
            continue
        pos = int(s.action_positive.sum())
        rows.append((pos / n, n, pos, 2 * pos - n, float(th)))
    if not rows:
        return None, "no_threshold"
    strong = [x for x in rows if x[0] >= 0.80]
    if strong:
        return max(strong, key=lambda x: (x[1], x[3], x[0], x[4])), "precision>=0.80"
    return max(rows, key=lambda x: (x[3], x[2], x[0], -x[1])), "max_net_fallback"


def question_metrics(fired: pd.DataFrame, all_questions: pd.DataFrame):
    """Apply one selected candidate per question and return exact-answer metrics."""
    q = all_questions.drop_duplicates("qa_id").copy()
    if not len(q):
        return dict(question_candidates=0, questions_changed=0, wr=0, rw=0,
                    precision=np.nan, net=0, before=0, after=0, delta=0)
    q = q.set_index("qa_id")
    selected = fired.sort_values(
        ["selected_minus_best_x_score", "dense_dino_probability_max", "pool_probability", "action"],
        ascending=[True, False, False, True],
    ).drop_duplicates("qa_id")
    before = q.champion_multi_correct.astype(int).copy()
    after = before.copy()
    for r in selected.itertuples():
        if r.qa_id in after.index and pd.notna(r.best_x_multi_correct):
            after.loc[r.qa_id] = int(r.best_x_multi_correct)
    changed = before != after
    wr = int(((before == 0) & (after == 1)).sum())
    rw = int(((before == 1) & (after == 0)).sum())
    return dict(
        question_candidates=int(len(fired)),
        questions_changed=int(changed.sum()),
        wr=wr,
        rw=rw,
        precision=wr / max(1, wr + rw),
        net=wr - rw,
        before=int(before.sum()),
        after=int(after.sum()),
        delta=int(after.sum() - before.sum()),
    )


def action_metrics(fired: pd.DataFrame):
    n = len(fired)
    pos = int(fired.action_positive.sum()) if n else 0
    return dict(action_candidates=n, action_correct_inclusion=pos,
                action_false_inclusion=n - pos,
                action_precision=pos / max(1, n),
                action_net=2 * pos - n,
                action_net_per_100=100 * (2 * pos - n) / max(1, n))


def record(protocol, rule_name, fold, params, fired, all_questions):
    a = action_metrics(fired)
    q = question_metrics(fired, all_questions)
    return dict(protocol=protocol, rule=rule_name, fold=fold,
                parameters=json.dumps(params, sort_keys=True), **a, **q)


def main():
    oof = prep(pd.read_csv(ATLAS), True)
    test = prep(pd.read_csv(TEST_ATLAS), False)
    oof_candidates = oof[oof.eligible_pair].copy()
    oof_questions = oof[oof.block_regime.isin(["pair", "triple", "1-clip"])].copy()

    fixed = {"gap": 5.0, "dino": 0.10, "pool": 0.10}
    rows = []
    rows.append(record("exploratory_full_oof", "fixed_one_add_gap5_dino10_pool10", "all",
                        fixed, oof_candidates[rule_mask(oof_candidates, fixed)], oof_questions))
    for regime in ["pair", "triple", "1-clip"]:
        qreg = oof_questions[oof_questions.block_regime.eq(regime)]
        fired = oof_candidates[rule_mask(oof_candidates, fixed)]
        # The fixed rule is pair-only; this records its unchanged out-of-regime baseline.
        rows.append(record("exploratory_full_oof_by_regime", "fixed_one_add_gap5_dino10_pool10",
                            regime, fixed, fired[fired.block_regime.eq(regime)], qreg))

    # Nested subject-disjoint selection.  The candidate universe is pair-only, while all
    # Multi questions in the held-out fold are used for exact propagation.
    nested_fired = []
    for hold in sorted(oof_candidates.fold.unique()):
        train = oof_candidates[oof_candidates.fold.ne(hold)]
        valid = oof_candidates[oof_candidates.fold.eq(hold)]
        chosen, selection_reason = choose_nested(train, min_n=3)
        params = chosen[-1] if chosen else {}
        fired = valid[rule_mask(valid, params)] if chosen else valid.iloc[0:0]
        qvalid = oof_questions[oof_questions.fold.eq(hold)]
        r = record("nested_subject_disjoint", "grid_one_add_gap_dino_pool", int(hold),
                   {**params, "selection": selection_reason, "min_train_n": 3}, fired, qvalid)
        r["train_selected_n"] = int(chosen[1]) if chosen else 0
        r["train_selected_precision"] = float(chosen[0]) if chosen else np.nan
        r["train_selected_net"] = int(chosen[3]) if chosen else 0
        rows.append(r)
        if len(fired):
            nested_fired.append(fired.assign(held_fold=hold))
    nested = pd.concat(nested_fired, ignore_index=True) if nested_fired else oof_candidates.iloc[0:0]
    rows.append(record("nested_subject_disjoint_pooled", "grid_one_add_gap_dino_pool", "pooled",
                        {"min_train_n": 3}, nested, oof_questions))

    # A nested logistic selector is an additional model-based check.  The model only sees
    # test-visible fields and action identity; its threshold is learned on the other folds.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = ["selected_minus_best_x_score", "pool_probability",
           "dense_skel_imu_probability_max", "dense_dino_probability_max",
           "best_pool_containing_action_rank", "napp", "n_multi", "n_comb", "n_seq"]
    xcols = num + ["one_action_add", "action"]
    def frame(z):
        x = z[xcols].copy()
        x[num] = x[num].replace([np.inf, -np.inf], np.nan).fillna(0)
        x["one_action_add"] = x.one_action_add.astype(int)
        return x
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num + ["one_action_add"]),
        ("action", OneHotEncoder(handle_unknown="ignore"), ["action"]),
    ])
    model_fired = []
    for hold in sorted(oof_candidates.fold.unique()):
        train = oof_candidates[oof_candidates.fold.ne(hold)]
        valid = oof_candidates[oof_candidates.fold.eq(hold)]
        model = make_pipeline(pre, LogisticRegression(C=0.01, class_weight="balanced",
                                                       max_iter=2000, random_state=0))
        model.fit(frame(train), train.action_positive)
        tr_score = model.predict_proba(frame(train))[:, 1]
        chosen, reason = choose_threshold_for_model(train, tr_score, min_n=3)
        th = chosen[-1] if chosen else np.inf
        va_score = model.predict_proba(frame(valid))[:, 1]
        fired = valid[va_score >= th].copy() if chosen else valid.iloc[0:0]
        qvalid = oof_questions[oof_questions.fold.eq(hold)]
        r = record("nested_subject_disjoint", "logistic_test_visible_features", int(hold),
                   {"threshold": th, "selection": reason, "min_train_n": 3}, fired, qvalid)
        r["train_selected_n"] = int(chosen[1]) if chosen else 0
        r["train_selected_precision"] = float(chosen[0]) if chosen else np.nan
        r["train_selected_net"] = int(chosen[3]) if chosen else 0
        rows.append(r)
        if len(fired):
            model_fired.append(fired.assign(held_fold=hold))
    pooled_model = pd.concat(model_fired, ignore_index=True) if model_fired else oof_candidates.iloc[0:0]
    rows.append(record("nested_subject_disjoint_pooled", "logistic_test_visible_features", "pooled",
                        {"C": 0.01, "min_train_n": 3}, pooled_model, oof_questions))

    result = pd.DataFrame(rows)
    result.to_csv(HERE / "selector_results.csv", index=False)

    # Test-side output is intentionally only a reason ledger, not a submission.  The fixed
    # rule was selected for audit readability, but is not promoted unless its nested result
    # clears the project bar.
    tfired = test[rule_mask(test, fixed)].copy()
    if len(tfired):
        tfired[["qa_id", "block_id", "action", "champion_answer", "best_x_answer",
                "selected_minus_best_x_score", "pool_probability",
                "dense_skel_imu_probability_max", "dense_dino_probability_max",
                "best_pool_containing_action_rank", "best_x_added_actions",
                "best_x_removed_actions"]].assign(
                    rule="exploratory_fixed_one_add_gap5_dino10_pool10",
                    reason="pair block; one-action add; gap<=5; pool prob>=0.10; cached DINO prob>=0.10",
                ).to_csv(HERE / "test_overrides_exploratory.csv", index=False)
    else:
        pd.DataFrame(columns=["qa_id", "rule", "reason"]).to_csv(
            HERE / "test_overrides_exploratory.csv", index=False)

    s = json.loads((HERE / "summary.json").read_text())
    fixed_row = result[(result.protocol == "exploratory_full_oof")].iloc[0].to_dict()
    pooled_grid = result[(result.protocol == "nested_subject_disjoint_pooled") &
                         (result.rule == "grid_one_add_gap_dino_pool")].iloc[0].to_dict()
    pooled_log = result[(result.protocol == "nested_subject_disjoint_pooled") &
                        (result.rule == "logistic_test_visible_features")].iloc[0].to_dict()
    fn = oof[oof.false_negative_action.eq(1)]
    fn_available = fn.best_pool_containing_action.fillna("").ne("")
    lines = [
        "# Multi pair-block action-pool selector audit",
        "",
        f"Frozen champion: `{CHAMPION.name}` (SHA-256 `{s['champion_sha256']}`); no submission was made.",
        "",
        "## Atlas and residual",
        "",
        f"Matched protocol: five subject-disjoint folds, pair_frac={s['oof_protocol']['pair_frac']}, "
        f"withheld-trial policy `{s['oof_protocol']['withheld_trial_policy']}`, validated repair/conformance-first.",
        f"The atlas contains {s['oof_action_rows']} action rows across {s['oof_multi_questions']} Multi questions, "
        f"with {s['oof_false_negative_actions']} false-negative actions and {s['oof_spurious_actions']} spurious actions.",
        f"Among pair-block omitted actions with a satisfying best-X candidate, there are {len(oof_candidates)} "
        f"eligible rows and {int(oof_candidates.action_positive.sum())} genuine inclusions "
        f"({oof_candidates.action_positive.mean():.4f} action precision before gating).",
        f"Of the {len(fn)} false-negative actions, {int(fn_available.sum())} have a candidate pool containing X; "
        f"{int((~fn_available).sum())} are structurally unavailable under the current hard constraints.",
        "",
        "## Selector results",
        "",
        "The fixed gate is exploratory (chosen after inspecting the atlas): pair + one-action add + "
        "score gap <= 5 + pool probability >= 0.10 + cached DINO dense probability >= 0.10.",
        f"On the full OOF atlas it fires {int(fixed_row['action_candidates'])} action candidates, "
        f"{int(fixed_row['action_correct_inclusion'])} correct ({fixed_row['action_precision']:.3f}); "
        f"question-level W→R={int(fixed_row['wr'])}, R→W={int(fixed_row['rw'])}, "
        f"net={int(fixed_row['net'])}.",
        f"Nested grid selection pooled across held-out folds fires {int(pooled_grid['action_candidates'])} "
        f"actions, {int(pooled_grid['action_correct_inclusion'])} correct ({pooled_grid['action_precision']:.3f}); "
        f"question W→R={int(pooled_grid['wr'])}, R→W={int(pooled_grid['rw'])}, net={int(pooled_grid['net'])}.",
        f"Nested logistic selection pooled fires {int(pooled_log['action_candidates'])} actions, "
        f"{int(pooled_log['action_correct_inclusion'])} correct ({pooled_log['action_precision']:.3f}); "
        f"question W→R={int(pooled_log['wr'])}, R→W={int(pooled_log['rw'])}, net={int(pooled_log['net'])}.",
        "",
        "## Test-side inspection",
        "",
        f"The exploratory fixed gate identifies {len(tfired)} test action overrides across "
        f"{tfired.qa_id.nunique()} Multi questions. They are recorded in "
        "`test_overrides_exploratory.csv` only; no candidate submission is produced because the "
        "nested selector does not establish the required precision bar.",
        "",
        "Full action rows, candidate pools, nested metrics, and the frozen artifact hash are retained "
        "alongside this report for follow-up work.",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")

    print("wrote", HERE / "selector_results.csv")
    print(result[["protocol", "rule", "fold", "action_candidates", "action_correct_inclusion",
                  "action_precision", "wr", "rw", "net"]].to_string(index=False))
    print("test exploratory overrides", len(tfired), tfired.qa_id.nunique())


if __name__ == "__main__":
    main()
