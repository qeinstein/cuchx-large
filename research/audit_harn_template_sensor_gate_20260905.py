"""Subject-disjoint exact-template versus aggregate-sensor HARn single audit.

The template key is entirely QA-visible.  Sensor probabilities are trained only on users
outside the held-out fold.  The test ledger has no label-derived fields.
"""
from __future__ import annotations

import os
import sys
import re
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
sys.path.insert(0, os.path.join(ROOT, "research"))

import harn as H  # noqa: E402
from pseudotest import folds  # noqa: E402
import exact_visible_template_audit_20260905 as T  # noqa: E402


def add_user(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["user"] = d.path.str.extract(r"(user\d+)")
    return d


def template_prediction(row, donors):
    seen = donors.get(T.signature(row), [])
    if not seen:
        return None, 0, 0.0
    counts = Counter(seen)
    answer, votes = counts.most_common(1)[0]
    if votes != len(seen):
        return None, len(seen), votes / len(seen)
    return T.to_letters(row, answer), len(seen), 1.0


def sensor_scores(row, meta_row, clf, cols):
    prob = clf.predict_proba(meta_row[cols].to_numpy(float).reshape(1, -1))[0]
    classes = list(clf.classes_)
    has_sensor = np.isfinite(meta_row.f0)
    allowed = (set(classes) - H.NO_SENSOR_ACTIONS) if has_sensor else set(H.NO_SENSOR_ACTIONS)
    values = []
    for letter in "ABCD":
        option = str(row.get(letter, "nan")).strip()
        action = H.S2A.get(option)
        if action is None or action not in classes or action not in allowed:
            values.append(-1e12)
        else:
            values.append(float(np.log(max(prob[classes.index(action)], 1e-12))))
    if max(values) <= -1e11:
        return None
    order = np.argsort(values)[::-1]
    return dict(
        prediction="ABCD"[int(order[0])],
        margin=float(values[order[0]] - values[order[1]]),
        scores=values,
        prob=prob,
        classes=classes,
    )


def meta_key(path: str) -> str:
    """Training QA paths are metadata keys; test QA paths contain the LM id."""
    if str(path).startswith("HARn/"):
        return str(path)
    match = re.search(r"(LM_test_\d+)", str(path))
    return match.group(1) if match else str(path)


def build_row(row, split, fold, template, support, purity, sensor):
    truth = "".join(x for x in str(row.answer) if x in "ABCD") if split == "oof" else None
    ti = "ABCD".index(template)
    si = "ABCD".index(sensor["prediction"])
    sv = sensor["scores"]
    return dict(
        qa_id=row.qa_id,
        split=split,
        fold=fold,
        user=getattr(row, "user", None),
        support=support,
        purity=purity,
        truth=truth,
        sensor=sensor["prediction"],
        template=template,
        disagree=int(sensor["prediction"] != template),
        sensor_margin=sensor["margin"],
        template_logp=sv[ti],
        sensor_logp=sv[si],
        template_gap=float(sv[ti] - sv[si]),
        sensor_correct=(int(sensor["prediction"] == truth) if truth is not None else np.nan),
        template_correct=(int(template == truth) if truth is not None else np.nan),
    )


def main() -> None:
    tr = add_user(pd.read_csv(os.path.join(ROOT, "training_qa.csv")))
    te = pd.read_csv(os.path.join(ROOT, "test_qa.csv"))
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    mi = meta.set_index("qa_path")
    users = sorted(meta.loc[meta.kind == "train_harn", "user"].dropna().unique())

    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        clf, cols = H.fit_action_clf(meta, hold)
        donors = T.donor_map(tr[~tr.user.isin(hold)])
        target = tr[(tr.source == "HARn") & (tr.category == "single") & tr.user.isin(hold)]
        for _, r in target.iterrows():
            template, support, purity = template_prediction(r, donors)
            mk = meta_key(r.path)
            if template is None or mk not in mi.index:
                continue
            sensor = sensor_scores(r, mi.loc[mk], clf, cols)
            if sensor is not None:
                rows.append(build_row(r, "oof", fi, template, support, purity, sensor))
        print(f"fold {fi} rows {sum(x['fold'] == fi for x in rows)}", flush=True)

    oof = pd.DataFrame(rows)
    oof_path = os.path.join(ROOT, "research", "harn_template_sensor_gate_oof_20260905.csv")
    oof.to_csv(oof_path, index=False)

    clf, cols = H.fit_action_clf(meta, [])
    donors = T.donor_map(tr)
    test_rows = []
    current = pd.read_csv(os.path.join(ROOT, "submission_095614_327of342_CHAMPION.csv"))
    cur = current.set_index("qa_id").prediction.astype(str)
    target = te[(te.source == "HARn") & (te.category == "single")]
    for _, r in target.iterrows():
        template, support, purity = template_prediction(r, donors)
        mk = meta_key(r.path)
        if template is None or mk not in mi.index:
            continue
        sensor = sensor_scores(r, mi.loc[mk], clf, cols)
        if sensor is None:
            continue
        d = build_row(r, "test", -1, template, support, purity, sensor)
        d["champion"] = cur[r.qa_id]
        d["template_vs_champion"] = int(template != cur[r.qa_id])
        test_rows.append(d)
    test = pd.DataFrame(test_rows)
    test_path = os.path.join(ROOT, "research", "harn_template_sensor_gate_test_20260905.csv")
    test.to_csv(test_path, index=False)

    disagree = oof[oof.disagree == 1].copy()
    disagree["transition"] = np.where(
        (disagree.sensor_correct == 0) & (disagree.template_correct == 1), "W->R",
        np.where((disagree.sensor_correct == 1) & (disagree.template_correct == 0), "R->W", "W->W"),
    )
    print("\nOOF", len(oof), "sensor", int(oof.sensor_correct.sum()),
          "template", int(oof.template_correct.sum()))
    print("disagreements", len(disagree), disagree.transition.value_counts().to_dict())
    if len(disagree):
        print(disagree[["qa_id", "fold", "support", "truth", "sensor", "template",
                        "sensor_margin", "template_gap", "transition"]].to_string(index=False))
    print("\nTEST TEMPLATE/CHAMPION DISAGREEMENTS")
    print(test[test.template_vs_champion == 1].to_string(index=False))
    print("wrote", oof_path, test_path)


if __name__ == "__main__":
    main()
