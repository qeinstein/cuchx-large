# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CUHK-X Large Model Track — Category-Mode Baseline
#
# Rule-based predictor: for each test QA sample, predict the most frequent training answer
# for its category. No internet, no API keys, no heavy models.

# %%
from __future__ import annotations

from pathlib import Path

import pandas as pd

COMPETITION = "cuhk-x-competition-large-model-track"
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")


def find_input_file(filename: str) -> Path | None:
    for root in [INPUT_ROOT, INPUT_ROOT / COMPETITION, Path("data/raw")]:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    matches = list(INPUT_ROOT.rglob(filename)) if INPUT_ROOT.exists() else []
    matches = [p for p in matches if COMPETITION in str(p)]
    if len(matches) == 1:
        return matches[0]
    return None


train_path = find_input_file("training_qa.csv")
test_path = find_input_file("test_qa.csv")
if train_path is None or test_path is None:
    raise FileNotFoundError("training_qa.csv or test_qa.csv not found in input")

train = pd.read_csv(train_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

required_train = {
    "qa_id", "source", "path", "category", "question", "A", "B", "C", "D", "answer"
}
required_test = {
    "qa_id", "source", "path", "category", "question", "A", "B", "C", "D"
}
assert not (required_train - set(train.columns)), train.columns.tolist()
assert not (required_test - set(test.columns)), test.columns.tolist()
assert len(test) > 0, "test_qa.csv is empty"

# Category-mode predictor
mode_map = (
    train.groupby("category")["answer"]
    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "A")
    .to_dict()
)

predictions = []
for _, row in test.iterrows():
    cat = str(row.get("category", "")).strip()
    predictions.append(mode_map.get(cat, "A"))

submission = pd.DataFrame({"qa_id": test["qa_id"].astype(str), "prediction": predictions})
assert submission.columns.tolist() == ["qa_id", "prediction"]
assert submission["prediction"].notna().all()

# Validate prediction format per category
LABELS = "ABCD"
SINGLE = {"single", "combination", "emotion", "object_interaction"}

def valid_pred(value: str, category: str) -> bool:
    value = str(value).strip().upper()
    if not value or any(ch not in LABELS for ch in value):
        return False
    if len(set(value)) != len(value):
        return False
    if category in SINGLE:
        return len(value) == 1
    if category == "multi":
        return 1 <= len(value) <= 4
    if category == "sequence":
        return len(value) == 4 and set(value) == set(LABELS)
    return False

for pred, cat in zip(submission["prediction"], test["category"]):
    assert valid_pred(pred, cat), f"Invalid prediction {pred!r} for category {cat!r}"

out = WORK_ROOT / "submission.csv" if WORK_ROOT.is_dir() else Path("submission.csv")
submission.to_csv(out, index=False)
print(f"Wrote {len(submission)} rows to {out}")
print(submission.head())

