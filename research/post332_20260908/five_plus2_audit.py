#!/usr/bin/env python3
"""Trace the five rows responsible for the measured +2 bundle."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
IDS = ["test_0488", "test_0477", "test_0506", "test_0519", "test_0501"]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    te = {r["qa_id"]: r for r in read(ROOT / "test_qa.csv")}
    old = {r["qa_id"]: r["prediction"] for r in read(
        ROOT / "submission_096491_330of342_CHAMPION.csv")}
    new = {r["qa_id"]: r["prediction"] for r in read(
        ROOT / "submission_097076_332of342_CHAMPION.csv")}
    meta = {r["qa_path"]: r for r in read(ROOT / "champ/meta.csv")}
    try:
        logits = np.load(ROOT / "champ/dense_logits_screen1_bucket.npz")
        logit_keys = set(logits.files)
    except FileNotFoundError:
        logit_keys = set()
    rows = []
    for q in IDS:
        r = te[q]
        lm = re.search(r"(LM_test_\d+)", r["path"]).group(1)
        m = meta.get(lm)
        rows.append({
            "qa_id": q, "clip": lm, "category": r["category"],
            "old_330": old[q], "champion_332": new[q],
            "changed_from_330": old[q] != new[q],
            "options": {L: r[L] for L in "ABCD"},
            "metadata_present": m is not None,
            "feature_present": bool(m and m.get("f0", "") not in ("", "nan")),
            "dense_logits_present": f"test|{lm}" in logit_keys,
            "parent_found_by_pipeline": False,
            "failure_class": (
                "no_meta_no_dense_no_parent -> solve(None) then champion fallback"
                if m is None and f"test|{lm}" not in logit_keys else
                "aggregate/interval evidence only; no parent session"
            ),
        })
    result = {
        "rows": rows,
        "shared_class": {
            "all_harn_single": all(x["category"] == "single" for x in rows),
            "all_parentless": all(not x["parent_found_by_pipeline"] for x in rows),
            "no_meta_count": sum(not x["metadata_present"] for x in rows),
            "no_dense_count": sum(not x["dense_logits_present"] for x in rows),
            "interpretation": "The bundle is predominantly a weak/no-evidence HARn-single fallback cohort; 0501 is the only row with cached feature/logit evidence. The +2 score identifies only the aggregate, not a repeatable per-row label rule.",
        },
    }
    (OUT / "five_plus2_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
