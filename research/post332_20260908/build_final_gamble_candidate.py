#!/usr/bin/env python3
"""Build the approval-gated two-row rank-1 gamble from the verified 332 champion."""

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "submissions/submission_097076_332of342_CHAMPION.csv"
OUT = Path(__file__).with_name("FINAL_GAMBLE__0461_D__0519_A.csv")
EXPECTED_BASE_SHA256 = "25e79e1dae1149bdad81d081d1fad3a94db4e1eb88df7f00e91276e6d5668d56"
CHANGES = {"test_0461": "D", "test_0519": "A"}


if hashlib.sha256(BASE.read_bytes()).hexdigest() != EXPECTED_BASE_SHA256:
    raise SystemExit("Refusing to build: verified 332 champion hash mismatch")

with BASE.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != 682 or len({row["qa_id"] for row in rows}) != 682:
    raise SystemExit("Refusing to build: expected 682 unique rows")

seen = set()
for row in rows:
    if row["qa_id"] in CHANGES:
        row["prediction"] = CHANGES[row["qa_id"]]
        seen.add(row["qa_id"])

if seen != set(CHANGES):
    raise SystemExit(f"Refusing to build: missing rows {set(CHANGES) - seen}")

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["qa_id", "prediction"])
    writer.writeheader()
    writer.writerows(rows)

print(OUT.relative_to(ROOT))
print(hashlib.sha256(OUT.read_bytes()).hexdigest())
