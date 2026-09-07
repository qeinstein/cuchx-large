"""Build audited candidates over the immutable 0.94736 submission.

No test labels are read.  Conservative changes use exact paired-cohort emotion transfer and
exact visible object templates.  The maximal research arm additionally includes exact
visible HARn single templates, whose disagreement precision cannot be reconstructed until
the ignored production OOF cache is restored.
"""

from __future__ import annotations

import hashlib
import os

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "submission_094736_mixed14_SUBMITTED.csv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    base = pd.read_csv(BASE)
    cohort = pd.read_csv(os.path.join(ROOT, "research", "user_template_test_audit_20260904.csv"))
    exact = pd.read_csv(os.path.join(ROOT, "research", "exact_visible_template_test_20260905.csv"))

    cohort = cohort[(cohort.category == "emotion") & (cohort.changed == 1)]
    objects = exact[(exact.category == "object_interaction") & (exact.purity == 1.0) &
                    (exact.support >= 1) & (exact.changed == 1)]
    singles = exact[(exact.category == "single") & (exact.purity == 1.0) &
                    (exact.support >= 1) & (exact.changed == 1)]

    def build(name, pieces):
        changes = pd.concat(pieces, ignore_index=True)
        if changes.qa_id.duplicated().any():
            raise RuntimeError("duplicate candidate decision")
        updates = dict(zip(changes.qa_id, changes.prediction.where(
            changes.prediction.notna(), changes.get("new"))))
        # Cohort rows call the prediction column `new`.
        for _, r in cohort.iterrows():
            if r.qa_id in set(changes.qa_id):
                updates[r.qa_id] = str(r.new)
        out = base.copy()
        out["prediction"] = [updates.get(q, str(p)) for q, p in zip(out.qa_id, out.prediction)]
        path = os.path.join(ROOT, "research", name)
        out.to_csv(path, index=False)
        audit = base.merge(out, on="qa_id", suffixes=("_old", "_new"))
        audit = audit[audit.prediction_old.astype(str) != audit.prediction_new.astype(str)]
        print(name, "changes", len(audit), "sha256", sha256(path))
        print(audit.to_string(index=False))

    # Normalize the two sources to a common schema before concatenation.
    c = cohort[["qa_id", "new"]].rename(columns={"new": "prediction"})
    o = objects[["qa_id", "prediction"]]
    s = singles[["qa_id", "prediction"]]
    build("submission_rank1_conservative_20260905.csv", [c, o])
    build("submission_rank1_maximal_20260905.csv", [c, o, s])


if __name__ == "__main__":
    main()
