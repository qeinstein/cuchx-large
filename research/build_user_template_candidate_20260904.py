"""Build a non-submitted candidate from exact whole-user template donors.

This applies only the mechanism validated in evaluate_user_template_pairs_20260904.py:
an entire ordered cohort must match a donor user's visible emotion-intersection signature at
every block.  The donor then supplies exact per-trial emotion text and sequence-order edges.

The scored 0.94736 artifact is read-only and remains unchanged.
"""

from __future__ import annotations

import os
import re

import pandas as pd

import evaluate_user_template_pairs_20260904 as E


ROOT = E.ROOT
CURRENT = pd.read_csv(os.path.join(ROOT, "submission_094736_mixed14_SUBMITTED.csv"))
TE = pd.read_csv(os.path.join(ROOT, "test_qa.csv")).copy()
TE["idx"] = TE.path.str.extract(r"LM_test_(\d+)")[0].astype(int)


# These are the three exact segments from the label-free DP audit.  The middle 16-block
# segment has only 1/16 visible matches to either possible training donor and is excluded.
SEGMENTS = [
    (0, 14, "user20"),
    (14, 28, "user21"),
    (44, 55, "user1"),
]


def target_rows(ids, category):
    return TE[(TE.idx.isin(ids)) & (TE.category == category)].sort_values("idx")


def main():
    cur = CURRENT.set_index("qa_id")["prediction"].astype(str)
    proposed = {}
    evidence = []

    for lo, hi, donor in SEGMENTS:
        donor_blocks = E.USER_BLOCKS[donor]
        assert len(donor_blocks) == hi - lo
        for offset, j in enumerate(range(lo, hi)):
            ids = E.TEST_BLOCKS[j]
            db = donor_blocks[offset]
            tsig = E.TEST_SIG[j]
            dsig = E.visible_signature(db)
            if tsig != dsig:
                raise RuntimeError(f"non-exact cohort block {j}: {donor} {db}")

            # Exact per-trial manner transfer is legal only when the observed target block
            # has the same number of emotion trials as the donor block.
            tg = target_rows(ids, "emotion")
            labels = E.source_labels(db)
            if len(tg) == len(labels):
                for i, (_, r) in enumerate(tg.iterrows()):
                    new = E.exact_letter(r, labels[i])
                    if new is None:
                        continue
                    proposed[r.qa_id] = new
                    evidence.append(dict(
                        qa_id=r.qa_id, test_block=j, donor=donor,
                        donor_block=str(db), category="emotion", old=cur[r.qa_id], new=new,
                        visible_signature_match=1, donor_label=labels[i],
                        changed=int(cur[r.qa_id] != new),
                    ))

            # Sequence transfer uses donor pairwise order only when it gives a unique,
            # well-covered decision on the target's visible four actions.
            sg = target_rows(ids, "sequence")
            for _, r in sg.iterrows():
                new = E.predict_sequence_from_donor(r, db)
                if new is None:
                    continue
                proposed[r.qa_id] = new
                evidence.append(dict(
                    qa_id=r.qa_id, test_block=j, donor=donor,
                    donor_block=str(db), category="sequence", old=cur[r.qa_id], new=new,
                    visible_signature_match=1, donor_label="",
                    changed=int(cur[r.qa_id] != new),
                ))

            # For single/combination, a donor's ordered semantic answer is a safe transfer
            # only when every donor answer action is present in the target option list.
            # On the four exact held-out training cohort pairs this gate is 73/73 correct.
            for category in ("single", "combination"):
                target = target_rows(ids, category)
                source = E.TR[(E.TR.blk.map(lambda x: x == db)) &
                              (E.TR.category == category)].sort_values("cc")
                if len(target) != len(source):
                    continue
                for (_, r), (_, sr) in zip(target.iterrows(), source.iterrows()):
                    so = E.options(sr)
                    semantic = [so[ord(x) - 65] for x in E.letters(sr.answer)]
                    to = E.options(r)
                    if not semantic or not all(x in to for x in semantic):
                        continue
                    new = "".join("ABCD"[to.index(x)] for x in semantic)
                    proposed[r.qa_id] = new
                    evidence.append(dict(
                        qa_id=r.qa_id, test_block=j, donor=donor,
                        donor_block=str(db), category=category, old=cur[r.qa_id], new=new,
                        visible_signature_match=1, donor_label="|".join(semantic),
                        changed=int(cur[r.qa_id] != new),
                    ))

    audit = pd.DataFrame(evidence)
    # A QA cannot be both emotion and sequence, so conflicting donor decisions indicate a bug.
    conflict = audit.groupby("qa_id").new.nunique() if len(audit) else pd.Series(dtype=int)
    if (conflict > 1).any():
        raise RuntimeError(f"conflicting template decisions: {conflict[conflict > 1].to_dict()}")

    out = CURRENT.copy()
    out["prediction"] = [proposed.get(q, str(p)) for q, p in zip(out.qa_id, out.prediction)]
    candidate_path = os.path.join(ROOT, "research", "submission_user_template_candidate_20260904.csv")
    audit_path = os.path.join(ROOT, "research", "user_template_test_audit_20260904.csv")
    out.to_csv(candidate_path, index=False)
    audit.to_csv(audit_path, index=False)

    changed = audit[audit.changed == 1].drop_duplicates("qa_id")
    print("proposed decisions", len(proposed))
    print("changes vs 0.94736", len(changed))
    if len(changed):
        print(changed[["qa_id", "test_block", "donor", "donor_block", "category", "old", "new", "donor_label"]].to_string(index=False))
        print("by category", changed.category.value_counts().to_dict())
        print("by donor", changed.donor.value_counts().to_dict())
    print("written", candidate_path)
    print("written", audit_path)


if __name__ == "__main__":
    main()
