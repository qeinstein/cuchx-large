"""Subject-disjoint audit of exact visible question templates.

The key contains only source, category, normalized question text and the unordered option
multiset.  Answers are transferred as semantic option text and converted back to letters.
Test answers and sensor-derived paths/features are never used.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = ("single", "object_interaction", "sequence", "emotion", "multi", "combination")


def options(r):
    return [str(r[x]).strip() for x in "ABCD"]


def signature(r):
    return (str(r.source), str(r.category), str(r.question).strip(), tuple(sorted(options(r))))


def semantic_answer(r):
    oo = options(r)
    return tuple(oo[ord(x) - 65] for x in str(r.answer) if x in "ABCD")


def to_letters(r, answer):
    oo = options(r)
    if not answer or not all(x in oo for x in answer):
        return None
    return "".join("ABCD"[oo.index(x)] for x in answer)


def donor_map(rows):
    out = defaultdict(list)
    for _, r in rows.iterrows():
        out[signature(r)].append(semantic_answer(r))
    return out


def oof_audit(tr):
    rows = []
    for user in sorted(tr.user.dropna().unique(), key=lambda x: int(x[4:])):
        donors = donor_map(tr[tr.user != user])
        for _, r in tr[tr.user == user].iterrows():
            seen = donors.get(signature(r), [])
            if not seen:
                continue
            counts = Counter(seen)
            answer, votes = counts.most_common(1)[0]
            pred = to_letters(r, answer)
            truth = "".join(x for x in str(r.answer) if x in "ABCD")
            rows.append(dict(
                qa_id=r.qa_id, user=user, category=r.category, support=len(seen),
                answer_count=len(counts), purity=votes / len(seen), prediction=pred,
                truth=truth, correct=int(pred == truth), semantic="|".join(answer),
            ))
    return pd.DataFrame(rows)


def test_audit(tr, te, current):
    donors = donor_map(tr)
    cur = current.set_index("qa_id").prediction.astype(str)
    rows = []
    for _, r in te.iterrows():
        seen = donors.get(signature(r), [])
        if not seen:
            continue
        counts = Counter(seen)
        answer, votes = counts.most_common(1)[0]
        pred = to_letters(r, answer)
        rows.append(dict(
            qa_id=r.qa_id, category=r.category, support=len(seen),
            answer_count=len(counts), purity=votes / len(seen), prediction=pred,
            current=cur[r.qa_id], changed=int(pred is not None and pred != cur[r.qa_id]),
            semantic="|".join(answer),
        ))
    return pd.DataFrame(rows)


def object_fallback_audit(tr):
    """Compare templates with the global-object prior used by W2 on no-sibling rows."""
    rows = []
    packed = [(r.user, r.category, signature(r), semantic_answer(r), options(r), r.qa_id)
              for _, r in tr.iterrows()]
    for user in sorted(tr.user.dropna().unique(), key=lambda x: int(x[4:])):
        donors = defaultdict(list)
        global_objects = Counter()
        for u, category, sig, answer, _oo, _qid in packed:
            if u == user:
                continue
            donors[sig].append(answer)
            if category == "object_interaction":
                global_objects.update(answer)
        for u, category, sig, truth_tuple, oo, qid in packed:
            if u != user or category != "object_interaction":
                continue
            seen = donors.get(sig, [])
            if not seen:
                continue
            counts = Counter(seen)
            answer, votes = counts.most_common(1)[0]
            if votes != len(seen):
                continue
            fallback = oo[max(range(4), key=lambda i: global_objects[oo[i]])]
            truth = truth_tuple[0]
            rows.append(dict(
                qa_id=qid, user=user, support=len(seen), truth=truth,
                fallback=fallback, template=answer[0],
                changed=int(fallback != answer[0]),
                fallback_correct=int(fallback == truth),
                template_correct=int(answer[0] == truth),
            ))
    return pd.DataFrame(rows)


def main():
    tr = pd.read_csv(os.path.join(ROOT, "training_qa.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test_qa.csv"))
    current = pd.read_csv(os.path.join(ROOT, "submission_094736_mixed14_SUBMITTED.csv"))
    tr["user"] = tr.path.str.extract(r"(user\d+)")
    oof = oof_audit(tr)
    test = test_audit(tr, te, current)
    obj = object_fallback_audit(tr)
    oof.to_csv(os.path.join(ROOT, "research", "exact_visible_template_oof_20260905.csv"), index=False)
    test.to_csv(os.path.join(ROOT, "research", "exact_visible_template_test_20260905.csv"), index=False)
    obj.to_csv(os.path.join(ROOT, "research", "exact_visible_object_fallback_oof_20260905.csv"), index=False)

    print("OOF unanimous semantic-answer gates")
    for support in (1, 2, 3, 4, 5):
        g = oof[(oof.purity == 1.0) & (oof.support >= support)]
        print("support >=", support, "n", len(g), "correct", int(g.correct.sum()),
              "accuracy", round(float(g.correct.mean()), 6) if len(g) else None)
        print(g.groupby("category").correct.agg(["count", "sum", "mean"]).round(6).to_string())
    print("\nTest disagreements at unanimous support >= 1")
    print(test[(test.purity == 1.0) & (test.changed == 1)].to_string(index=False))
    print("\nObject template vs global-prior fallback")
    for support in (1, 2, 3):
        g = obj[obj.support >= support]
        f = g[g.changed == 1]
        wr = int(((f.fallback_correct == 0) & (f.template_correct == 1)).sum())
        rw = int(((f.fallback_correct == 1) & (f.template_correct == 0)).sum())
        print("support >=", support, "n", len(g), "fallback", int(g.fallback_correct.sum()),
              "template", int(g.template_correct.sum()), "flips", len(f),
              "W->R", wr, "R->W", rw, "precision", wr / max(1, wr + rw))


if __name__ == "__main__":
    main()
