#!/usr/bin/env python3
"""Stress-test the exact object-template override without touching submissions.

The incumbent object fallback is approximated by the leave-user-out global object
prior used by the original audit.  For every exact option signature we compute the
leave-user-out modal semantic answer, then sweep support/purity/margin gates.  The
test-side table is only a proposal screen; it is never interpreted as a label.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def user_of(path):
    m = re.search(r"(user\d+)", path)
    return m.group(1) if m else ""


def signature(r):
    return (r["source"], r["category"], r["question"].strip(),
            tuple(sorted(r[L].strip() for L in "ABCD")))


def semantic(r):
    if "answer" not in r:
        return None
    a = str(r["answer"])
    return r[a[0]].strip() if a and a[0] in "ABCD" else ""


def modal(counter, options):
    # Stable tie break follows option order, never lexical answer text.
    return max(options, key=lambda x: (counter[x], -options.index(x)))


def one_row(r, donors, all_other):
    answers = Counter(semantic(x) for x in donors)
    top = max(answers.values())
    second = sorted(answers.values(), reverse=True)[1] if len(answers) > 1 else 0
    options = [r[L].strip() for L in "ABCD"]
    fallback = modal(Counter(semantic(x) for x in all_other), options)
    template = modal(answers, options)
    truth = semantic(r)
    return {
        "qa_id": r["qa_id"],
        "user": r["user"],
        "support": len(donors),
        "purity": top / len(donors),
        "vote_margin": top - second,
        "answer_count": len(answers),
        "template": template,
        "fallback": fallback,
        "truth": truth,
        "changed": int(template != fallback),
        "template_correct": None if truth is None else int(template == truth),
        "fallback_correct": None if truth is None else int(fallback == truth),
    }


def score(rows):
    changed = [r for r in rows if r["changed"]]
    wr = sum(r["template_correct"] and not r["fallback_correct"] for r in changed)
    rw = sum(r["fallback_correct"] and not r["template_correct"] for r in changed)
    return {
        "flips": len(changed),
        "w_to_r": wr,
        "r_to_w": rw,
        "precision": wr / max(1, wr + rw),
        "net": wr - rw,
    }


def main():
    tr = [r for r in read_csv(ROOT / "training_qa.csv")
          if r["category"] == "object_interaction"]
    for r in tr:
        r["user"] = user_of(r["path"])
    by_sig = defaultdict(list)
    for r in tr:
        by_sig[signature(r)].append(r)

    oof = []
    for r in tr:
        donors = [x for x in by_sig[signature(r)] if x["user"] != r["user"]]
        if donors:
            oof.append(one_row(r, donors, [x for x in tr if x["user"] != r["user"]]))

    test = read_csv(ROOT / "test_qa.csv")
    test = [r for r in test if r["category"] == "object_interaction"]
    test_by_id = {r["qa_id"]: r for r in test}
    current = {r["qa_id"]: r["prediction"] for r in read_csv(
        ROOT / "submission_097076_332of342_CHAMPION.csv")}
    test_rows = []
    for r in test:
        r["user"] = ""
        donors = by_sig[signature(r)]
        if donors:
            x = one_row(r, donors, tr)
            x["current"] = current[r["qa_id"]]
            test_rows.append(x)

    # Sweep conservative gates.  A gate is promotable only if it has zero R->W on
    # all changed OOF rows and does not merely rediscover the unanimous rule.
    sweep = []
    for support in range(1, 9):
        for purity in (0.5, 0.6, 2/3, 0.75, 0.8, 5/6, 0.9, 1.0):
            for margin in (0, 1, 2):
                g = [r for r in oof if r["support"] >= support and
                     r["purity"] >= purity and r["vote_margin"] >= margin]
                z = score(g)
                z.update({"support": support, "purity": purity, "vote_margin": margin,
                          "rows": len(g), "unanimous_equivalent": purity == 1.0})
                sweep.append(z)

    # Report all test disagreements for potentially useful (but not automatically
    # approved) gates.  The OOF score is the evidence; test output is a screen only.
    gates = [x for x in sweep if x["flips"] and x["r_to_w"] == 0 and
             not x["unanimous_equivalent"]]
    proposals = []
    for g in gates:
        rows = [r for r in test_rows if r["support"] >= g["support"] and
                r["purity"] >= g["purity"] and r["vote_margin"] >= g["vote_margin"] and
                r["template"] != r["fallback"]]
        if rows:
            proposals.append({"gate": {k: g[k] for k in ("support", "purity", "vote_margin")},
                              "oof": {k: g[k] for k in ("flips", "w_to_r", "r_to_w", "precision", "net")},
                              "test": [{k: r[k] for k in ("qa_id", "support", "purity",
                                  "vote_margin", "template", "fallback", "current")}
                                       for r in rows]})

    # A crucial guard against a misleading aggregate precision: if a test
    # signature is intrinsically ambiguous, the test-side donor majority can
    # look like a useful flip simply because the OOF rows did not disagree with
    # the global fallback.  Evaluate each test signature's own leave-user-out
    # transfer, including unchanged predictions.
    signature_groups = []
    for r in test_rows:
        g = by_sig[signature(test_by_id[r["qa_id"]])]
        if not g:
            continue
        held = []
        for x in g:
            donors = [z for z in g if z["user"] != x["user"]]
            if donors:
                c = Counter(semantic(z) for z in donors)
                pred = modal(c, [x[L].strip() for L in "ABCD"])
                held.append({"qa_id": x["qa_id"], "user": x["user"],
                             "truth": semantic(x), "prediction": pred,
                             "correct": int(pred == semantic(x)),
                             "support": len(donors)})
        signature_groups.append({
            "test_qa_id": r["qa_id"],
            "full_donor_support": r["support"],
            "full_donor_counts": dict(Counter(semantic(z) for z in g)),
            "test_template": r["template"], "test_fallback": r["fallback"],
            "test_current": r["current"],
            "oof_leave_user_out": held,
            "oof_accuracy": (sum(x["correct"] for x in held) / len(held)
                              if held else None),
        })

    result = {
        "oof_rows": len(oof),
        "oof_unanimous": score([r for r in oof if r["purity"] == 1.0]),
        "test_rows_with_exact_donors": len(test_rows),
        "test_rows": test_rows,
        "test_signature_group_audit": signature_groups,
        "best_nonunanimous_zero_rw_gates": proposals,
        "conclusion": (
            "No non-unanimous gate should be promoted unless its proposed test rows also "
            "have independent clip evidence; non-unanimous 0526/0527 remain ambiguous."
        ),
    }
    (OUT / "object_gate_relaxation_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("oof_rows", "oof_unanimous",
                                              "test_rows_with_exact_donors",
                                              "best_nonunanimous_zero_rw_gates",
                                              "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
