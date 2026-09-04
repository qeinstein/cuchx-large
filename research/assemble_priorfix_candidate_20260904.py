"""Assemble non-submitted candidate vectors over the preserved 0.93859 champion.

Inputs are kept separate so every change remains attributable:
  * the corrected manner-prior implementation's real-test delta;
  * the previously audited hidden-template candidate;
  * a new, conservative sequence lookup whose action-set order is identical in at least
    three training examples;
  * an optional high-margin donor-slot hypothesis for test_0449 only.
"""
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sequence_rows(tr, te, current):
    train = tr[(tr.source == "HAU") & (tr.category == "sequence")]
    stable = {}
    for key, g in train.groupby(train.apply(
            lambda r: tuple(sorted(str(r[L]).strip() for L in "ABCD")), axis=1)):
        orders = []
        for _, r in g.iterrows():
            opts = [str(r[L]).strip() for L in "ABCD"]
            orders.append(tuple(opts["ABCD".index(x)] for x in str(r.answer)))
        counts = Counter(orders)
        if len(g) >= 3 and len(counts) == 1:
            stable[key] = (orders[0], len(g))

    out = []
    test = te[(te.source == "HAU") & (te.category == "sequence")]
    for _, r in test.iterrows():
        opts = [str(r[L]).strip() for L in "ABCD"]
        hit = stable.get(tuple(sorted(opts)))
        if hit is None:
            continue
        order, n = hit
        pred = "".join("ABCD"[opts.index(x)] for x in order)
        old = str(current.loc[r.qa_id, "prediction"])
        if pred != old:
            out.append(dict(qa_id=r.qa_id, old=old, new=pred, source="sequence_set_stable",
                            reason=f"same action set has {n} identical training orders"))
    return pd.DataFrame(out, columns=["qa_id", "old", "new", "source", "reason"])


def apply(base, changes, audit, source):
    for r in changes.itertuples():
        q = r.qa_id
        old = str(base.loc[q, "prediction"])
        if old == str(r.new):
            continue
        audit.append(dict(qa_id=q, before=old, after=str(r.new), source=source,
                          reason=str(r.reason), action="applied"))
        base.loc[q, "prediction"] = str(r.new)


def main():
    from champ.core import load_all
    tr, te, _ = load_all()
    champion = pd.read_csv(os.path.join(ROOT, "submission_093859_SUBMITTED.csv"))
    oldbase = pd.read_csv(os.path.join(ROOT, "submission_092105_SUBMITTED.csv"))
    prior = pd.read_csv(os.path.join(ROOT, "research", "submission_priorfix_base.csv"))
    hidden = pd.read_csv(os.path.join(ROOT, "research", "submission_hidden_template_candidate.csv"))

    def keyed(d):
        return d.set_index("qa_id")

    ch = keyed(champion)
    ob = keyed(oldbase)
    pr = keyed(prior)
    hi = keyed(hidden)
    seq = sequence_rows(tr, te, pr)

    base = ch.copy()
    audit = []
    # The champion's existing 30 overrides are protected.  Apply only prior-fix rows that
    # were not already changed by those validated layers.
    for q in pr.index:
        if str(pr.loc[q, "prediction"]) == str(ob.loc[q, "prediction"]):
            continue
        if str(ch.loc[q, "prediction"]) != str(ob.loc[q, "prediction"]):
            continue
        audit.append(dict(qa_id=q, before=str(base.loc[q, "prediction"]),
                          after=str(pr.loc[q, "prediction"]), source="manner_prior_fix",
                          reason="per-clip group/position prior counts corrected; full OOF net +16",
                          action="applied"))
        base.loc[q, "prediction"] = str(pr.loc[q, "prediction"])

    # Hidden-template candidate is already a delta against the exact champion.  Do not
    # re-interpret its decisions here; retain its own audited structural provenance.
    for q in hi.index:
        if str(hi.loc[q, "prediction"]) == str(ch.loc[q, "prediction"]):
            continue
        audit.append(dict(qa_id=q, before=str(base.loc[q, "prediction"]),
                          after=str(hi.loc[q, "prediction"]), source="hidden_template_audit",
                          reason="clean donor cohort / previously audited sequence mechanism",
                          action="applied"))
        base.loc[q, "prediction"] = str(hi.loc[q, "prediction"])

    apply(base, seq, audit, "sequence_set_stable")
    no_slot = base.copy()
    a0 = pd.DataFrame(audit)
    out0 = os.path.join(ROOT, "research", "submission_priorfix_structural_candidate.csv")
    no_slot.reset_index().to_csv(out0, index=False)
    a0.to_csv(os.path.join(ROOT, "research", "submission_priorfix_structural_candidate_audit.csv"),
              index=False)

    # Separate optional version for the only donor-slot decision whose training alignment
    # margin clears the held-out sign-score threshold (0.25).  It is not mixed into the
    # conservative vector until reviewed against the rest of the evidence.
    with_slot = no_slot.copy()
    if "test_0449" in with_slot.index:
        before = str(with_slot.loc["test_0449", "prediction"])
        if before != "C":
            with_slot.loc["test_0449", "prediction"] = "C"
            a0 = pd.concat([a0, pd.DataFrame([dict(
                qa_id="test_0449", before=before, after="C", source="donor_slot_sign_high_margin",
                reason="user1 template block 1-1; sign-alignment margin 0.606; chosen slots 1-2",
                action="applied")])], ignore_index=True)
    out1 = os.path.join(ROOT, "research", "submission_priorfix_structural_slot187_candidate.csv")
    with_slot.reset_index().to_csv(out1, index=False)
    a0.to_csv(os.path.join(ROOT, "research",
                           "submission_priorfix_structural_slot187_candidate_audit.csv"),
              index=False)

    print("stable sequence changes:")
    print(seq.to_string(index=False) if len(seq) else "none")
    for p in (out0, out1):
        d = keyed(pd.read_csv(p))
        print(os.path.basename(p), "changes_vs_champion", int((d.prediction != ch.prediction).sum()))


if __name__ == "__main__":
    main()
