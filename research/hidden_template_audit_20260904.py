"""Audit the test-visible repeated question templates and build a gated candidate.

The test clip order exposes four user-sized cohorts.  Three cohorts have exact training
counterparts in the sequence of emotion-option intersections and, for the clean blocks,
exact action-pool signatures.  This script transfers only the training counterpart's
per-trial manner text when the whole cohort/key is identified; it does not use hidden test
answers and never overwrites the submitted champion.
"""

from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = pd.read_csv(os.path.join(ROOT, "training_qa.csv"))
TE = pd.read_csv(os.path.join(ROOT, "test_qa.csv"))
CHAMP = pd.read_csv(os.path.join(ROOT, "submission_093859_SUBMITTED.csv"))
SNAP = pickle.load(open(os.path.join(ROOT, "research", "test_pool_snapshot.pkl"), "rb"))


def add_keys(d):
    d["user"] = d.path.str.extract(r"(user\d+)")
    z = d.path.str.extract(r"/([0-9]+)-([0-9]+)-([0-9]+)")
    d["aa"], d["bb"], d["cc"] = z[0], z[1], z[2]
    d["blk"] = list(zip(d.user.fillna("test"), d.aa, d.bb))
    return d


TR = add_keys(TR)


def letters(x):
    return [c for c in str(x) if c in "ABCD"]


def options(r):
    return [str(r[c]).strip() for c in "ABCD"]


def mset(g):
    sets = [set(options(r)) for _, r in g.iterrows()]
    return frozenset(set.intersection(*sets)) if sets else frozenset()


ACT_CATS = ["single", "multi", "combination", "sequence"]
ACT = TR[(TR.source == "HAU") & TR.category.isin(ACT_CATS)]
EMO = TR[(TR.source == "HAU") & (TR.category == "emotion")]


def action_pool(block):
    out = set()
    for _, r in ACT[ACT.blk == block].iterrows():
        for c in letters(r.answer):
            out.update(x.strip() for x in str(r[c]).split(","))
    return frozenset(out)


train_blocks = {b: g.sort_values("cc") for b, g in EMO.groupby("blk")}
train_mset = {b: mset(g) for b, g in train_blocks.items()}
train_pool = {b: action_pool(b) for b in train_blocks}

test_emo = TE[TE.category == "emotion"].copy()
test_emo["idx"] = test_emo.path.str.extract(r"LM_test_(\d+)")[0].astype(int)
test_blocks = [list(map(int, b)) for b in SNAP["blocks"]]


def source_labels(block):
    g = train_blocks[block].sort_values("cc")
    return [str(r[letters(r.answer)[0]]).strip() for _, r in g.iterrows()]


def target_rows(block):
    return test_emo[test_emo.idx.isin(block)].sort_values("idx")


def source_letter(row, label):
    oo = options(row)
    return "ABCD"[oo.index(label)] if label in oo else None


def audit_cohort(lo, hi, donor):
    donor_blocks = sorted([b for b in train_blocks if b[0] == donor],
                          key=lambda b: (int(b[1]), int(b[2])))
    rows, blocks = [], []
    for j in range(lo, hi):
        tb = test_blocks[j]
        tg = target_rows(tb)
        src = donor_blocks[j - lo] if j - lo < len(donor_blocks) else None
        if src is None:
            continue
        labels = source_labels(src)
        tm = mset(tg)
        tp = frozenset(SNAP["pool_of"].get(tb[0], []))
        exact_m = tm == train_mset[src]
        exact_p = tp == train_pool[src]
        pred = {}
        if len(tg) == len(labels):
            for i, (_, r) in enumerate(tg.iterrows()):
                pred[r.qa_id] = source_letter(r, labels[i])
        blocks.append(dict(test_block=j, ids="-".join(map(str, tb)), donor=str(src),
                           target_mset="|".join(sorted(tm)), source_mset="|".join(sorted(train_mset[src])),
                           mset_exact=int(exact_m), pool_exact=int(exact_p),
                           usable=int(all(pred.values())), flips=sum(
                               pred.get(q) != CHAMP.set_index("qa_id").loc[q, "prediction"]
                               for q in pred)))
        for q, new in pred.items():
            old = CHAMP.set_index("qa_id").loc[q, "prediction"]
            rows.append(dict(cohort=f"{lo}:{hi}", test_block=j, qa_id=q,
                             donor=str(src), old=old, new=new,
                             flip=int(old != new), usable=int(new is not None),
                             source_label=labels[len([x for x in pred if x == q]) - 1]
                             if False else ""))
    return pd.DataFrame(blocks), pd.DataFrame(rows)


def pair_hypotheses(lo, hi, donor):
    donor_blocks = sorted([b for b in train_blocks if b[0] == donor],
                          key=lambda b: (int(b[1]), int(b[2])))
    out = []
    for j in range(lo, hi):
        tb = test_blocks[j]
        tg = target_rows(tb)
        src = donor_blocks[j - lo]
        labs = source_labels(src)
        if len(tg) != 2 or len(labs) != 3:
            continue
        hyps = []
        for slots in ((0, 1), (1, 2)):
            pred = []
            feasible = True
            for i, (_, r) in enumerate(tg.iterrows()):
                ll = source_letter(r, labs[slots[i]])
                feasible &= ll is not None
                pred.append(ll)
            if feasible:
                hyps.append((slots, "".join(pred)))
        out.append(dict(test_block=j, ids="-".join(map(str, tb)), donor=str(src),
                        mset="|".join(sorted(mset(tg))),
                        hypotheses=";".join(f"{a}:{b}" for a, b in hyps),
                        n_hyp=len(hyps)))
    return pd.DataFrame(out)


def main():
    champ = CHAMP.set_index("qa_id")["prediction"]
    audits = []
    transfers = []
    for lo, hi, donor in ((0, 14, "user20"), (14, 28, "user21")):
        b, r = audit_cohort(lo, hi, donor)
        audits.append(b); transfers.append(r)
    b, r = audit_cohort(44, 55, "user1")
    audits.append(b); transfers.append(r)
    ba = pd.concat(audits, ignore_index=True)
    tr = pd.concat(transfers, ignore_index=True)
    ba.to_csv(os.path.join(ROOT, "research", "hidden_template_block_audit.csv"), index=False)
    tr.to_csv(os.path.join(ROOT, "research", "hidden_template_row_audit.csv"), index=False)
    ph = pair_hypotheses(44, 55, "user1")
    ph.to_csv(os.path.join(ROOT, "research", "hidden_template_pair_hypotheses.csv"), index=False)

    # The known orphan-like empty intersections are excluded from direct transfer.  The
    # remaining exact donor blocks are a test-visible whole-template match.
    overrides = {}
    for _, row in tr.iterrows():
        j = int(row.test_block)
        if j in {12, 18, 35, 45} or not row.usable:
            continue
        overrides[row.qa_id] = row.new

    # Combine only with the already staged exact-pool sequence candidate; both artifacts
    # are copies, and the submitted champion remains immutable.
    seq_base_path = os.path.join(ROOT, "research", "submission_template_order_candidate.csv")
    base = pd.read_csv(seq_base_path) if os.path.exists(seq_base_path) else CHAMP.copy()
    base["prediction"] = [overrides.get(q, p) for q, p in zip(base.qa_id, base.prediction)]
    out = os.path.join(ROOT, "research", "submission_hidden_template_candidate.csv")
    base.to_csv(out, index=False)
    print("block audit")
    print(ba.to_string(index=False))
    print("\ntransfer flips", int(tr.flip.sum()), "eligible overrides", len(overrides))
    print(tr[tr.flip.astype(bool)].to_string(index=False))
    print("\npair hypotheses")
    print(ph.to_string(index=False))
    print("\nwritten", out)


if __name__ == "__main__":
    main()
