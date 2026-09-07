"""Decision-level audit of action-category transfer across exact user-template pairs."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import evaluate_user_template_pairs_20260904 as E


ROOT = E.ROOT
TR = E.TR
POOL_LEDGER = os.path.join(ROOT, "research", "pool_full_subblocks_paired_oof_20260905.csv")
if os.path.exists(POOL_LEDGER):
    OOF = (pd.read_csv(POOL_LEDGER)[["qa_id", "pred_base"]]
           .rename(columns={"pred_base": "pred"}).drop_duplicates("qa_id")
           .set_index("qa_id"))
else:
    OOF = E.OOF
PAIRS = {
    "user6": "user16", "user16": "user6",
    "user7": "user17", "user17": "user7",
    "user8": "user18", "user18": "user8",
    "user9": "user19", "user19": "user9",
}


def correct_texts(r):
    return [str(r[c]).strip() for c in E.letters(r.answer)]


def split_actions(text):
    return frozenset(x.strip() for x in str(text).split(",") if x.strip())


def target_pred_from_donor(target_row, donor_row):
    cat = target_row.category
    oo = E.options(target_row)
    if cat == "single":
        src = correct_texts(donor_row)
        if len(src) != 1 or src[0] not in oo:
            return None
        return "ABCD"[oo.index(src[0])]
    if cat == "multi":
        src = set(correct_texts(donor_row))
        if not src or not src.issubset(set(oo)):
            return None
        return "".join("ABCD"[i] for i, x in enumerate(oo) if x in src)
    if cat == "combination":
        src = correct_texts(donor_row)
        if len(src) != 1:
            return None
        want = split_actions(src[0])
        hits = [i for i, x in enumerate(oo) if split_actions(x) == want]
        if len(hits) != 1:
            return None
        return "ABCD"[hits[0]]
    return None


def block_rows(user, block, category):
    a, b = block[1], block[2]
    return TR[(TR.user == user) & (TR.aa == a) & (TR.bb == b) &
              (TR.category == category)].sort_values("cc", key=lambda s: s.astype(int))


def summarize(d):
    if d.empty:
        return {"n": 0, "flips": 0, "wr": 0, "rw": 0, "net": 0, "precision": 0.0}
    bc = d.base.astype(str) == d.truth.astype(str)
    nc = d.new.astype(str) == d.truth.astype(str)
    flip = d.base.astype(str) != d.new.astype(str)
    wr = int((flip & ~bc & nc).sum())
    rw = int((flip & bc & ~nc).sum())
    return {"n": len(d), "flips": int(flip.sum()), "wr": wr, "rw": rw,
            "net": wr-rw, "precision": wr/max(1, wr+rw),
            "new_acc": float(nc.mean()), "base_acc": float(bc.mean())}


def main():
    rows = []
    for target, donor in PAIRS.items():
        tb = E.USER_BLOCKS[target]
        db = E.USER_BLOCKS[donor]
        assert len(tb) == len(db)
        assert all(E.visible_signature(a) == E.visible_signature(b) for a, b in zip(tb, db))
        for bi, (tblock, dblock) in enumerate(zip(tb, db)):
            for cat in ["single", "multi", "combination"]:
                tg = block_rows(target, tblock, cat)
                dg = block_rows(donor, dblock, cat)
                # The generator emits at most one row per category/trial.  Align by c.
                donor_by_c = {str(r.cc): r for _, r in dg.iterrows()}
                for _, r in tg.iterrows():
                    dr = donor_by_c.get(str(r.cc))
                    if dr is None:
                        continue
                    new = target_pred_from_donor(r, dr)
                    if new is None or r.qa_id not in OOF.index:
                        continue
                    rows.append(dict(
                        target=target, donor=donor, block_index=bi, qa_id=r.qa_id,
                        category=cat, base=str(OOF.loc[r.qa_id, "pred"]), new=new,
                        truth="".join(E.letters(r.answer)),
                    ))
    d = pd.DataFrame(rows)
    if d.empty:
        d = pd.DataFrame(columns=["target", "donor", "block_index", "qa_id",
                                  "category", "base", "new", "truth"])
    d.to_csv(os.path.join(ROOT, "research", "user_template_action_decisions.csv"), index=False)
    print("all", summarize(d))
    for cat in ["single", "multi", "combination"]:
        print(cat, summarize(d[d.category == cat]))
    for target in sorted(PAIRS, key=lambda u: int(u[4:])):
        print(target, summarize(d[d.target == target]))


if __name__ == "__main__":
    main()
