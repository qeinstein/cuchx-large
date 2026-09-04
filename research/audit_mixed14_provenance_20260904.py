"""Controlled provenance audit for the scored 14-row mixed candidate.

Run the exact current real-test inference path twice with all code/config held fixed.
The only difference is the historical indentation bug in ``fit_manner``: the buggy arm
updates the group/rank/manner counters once per variant using the last clip, while the
fixed arm uses the repository's current implementation.

This script never writes a submission.  It writes an audit CSV under ``research/``.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))

import core as C
import pipeline as P


MIXED_IDS = {
    "test_0330": "sequence_template",
    "test_0334": "sequence_template",
    "test_0342": "sequence_template",
    "test_0353": "sequence_template",
    "test_0386": "hidden_template_emotion",
    "test_0387": "hidden_template_emotion",
    "test_0426": "priorfix_claim",
    "test_0427": "priorfix_claim",
    "test_0429": "priorfix_claim",
    "test_0443": "priorfix_claim",
    "test_0447": "priorfix_claim",
    "test_0458": "priorfix_claim",
    "test_0488": "priorfix_claim",
    "test_0519": "priorfix_claim",
}


def fit_manner_buggy(tr, meta):
    """Historical implementation from the scored-candidate parent commit."""
    mfeat = {
        r.qa_path: {c: getattr(r, c) for c in C.PHYS if hasattr(r, c)}
        for r in meta.itertuples()
    }
    e = tr[tr.category == "emotion"]
    X, y = [], []
    cnt_mp = Counter(); cnt_m = Counter()
    cnt_gp = Counter(); cnt_g = Counter()
    cnt_mr = Counter(); cnt_mr_tot = Counter()
    cnt_gr = Counter(); cnt_gr_tot = Counter()
    lab_by_grp = Counter()
    import itertools as _it

    for (_u, _a, _b), g in e.groupby(["user", "aa", "bb"]):
        g = g.sort_values("cc")
        rows_all = [r for _, r in g.iterrows()]
        variants = [rows_all]
        if len(rows_all) >= 3:
            variants += [list(c) for c in _it.combinations(rows_all, 2)]
        for rows_v in variants:
            blk = [r.path for r in rows_v]
            k = len(blk)
            bf = C.block_features(blk, mfeat, k)
            for i, r in enumerate(rows_v):
                lab = str(r[C.gt_letters(r)[0]]).strip()
                gr = C.mgroup(lab)
                X.append(bf[i]); y.append(gr)
                cnt_mp[(lab, i, k)] += 1; cnt_m[(lab, k)] += 1
            # Historical indentation defect: these use only the final clip of rows_v.
            cnt_gp[(gr, i, k)] += 1; cnt_g[(gr, k)] += 1
            rb = 0 if k == 1 else int(round(2 * i / (k - 1)))
            cnt_mr[(lab, rb)] += 1; cnt_mr_tot[lab] += 1
            cnt_gr[(gr, rb)] += 1; cnt_gr_tot[gr] += 1
            lab_by_grp[(gr, lab)] += 1

    Xd = pd.DataFrame(X)
    cols = list(Xd.columns)
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        random_state=0,
    )
    clf.fit(Xd[cols].to_numpy(float), y)

    def p_pos_given_manner(lab, i, k):
        gr = C.mgroup(lab)
        rb = 0 if k <= 1 else int(round(2 * i / (k - 1)))
        p_gr = (cnt_gr.get((gr, rb), 0) + 1.0) / (cnt_gr_tot.get(gr, 0) + 3.0)
        p_mr = (cnt_mr.get((lab, rb), 0) + 3.0 * p_gr) / (cnt_mr_tot.get(lab, 0) + 3.0)
        p_gp = (cnt_gp.get((gr, i, k), 0) + 2.0 * p_mr) / (cnt_g.get((gr, k), 0) + 2.0)
        base = 0.5 * p_mr + 0.5 * p_gp
        alpha = 4.0
        nm = cnt_m.get((lab, k), 0)
        return (cnt_mp.get((lab, i, k), 0) + alpha * base) / (nm + alpha)

    gtot = Counter()
    for (gr, lab), c in lab_by_grp.items():
        gtot[gr] += c
    pmg = {
        lab: (c + 0.5) / (gtot[gr] + 0.5 * 60)
        for (gr, lab), c in lab_by_grp.items()
    }
    ntot = sum(gtot.values())
    return dict(
        clf=clf,
        cols=cols,
        pmg=pmg,
        mfeat=mfeat,
        ppm=p_pos_given_manner,
        gprior={g: (gtot[g] + 1) / (ntot + 5) for g in C.GROUPS},
        classes=list(clf.classes_),
    )


def run_test(fit_fn, tr, te, meta):
    old = P.fit_manner
    P.fit_manner = fit_fn
    try:
        ctx = P.fit_all(tr, meta, hold_users=[], split_train="oof")
        vis = P.real_test_view(te)
        pred, _blocks, _pool_of, _diag = P.solve(vis, ctx, "test")
    finally:
        P.fit_manner = old
    v8 = pd.read_csv(os.path.join(ROOT, "submission_v8.csv")).set_index("qa_id").prediction
    return pd.Series(
        {q: (pred.get(q) if pred.get(q) is not None else v8.loc[q]) for q in te.qa_id},
        name="prediction",
    )


def main():
    tr, te, meta = P.load_all()
    P.caches(meta)
    fixed = run_test(C.fit_manner, tr, te, meta)
    buggy = run_test(fit_manner_buggy, tr, te, meta)
    champ = pd.read_csv(os.path.join(ROOT, "submission_093859_SUBMITTED.csv")).set_index("qa_id").prediction.astype(str)
    mixed = pd.read_csv(os.path.join(ROOT, "submission_094736_mixed14_SUBMITTED.csv")).set_index("qa_id").prediction.astype(str)
    prior = pd.read_csv(os.path.join(ROOT, "research", "submission_priorfix_base.csv")).set_index("qa_id").prediction.astype(str)

    rows = []
    for q in te.qa_id:
        if q not in MIXED_IDS and str(fixed[q]) == str(buggy[q]):
            continue
        rows.append({
            "qa_id": q,
            "mixed_mechanism": MIXED_IDS.get(q, "non_mixed_bug_delta"),
            "champion": str(champ[q]),
            "mixed": str(mixed[q]),
            "priorfix_artifact": str(prior[q]),
            "controlled_buggy": str(buggy[q]),
            "controlled_fixed": str(fixed[q]),
            "bug_changes_decision": int(str(buggy[q]) != str(fixed[q])),
            "mixed_matches_fixed": int(str(mixed[q]) == str(fixed[q])),
            "mixed_matches_buggy": int(str(mixed[q]) == str(buggy[q])),
        })
    out = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "research", "mixed14_controlled_provenance_20260904.csv")
    out.to_csv(out_path, index=False)

    m = out[out.qa_id.isin(MIXED_IDS)]
    print("controlled fixed-vs-buggy deltas:", int((fixed.astype(str) != buggy.astype(str)).sum()))
    print("mixed 14 directly caused by bug:", int(m.bug_changes_decision.sum()), "/ 14")
    print(m.to_string(index=False))
    print("written", out_path)


if __name__ == "__main__":
    main()
