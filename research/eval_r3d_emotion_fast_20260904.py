"""Fast subject-disjoint evaluation of the opt-in R3D emotion head.

This reuses the production HAU grouping and emotion solver but intentionally omits the
unrelated HARn classifier/pool fit.  It is an evaluation harness, not a replacement
pipeline: the comparison is R3D-off versus R3D-on on exactly the same pseudo-test views.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
from core import load_all, make_pseudo, fit_group_model, infer_blocks, fit_manner, solve_emotion
from pseudotest import folds, thin_to_pairs


def run(weight):
    os.environ["CHAMP_EMO_R3D"] = str(weight)
    w_phys = float(os.environ.get("OOF_W_PHYS", "1.0"))
    w_pos = float(os.environ.get("OOF_W_POS", "1.0"))
    tr, te, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        trn = tr[~tr.user.isin(hold)]
        ev = thin_to_pairs(tr, hold, 0.38, seed=fi, policy="ends")
        vis, key, aux = make_pseudo(ev, meta, hold, seed=100 + fi)
        mm = fit_manner(trn, meta)
        gm = fit_group_model(trn)
        blocks = infer_blocks(vis, gm)
        pred, diag = solve_emotion(vis, blocks, mm, w_phys=w_phys, w_pos=w_pos,
                                   w_pair=0.0, pool_of={})
        ans = dict(zip(key.qa_id, key.answer))
        # Only emotion questions are in scope.  Keep the inferred block size for
        # separate triple/pair reporting and retain the exact row-level audit.
        bsz = {q: len(b) for b in blocks for idx in b
               for q in vis[(vis.idx == idx) & (vis.category == "emotion")].qa_id}
        for q in vis[vis.category == "emotion"].qa_id:
            rows.append(dict(qa_id=q, fold=fi, bsz=bsz.get(q), pred=pred.get(q),
                             answer=ans[q], correct=int(pred.get(q) == ans[q])))
        print(f"weight={weight} fold={fi} rows={sum(r['fold'] == fi for r in rows)}",
              flush=True)
    d = pd.DataFrame(rows)
    out = os.path.join(ROOT, "research", f"r3d_emotion_oof_w{str(weight).replace('.', 'p')}.csv")
    d.to_csv(out, index=False)
    print("weight", weight, "total", int(d.correct.sum()), "/", len(d),
          "acc", round(float(d.correct.mean()), 6), "by bsz")
    print(d.groupby("bsz").correct.agg(n="size", ok="sum", acc="mean").round(4).to_string())
    return d


if __name__ == "__main__":
    weights = [float(x) for x in sys.argv[1:]] or [0.0, 0.15, 0.3, 0.5, 1.0]
    for w in weights:
        run(w)
