"""Fast matched-regime audit of the corrected per-clip manner priors.

This deliberately evaluates only the emotion path: the full pipeline's HARn classifier
dominates runtime but cannot be affected by the fit_manner counter fix.  Pair thinning uses
the real-test adjacent-end policy, and all fits remain subject-disjoint.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))

from core import load_all, make_pseudo, fit_group_model, infer_blocks, fit_manner, solve_emotion
from pseudotest import folds, thin_to_pairs


def main():
    tr, _, meta = load_all()
    users = sorted(tr.user.dropna().unique())
    rows = []
    for fi, hold in enumerate(folds(users, 5)):
        trn = tr[~tr.user.isin(hold)]
        ev = thin_to_pairs(tr, hold, 0.38, seed=fi, policy="ends")
        vis, key, _ = make_pseudo(ev, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, fit_group_model(trn))
        pred, _ = solve_emotion(vis, blocks, fit_manner(trn, meta), 1.0, 1.0)
        ans = dict(zip(key.qa_id, key.answer))
        eq = vis[vis.category == "emotion"]
        for r in eq.itertuples():
            rows.append(dict(qa_id=r.qa_id, fold=fi, pred=pred.get(r.qa_id),
                             answer=ans[r.qa_id], correct=int(pred.get(r.qa_id) == ans[r.qa_id])))
        print(f"fold {fi}: {sum(x['correct'] for x in rows if x['fold'] == fi)}/{len(eq)}",
              flush=True)
    out = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "research", "manner_pair_ends_oof_20260904.csv")
    out.to_csv(out_path, index=False)
    print(f"TOTAL {int(out.correct.sum())}/{len(out)} = {out.correct.mean():.6f}")
    print(out.groupby("fold").correct.agg(n="size", ok="sum", acc="mean").to_string())


if __name__ == "__main__":
    main()
