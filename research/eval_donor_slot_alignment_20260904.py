"""Evaluate cross-user alignment of a two-clip session to a known 3-slot template.

The hidden test tail has the exact manner option set of user1 sessions, but only two of
the three chronological clips.  This experiment asks whether the omitted slot can be
identified from test-visible physical features alone.  Target labels are used only for
the final audit; donor labels are taken from the other training user in an exact-template
pair, which is the inference situation we would have on the real tail.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
from core import load_all, fit_manner, fit_group_model, solve_emotion, mgroup, PHYS, block_features


PAIRS = [("user6", "user16"), ("user7", "user17"),
         ("user8", "user18"), ("user9", "user19")]


def label(r):
    return str(r[str(r["answer"])[0]]).strip()


def make_vis(rows, paths):
    d = rows[rows.path.isin(paths) & (rows.category == "emotion")].copy()
    d = d.drop(columns=[c for c in ["answer", "user", "aa", "bb", "cc", "harn_action"]
                      if c in d.columns])
    order = {p: i + 1 for i, p in enumerate(paths)}
    d["idx"] = d.path.map(order)
    d["clip"] = d.idx.map(lambda i: "LM_pseudo_%04d" % i)
    d["true_path"] = d.path
    return d.sort_values("idx").reset_index(drop=True)


def raw_sign_score(a, b, c, d, mfeat):
    """Agreement of within-pair feature directions, with no answer-derived input."""
    va = np.array([mfeat.get(a, {}).get(x, np.nan) for x in PHYS], float)
    vb = np.array([mfeat.get(b, {}).get(x, np.nan) for x in PHYS], float)
    vc = np.array([mfeat.get(c, {}).get(x, np.nan) for x in PHYS], float)
    vd = np.array([mfeat.get(d, {}).get(x, np.nan) for x in PHYS], float)
    ok = np.isfinite(va) & np.isfinite(vb) & np.isfinite(vc) & np.isfinite(vd)
    if not ok.any():
        return -np.inf
    return float(np.mean(np.sign(va[ok] - vb[ok]) == np.sign(vc[ok] - vd[ok])))


def z_distance(a, b, c, d, mfeat):
    """Distance between pair-normalised physical signatures."""
    ta = block_features([a, b], mfeat, 2)
    do = block_features([c, d], mfeat, 2)
    vals = []
    for x, y in zip(ta, do):
        for k in x:
            if k.startswith("z_") or k.startswith("r_"):
                vx, vy = x.get(k, np.nan), y.get(k, np.nan)
                if np.isfinite(vx) and np.isfinite(vy):
                    vals.append((vx - vy) ** 2)
    return float(np.mean(vals)) if vals else np.inf


def choose_target_slot(target_paths, donor_paths, donor_labs, mfeat, method):
    scores = []
    for sel in itertools.combinations(range(3), 2):
        ds = [donor_paths[i] for i in sel]
        if method == "sign":
            score = raw_sign_score(target_paths[0], target_paths[1], ds[0], ds[1], mfeat)
            scores.append((score, sel))
        elif method == "zdist":
            score = -z_distance(target_paths[0], target_paths[1], ds[0], ds[1], mfeat)
            scores.append((score, sel))
        else:
            raise ValueError(method)
    return max(scores)[1], scores


def mapped_groups(target_rows, target_paths, donor_paths, donor_labs, sel):
    out = {}
    for p, s in zip(target_paths, sel):
        rr = target_rows[(target_rows.path == p) & (target_rows.category == "emotion")].iloc[0]
        out[rr.qa_id] = mgroup(donor_labs[s])
    return out


def run():
    tr, te, meta = load_all()
    all_rows = []
    for donor_u, target_u in PAIRS:
        donor = tr[(tr.source == "HAU") & (tr.user == donor_u) & (tr.category == "emotion")]
        target = tr[(tr.source == "HAU") & (tr.user == target_u) & (tr.category == "emotion")]
        mm = fit_manner(tr[tr.user != target_u], meta)
        for (a, b), dg in donor.groupby(["aa", "bb"]):
            dg = dg.sort_values("cc")
            tg = target[(target.aa == a) & (target.bb == b)].sort_values("cc")
            dp = list(dg.path)
            tp = list(tg.path)
            if len(dp) != 3 or len(tp) != 3:
                continue
            dl = [label(r) for _, r in dg.iterrows()]
            for sel_true in itertools.combinations(range(3), 2):
                visible = [tp[i] for i in sel_true]
                vis = make_vis(target, visible)
                pred, _ = solve_emotion(vis, [[1, 2]], mm, w_phys=1.0, w_pos=1.0,
                                         w_pair=0.0, pool_of={})
                ans = {r.qa_id: str(r["answer"])[0]
                       for _, r in tg[tg.path.isin(visible)].iterrows()}
                ans_groups = {q: mgroup(next(str(r[x]).strip() for x in ["A", "B", "C", "D"]
                                             if x in str(r["answer"])))
                              for _, r in tg[tg.path.isin(visible)].iterrows()
                              for q in [r.qa_id]}
                base_ok = sum(pred.get(q) == ans[q] for q in ans)
                for method in ("sign", "zdist"):
                    sel, detail = choose_target_slot(visible, dp, dl, mm["mfeat"], method)
                    mp = mapped_groups(target, visible, dp, dl, sel)
                    ok = sum(mp[q] == ans_groups[q] for q in ans_groups)
                    all_rows.append(dict(donor=donor_u, target=target_u, aa=a, bb=b,
                                         true_slots="".join(map(str, sel_true)),
                                         chosen_slots="".join(map(str, sel)), method=method,
                                         base_ok=base_ok, donor_ok=ok,
                                         exact_slots=int(sel == sel_true),
                                         score_margin=sorted([s for s, _ in detail])[-1] -
                                                      sorted([s for s, _ in detail])[-2]))
    d = pd.DataFrame(all_rows)
    out = os.path.join(ROOT, "research", "donor_slot_alignment_oof_20260904.csv")
    d.to_csv(out, index=False)
    print(d.groupby("method").agg(n=("method", "size"), base=("base_ok", "sum"),
                                   donor=("donor_ok", "sum"), exact=("exact_slots", "sum"),
                                   base_acc=("base_ok", "mean"), donor_acc=("donor_ok", "mean"),
                                   slot_acc=("exact_slots", "mean")).round(4).to_string())
    print("written", out)


if __name__ == "__main__":
    run()
