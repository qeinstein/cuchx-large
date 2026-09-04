#!/usr/bin/env python3
"""PCRME secondary objective -- search action x manner-group cells for a high-precision
specialist, decision-level against the champion's own OOF manner PREDICTIONS.

The full manner problem may be hard while certain actions make manner (or a specific
sibling-orientation decision) obvious. This script measures, per action, whether the
action-conditioned phase-relative representation from Stage 1/2 can correct the champion's
frozen OOF prediction (`champ/audit_champ.csv`) at high precision, restricted to sessions
whose pool contains that action -- i.e. "when the session contains action X, manner can be
corrected at P precision" exactly as the brief's example describes.

This works at the SESSION level (not the isolated phase level): for each session, if it
contains action X in >=2 sibling trials, use the phase-relative evidence for X to re-rank the
plausible manner assignments and compare the resulting pick against (a) the truth and (b) the
champion's OOF prediction for that clip. This is therefore both a representation probe and a
decision-level audit in one pass, restricted action-by-action.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CHAMP = ROOT / "champ"
sys.path.insert(0, str(CHAMP))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import load_all, gt_letters, mgroup  # noqa: E402
from probe_falsification import build_phase_features  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    tr, te, meta = load_all()
    aligned = pd.read_csv(HERE / "aligned_phases.csv")
    feat_by_idx = build_phase_features(aligned)
    raw = aligned.loc[list(feat_by_idx.keys())].copy()
    Xb = np.stack([feat_by_idx[i] for i in raw.index])
    raw = raw.reset_index(drop=True)

    # champion's frozen OOF prediction + margin, per emotion clip
    audit = pd.read_csv(CHAMP / "audit_champ.csv")
    audit = audit[audit.category == "emotion"].set_index("qa_id")
    em = tr[tr.category == "emotion"].copy()
    em["truth_label"] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
    em["truth_group"] = em.truth_label.map(mgroup)
    path2qa = dict(zip(em.path, em.qa_id))
    raw["qa_id"] = raw.parent_qa_path.map(path2qa)
    raw = raw[raw.qa_id.isin(audit.index)]
    raw["champ_pred_letter"] = raw.qa_id.map(audit["pred"])
    raw["champ_ok"] = raw.qa_id.map(audit["correct"])
    raw["fold"] = raw.qa_id.map(audit["fold"])

    # session-relative feature (per phase row), aligned to `raw`'s current row order
    Xc = Xb.copy()
    for (sess, act), g in raw.reset_index().groupby(["session", "action"]):
        idx = g["index"].to_numpy()
        mu = Xb[idx].mean(0)
        Xc[idx] = Xb[idx] - mu

    ufold = {}
    for f in range(5):
        for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique():
            ufold[u] = f

    results = []
    n_actions_total = raw.action.nunique()
    for _ai, (action, g) in enumerate(raw.groupby("action")):
        print(f"  [{_ai+1}/{n_actions_total}] {action} (n_sessions={g.session.nunique()})",
              flush=True)
        if g.session.nunique() < 8:
            continue
        y = g.manner_group.to_numpy()
        fold = np.array([ufold.get(u, -1) for u in g.user])
        idx = g.index.to_numpy()
        Xg = Xc[idx]
        oof_pred = np.empty(len(g), dtype=object)
        for f in range(5):
            trm, tem = fold != f, fold == f
            if trm.sum() < 10 or tem.sum() == 0 or len(set(y[trm])) < 2:
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, max_depth=3, l2_regularization=1.0,
                random_state=0,
            ).fit(Xg[trm], y[trm])
            oof_pred[tem] = clf.predict(Xg[tem])
        g = g.assign(spec_group_pred=oof_pred)
        g["spec_ok_group"] = (g.spec_group_pred == g.manner_group)
        valid = g.spec_group_pred.notna()
        n_valid = int(valid.sum())
        if n_valid < 10:
            continue
        # decision-level: does the specialist's GROUP call disagree with the champion's own
        # predicted group, and if so, who is right? (group of champion's letter prediction)
        # map champion predicted LETTER -> its manner text -> group, from em row options
        opt_lookup = em.set_index("qa_id")
        rows = []
        for r in g[valid].itertuples():
            qid = r.qa_id
            if qid not in opt_lookup.index:
                continue
            orow = opt_lookup.loc[qid]
            champ_letter = r.champ_pred_letter
            if not isinstance(champ_letter, str) or champ_letter not in "ABCD":
                continue
            champ_text = str(orow[champ_letter]).strip()
            champ_group = mgroup(champ_text)
            disagree = champ_group != r.spec_group_pred
            rows.append(dict(
                action=action, qa_id=qid, fold=r.fold, truth_group=r.manner_group,
                champ_group=champ_group, spec_group=r.spec_group_pred,
                champ_ok=(champ_group == r.manner_group),
                spec_ok=(r.spec_group_pred == r.manner_group), disagree=disagree,
            ))
        D = pd.DataFrame(rows)
        if not len(D):
            continue
        dis = D[D.disagree]
        wr = int(((~dis.champ_ok) & dis.spec_ok).sum())
        rw = int((dis.champ_ok & (~dis.spec_ok)).sum())
        results.append(dict(
            action=action, n_sessions=g.session.nunique(), n_valid=n_valid,
            spec_group_acc=g[valid].spec_ok_group.mean(),
            n_disagreements=len(dis), w_to_r=wr, r_to_w=rw,
            flip_precision=wr / max(wr + rw, 1),
            net=wr - rw, n_folds_with_disagreement=dis.fold.nunique(),
        ))

    R = pd.DataFrame(results).sort_values("flip_precision", ascending=False)
    R.to_csv(HERE / "action_specialist_results.csv", index=False)
    print(R.to_string(index=False))
    print(f"\nwrote {HERE / 'action_specialist_results.csv'}")

    strong = R[(R.flip_precision >= 0.80) & (R.n_disagreements >= 8) & (R.n_folds_with_disagreement >= 3)]
    print(f"\nactions clearing >=0.80 precision, >=8 disagreements, >=3 folds: {len(strong)}")
    if len(strong):
        print(strong.to_string(index=False))


if __name__ == "__main__":
    main()
