#!/usr/bin/env python3
"""PCRME Stage 1 — build aligned action-phase tuples across sibling HAU trials.

Research question this whole directory answers: if we compare corresponding ACTION PHASES
across sibling trials of one session, rather than whole clips, can we learn a representation
of HOW an action was performed that generalises across held-out users?

This script does ONLY the data construction. It does not train anything and does not touch
the champion pipeline. Output is a flat table, one row per (session, action, trial) triple
that has a directly HARn-anchored segment, plus a purity report.

Definitions
-----------
"Session" = one (user, aa, bb) scenario-repeat group; its sibling trials are cc = 1, 2, [3].
"Anchored segment" = a HARn folder clip whose (t0, t1, f0, f1) uniquely nests inside exactly
one parent HAU trial, using the SAME containment rule `champ/pipeline.py` already uses in
production (t0<=parent.t0+0.5, t1>=parent.t1-0.5, f0<=parent.f0, f1>=parent.f1). We do not
invent a new nesting rule here.

Alignment across siblings is by ACTION IDENTITY (the HARn folder name / its mapped HAU action
text). Verified globally before writing any code that an action never occurs twice within one
trial's set of anchored segments (0/2927 in this corpus), so alignment by action name within
one trial is unambiguous -- there is no "which occurrence" decision to get wrong here. This is
recorded as `purity_report.json`'s `duplicate_action_in_trial` count so future data changes are
caught automatically rather than assumed.

This build is ANCHORED-ONLY: it does not use the dense frame-level localizer to fill in actions
that have no direct HARn segment in a given trial. That is a real limitation (coverage is
capped by how many sessions/actions happen to have a HARn clip recorded for every sibling), not
an oversight -- keeping it out avoids silently mixing a ground-truth-labelled segment with a
model-inferred one under one reliability level, which the research brief explicitly asked to
keep separate. `evidence_source` is written as a column so a future extension can add a
`dense_inferred` tier without touching this tier's rows.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHAMP = ROOT / "champ"
sys.path.insert(0, str(CHAMP))

from core import load_all, gt_letters, mgroup  # noqa: E402
import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent


def main():
    tr, te, meta = load_all()
    P.caches(meta)
    nest = P._C["nest"]  # HARn qa_path -> parent HAU qa_path (train + test, same containment rule)

    mi = meta.set_index("qa_path")
    vocab = json.load(open(CHAMP / "vocab.json"))
    h2h = vocab["HARN2HAU"]  # HARn folder name -> HAU action text

    # ---- emotion label per HAU trial (training only; test has no answer)
    em = tr[tr.category == "emotion"].copy()
    em["label"] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
    em["group"] = em.label.map(mgroup)
    lab_of = dict(zip(em.path, em.label))
    grp_of = dict(zip(em.path, em.group))

    # ---- anchored segments per parent HAU trial (train only: nest built for train+test but
    # test trials have no manner label so they are excluded from THIS supervised table; a
    # separate test-side extraction is needed for inference and is done by a later script)
    hau_train = meta[meta.kind == "train_hau"].copy()
    hau_train[["aa", "bb", "cc"]] = hau_train.trial.str.split("-", expand=True)
    hau_train["cc"] = hau_train.cc.astype(int)

    seg_by_parent = {}
    dup_in_trial = 0
    for harn_path, parent_path in nest.items():
        if parent_path not in mi.index or harn_path not in mi.index:
            continue
        prow = mi.loc[parent_path]
        if prow.kind != "train_hau":
            continue
        hrow = mi.loc[harn_path]
        folder = hrow.action
        if not isinstance(folder, str) or folder not in h2h:
            continue
        action_text = h2h[folder]
        seg_by_parent.setdefault(parent_path, []).append(
            dict(action=action_text, folder=folder, f0=hrow.f0, f1=hrow.f1,
                 t0=hrow.t0, t1=hrow.t1, harn_path=harn_path)
        )
    for p, segs in seg_by_parent.items():
        acts = [s["action"] for s in segs]
        if len(acts) != len(set(acts)):
            dup_in_trial += 1

    # ---- group parent trials into sessions
    sessions = {}
    for r in hau_train.itertuples():
        key = (r.user, r.aa, r.bb)
        sessions.setdefault(key, {})[r.cc] = dict(
            qa_path=r.qa_path, unit_dir=r.unit_dir, f0=r.f0, f1=r.f1, t0=r.t0, t1=r.t1,
            nf=r.nf, label=lab_of.get(r.qa_path), group=grp_of.get(r.qa_path),
        )

    n_sessions = len(sessions)
    n_sessions_2plus = sum(1 for v in sessions.values() if len(v) >= 2)
    n_sessions_labeled = sum(
        1 for v in sessions.values()
        if sum(1 for t in v.values() if t["label"] is not None) >= 2
    )

    # ---- build aligned (session, action) tuples: for every action that has an anchored
    # segment in >=2 sibling trials of the same session, emit one row per participating trial.
    # `t` (the trial dict) and `s` (the segment dict) both carry f0/f1/t0/t1 under those exact
    # names but with different meanings (parent-session span vs segment span), so they are kept
    # as two separate dicts and read from explicitly below rather than merged.
    rows = []
    for (u, aa, bb), trials in sessions.items():
        labeled_trials = {cc: t for cc, t in trials.items() if t["label"] is not None}
        if len(labeled_trials) < 2:
            continue
        by_action = {}
        for cc, t in labeled_trials.items():
            for s in seg_by_parent.get(t["qa_path"], []):
                by_action.setdefault(s["action"], {})[cc] = (t, s)
        for action, occ in by_action.items():
            if len(occ) < 2:
                continue
            for cc, (t, s) in occ.items():
                rows.append(dict(
                    session=f"{u}|{aa}|{bb}", user=u, aa=aa, bb=bb, cc=cc, action=action,
                    harn_folder=s["folder"], manner_label=t["label"], manner_group=t["group"],
                    parent_qa_path=t["qa_path"], parent_unit_dir=t["unit_dir"],
                    parent_f0=t["f0"], parent_f1=t["f1"], parent_t0=t["t0"], parent_t1=t["t1"],
                    seg_f0=s["f0"], seg_f1=s["f1"], seg_t0=s["t0"], seg_t1=s["t1"],
                    n_sibling_trials_with_action=len(occ),
                    evidence_source="harn_anchored",
                ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "aligned_phases.csv", index=False)

    n_pairs_or_triples = df.groupby(["session", "action"]).ngroups
    n_actions = df.action.nunique()
    n_users = df.user.nunique()

    report = dict(
        n_sessions_total=n_sessions,
        n_sessions_with_2plus_trials=n_sessions_2plus,
        n_sessions_with_2plus_labeled_trials=n_sessions_labeled,
        n_aligned_session_action_tuples=n_pairs_or_triples,
        n_rows_in_aligned_table=len(df),
        n_distinct_actions_covered=n_actions,
        n_distinct_users_covered=n_users,
        duplicate_action_within_one_trial=dup_in_trial,
        rows_per_tuple_size=df.groupby(["session", "action"]).size().value_counts().to_dict(),
        manner_group_row_counts=df.manner_group.value_counts().to_dict(),
        actions_by_n_tuples=df.groupby("action")["session"].nunique().sort_values(
            ascending=False).to_dict(),
    )
    json.dump(report, open(OUT / "purity_report.json", "w"), indent=2, default=str)

    print(f"sessions total={n_sessions}  with>=2 trials={n_sessions_2plus}  "
          f"with>=2 LABELED trials={n_sessions_labeled}")
    print(f"aligned (session, action) tuples={n_pairs_or_triples}  "
          f"table rows={len(df)}  distinct actions={n_actions}  distinct users={n_users}")
    print(f"duplicate action within one trial (should be 0): {dup_in_trial}")
    print("rows per tuple size (2=pair, 3=triple):", report["rows_per_tuple_size"])
    print("manner group row counts:", report["manner_group_row_counts"])
    print(f"\nwrote {OUT / 'aligned_phases.csv'} and {OUT / 'purity_report.json'}")


if __name__ == "__main__":
    main()
