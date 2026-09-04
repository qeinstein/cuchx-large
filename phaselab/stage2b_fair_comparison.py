#!/usr/bin/env python3
"""PCRME Stage 2b -- corrected, fair comparison against the champion's ACTUAL manner features.

`stage2_summary.json`'s arm A used raw ABSOLUTE PHYS values as the "champion" baseline. That
is not what the champion (or the historical 0.603 group classifier) actually uses: production
`champ/core.py:block_features()` already emits, per clip, the physical features in THREE forms
-- absolute (`a_*`), within-session z-score (`z_*`), and within-session rank (`r_*`) -- exactly
the kind of session-relative normalisation this whole PCRME hypothesis is about, just computed
over the WHOLE clip's PHYS vector rather than over individual action-phase segments. Comparing
a new session-relative PHASE feature against a session-UNAWARE whole-clip baseline is not a
fair test of "does phase-level conditioning add anything a whole-clip solver doesn't already
have" -- it mostly re-discovers that session-relative features beat session-blind ones, which
is already known and already shipped.

This script redoes the comparison with `block_features()`'s real output as the baseline, plus
the mandatory shuffle-label control on every arm (a jump this large earlier warranted treating
the first version as a suspected bug, per the brief's explicit leakage-first prior).
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

from core import load_all, gt_letters, mgroup, PHYS, block_features  # noqa: E402
from probe_falsification import build_phase_features  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    tr, te, meta = load_all()
    mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}
    em = tr[tr.category == "emotion"].copy()
    em["label"] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
    em["group"] = em.label.map(mgroup)
    em["blk"] = list(zip(em.user, em.aa, em.bb))

    # ---- the ACTUAL champion feature set: block_features() per session, exactly as
    # champ/core.py and champ/emopair.py compute it in production.
    bf_rows = {}
    for b, g in em.groupby("blk"):
        g = g.sort_values("cc")
        paths = list(g.path)
        k = len(paths)
        bf = block_features(paths, mfeat, k)
        for path, d in zip(paths, bf):
            bf_rows[path] = d
    bf_df = pd.DataFrame.from_dict(bf_rows, orient="index")
    bf_cols = [c for c in bf_df.columns if c not in ("pos", "k", "posfrac")]
    bf_df = bf_df.reindex(em.path).reset_index(drop=True)
    bf_df[["user", "group", "path"]] = em[["user", "group", "path"]].to_numpy()
    Xphys_full = bf_df[bf_cols].fillna(bf_df[bf_cols].median()).to_numpy(float)
    print(f"champion block_features(): {len(bf_cols)} columns (absolute+z+rank), "
          f"{len(bf_df)} clips")

    # ---- phase-relative aggregate, same construction as Stage 2
    aligned = pd.read_csv(HERE / "aligned_phases.csv")
    feat_by_idx = build_phase_features(aligned)
    raw = aligned.loc[list(feat_by_idx.keys())].copy()
    Xb = np.stack([feat_by_idx[i] for i in raw.index])
    raw = raw.reset_index(drop=True)
    Xc = Xb.copy()
    for (sess, act), g in raw.groupby(["session", "action"]):
        idx = g.index.to_numpy()
        Xc[idx] = Xb[idx] - Xb[idx].mean(0)
    raw["clip_key"] = raw.parent_qa_path
    clip_keys = list(raw.groupby("clip_key").groups.keys())
    Arel = np.stack([Xc[g.index.to_numpy()].mean(0) for _, g in raw.groupby("clip_key")])
    rel_df = pd.DataFrame(Arel, index=clip_keys)

    common = [p for p in bf_df.path if p in rel_df.index]
    bf_df = bf_df.set_index("path").loc[common].reset_index()
    Xphys = bf_df[bf_cols].fillna(bf_df[bf_cols].median()).to_numpy(float)
    Arel_c = rel_df.loc[common].to_numpy()
    y = bf_df.group.to_numpy()
    users = bf_df.user.to_numpy()

    ufold = {}
    for f in range(5):
        for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique():
            ufold[u] = f
    fold = np.array([ufold.get(u, -1) for u in users])
    print(f"clips with both representations: {len(common)} of 809")

    def ev(X, y, fold, label, shuffle_seed=None):
        rng = np.random.default_rng(shuffle_seed or 0)
        accs = []
        for f in range(5):
            trm, tem = fold != f, fold == f
            ytr = y[trm].copy()
            if shuffle_seed is not None:
                rng.shuffle(ytr)
            c = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
                random_state=0,
            ).fit(X[trm], ytr)
            accs.append(c.score(X[tem], y[tem]))
        print(f"  {label:56s} {np.mean(accs):.4f}  folds {[round(a,3) for a in accs]}")
        return np.mean(accs)

    print("\n=== FAIR comparison: champion's actual block_features() vs + phase-relative ===")
    a = ev(Xphys, y, fold, "champion block_features() [FAIR baseline]")
    comb = ev(np.concatenate([Xphys, Arel_c], 1), y, fold,
              "champion block_features() + phase-relative [COMBINED]")
    r = ev(Arel_c, y, fold, "phase-relative alone")

    print("\n=== mandatory shuffle-label controls (must collapse toward chance ~0.20-0.30) ===")
    ev(np.concatenate([Xphys, Arel_c], 1), y, fold, "combined, SHUFFLED seed=1", shuffle_seed=1)
    ev(np.concatenate([Xphys, Arel_c], 1), y, fold, "combined, SHUFFLED seed=2", shuffle_seed=2)
    ev(Arel_c, y, fold, "phase-relative alone, SHUFFLED seed=1", shuffle_seed=1)

    print(f"\nDelta (combined - fair baseline): {comb - a:+.4f}")
    print("Verdict written to stage2b_summary.json")

    import json
    json.dump(dict(fair_baseline_acc=a, combined_acc=comb, phase_relative_alone_acc=r,
                   delta=comb - a, n_clips=len(common)),
              open(HERE / "stage2b_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
