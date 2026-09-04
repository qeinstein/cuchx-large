#!/usr/bin/env python3
"""PCRME Stage 3 -- does a frozen, pretrained skeleton-motion transformer (MotionBERT,
NTU60-xsub action-finetuned) beat the handcrafted phase-relative representation from Stage 2,
and does it change the result against the champion's fair `block_features()` baseline?

Same protocol as `stage2b_fair_comparison.py` (same classifier, same folds, mandatory
shuffle-label control), applied to MotionBERT embeddings instead of hand-engineered stats, so
the two are directly comparable.

Arms
----
A. absolute pretrained embedding, mean-pooled per clip over its aligned phases (no session
   conditioning) -- the "does the raw representation already separate manner" control.
B. action-conditioned / session-relative residual: emb(trial, action) - mean_siblings(same
   session, same action). For k=2 sibling blocks this is exactly half the pairwise difference
   vector, so it subsumes the brief's "pairwise sibling difference" arm C rather than
   duplicating it under a different name.
D. champion's actual block_features() + B [[the decisive joint-integration test]].

Every arm also gets the mandatory shuffle-label control.
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

HERE = Path(__file__).resolve().parent


def main():
    tr, te, meta = load_all()
    mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}
    em = tr[tr.category == "emotion"].copy()
    em["label"] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
    em["group"] = em.label.map(mgroup)
    em["blk"] = list(zip(em.user, em.aa, em.bb))

    bf_rows = {}
    for b, g in em.groupby("blk"):
        g = g.sort_values("cc")
        paths = list(g.path)
        bf = block_features(paths, mfeat, len(paths))
        for path, d in zip(paths, bf):
            bf_rows[path] = d
    bf_df = pd.DataFrame.from_dict(bf_rows, orient="index")
    bf_cols = [c for c in bf_df.columns if c not in ("pos", "k", "posfrac")]
    bf_df = bf_df.reindex(em.path).reset_index(drop=True)
    bf_df[["user", "group", "path", "blk", "k"]] = em[["user", "group", "path", "blk"]].assign(
        k=em.blk.map(em.groupby("blk").size())).to_numpy()

    z = np.load(HERE / "motionbert_embeddings.npz")
    mb_idx = z["index"]
    mb_glob = z["global_pooled"]
    print(f"MotionBERT embeddings: {mb_glob.shape} for {len(mb_idx)} aligned-phase rows")

    aligned = pd.read_csv(HERE / "aligned_phases.csv").loc[mb_idx].copy()
    aligned = aligned.reset_index(drop=True)
    Xb = mb_glob  # (n_phase_rows, 512), aligned by position with `aligned`

    Xc = Xb.copy()
    for (sess, act), g in aligned.groupby(["session", "action"]):
        idx = g.index.to_numpy()
        Xc[idx] = Xb[idx] - Xb[idx].mean(0)

    aligned["clip_key"] = aligned.parent_qa_path
    clip_keys = list(aligned.groupby("clip_key").groups.keys())
    A_abs = np.stack([Xb[g.index.to_numpy()].mean(0) for _, g in aligned.groupby("clip_key")])
    A_rel = np.stack([Xc[g.index.to_numpy()].mean(0) for _, g in aligned.groupby("clip_key")])
    mb_abs_df = pd.DataFrame(A_abs, index=clip_keys)
    mb_rel_df = pd.DataFrame(A_rel, index=clip_keys)

    common = [p for p in bf_df.path if p in mb_rel_df.index]
    bf_df = bf_df.set_index("path").loc[common].reset_index()
    Xphys = bf_df[bf_cols].fillna(bf_df[bf_cols].median()).to_numpy(float)
    Xmb_abs = mb_abs_df.loc[common].to_numpy()
    Xmb_rel = mb_rel_df.loc[common].to_numpy()
    y = bf_df.group.to_numpy()
    users = bf_df.user.to_numpy()
    kblk = bf_df.k.to_numpy()

    ufold = {}
    for f in range(5):
        for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique():
            ufold[u] = f
    fold = np.array([ufold.get(u, -1) for u in users])
    print(f"clips with MotionBERT phase coverage: {len(common)} of 809")
    print(f"triple-block clips: {(kblk==3).sum()}   pair-block clips: {(kblk==2).sum()}")

    def ev(X, y, fold, label, shuffle_seed=None, mask=None):
        Xu, yu, fu = (X[mask], y[mask], fold[mask]) if mask is not None else (X, y, fold)
        rng = np.random.default_rng(shuffle_seed or 0)
        accs = []
        for f in range(5):
            trm, tem = fu != f, fu == f
            if trm.sum() == 0 or tem.sum() == 0:
                continue
            ytr = yu[trm].copy()
            if shuffle_seed is not None:
                rng.shuffle(ytr)
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                               l2_regularization=1.0, random_state=0)
            c.fit(Xu[trm], ytr)
            accs.append(c.score(Xu[tem], yu[tem]))
        print(f"  {label:56s} {np.mean(accs):.4f}  folds {[round(a,3) for a in accs]}")
        return np.mean(accs)

    print("\n=== Stage 3: MotionBERT vs fair champion baseline vs handcrafted phase-relative ===")
    a_champ = ev(Xphys, y, fold, "A0. champion block_features() [fair baseline]")
    a_abs = ev(Xmb_abs, y, fold, "A. MotionBERT absolute (no conditioning)")
    a_rel = ev(Xmb_rel, y, fold, "B. MotionBERT action-conditioned/session-relative")
    a_comb = ev(np.concatenate([Xphys, Xmb_rel], 1), y, fold,
                "D. champion block_features() + MotionBERT-relative [COMBINED]")

    stage2b = None
    p2b = HERE / "stage2b_summary.json"
    if p2b.exists():
        import json
        stage2b = json.load(open(p2b))
        print(f"\n[reference] Stage 2 handcrafted: fair baseline {stage2b['fair_baseline_acc']:.4f}"
              f"  combined {stage2b['combined_acc']:.4f}")

    print("\n=== mandatory shuffle-label controls ===")
    ev(Xmb_rel, y, fold, "B, SHUFFLED seed=1", shuffle_seed=1)
    ev(Xmb_rel, y, fold, "B, SHUFFLED seed=2", shuffle_seed=2)
    ev(np.concatenate([Xphys, Xmb_rel], 1), y, fold, "D, SHUFFLED seed=1", shuffle_seed=1)

    print("\n=== triple vs pair regime ===")
    for kk in (3, 2):
        m = kblk == kk
        if m.sum() < 20:
            continue
        print(f"  k={kk} (n={m.sum()}):")
        ev(Xphys, y, fold, f"    champion baseline", mask=m)
        ev(Xmb_rel, y, fold, f"    MotionBERT-relative", mask=m)
        ev(np.concatenate([Xphys, Xmb_rel], 1), y, fold, f"    combined", mask=m)

    import json
    json.dump(dict(champ_baseline=a_champ, mb_absolute=a_abs, mb_relative=a_rel,
                   combined=a_comb, n_clips=len(common),
                   handcrafted_reference=stage2b),
              open(HERE / "stage3_summary.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'stage3_summary.json'}")


if __name__ == "__main__":
    main()
