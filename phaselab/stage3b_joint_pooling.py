#!/usr/bin/env python3
"""PCRME Stage 3b -- does per-joint (body-part-local) pooling recover signal that global
mean-pooling of the MotionBERT representation destroys?

`motionbert_embeddings.npz`'s `joint_pooled` array is (n_phase_rows, 17, 512): time-averaged
but NOT joint-averaged, i.e. exactly the "body/joint-local embeddings" the brief asked to try
before collapsing to a single global vector. This concatenates all 17 joints (8704-dim),
PCA-reduces WITHIN EACH FOLD's training subjects only (no leakage), then reruns the identical
session-relative + combined-with-champion tests.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CHAMP = ROOT / "champ"
sys.path.insert(0, str(CHAMP)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import load_all, gt_letters, mgroup, PHYS, block_features

HERE = Path(__file__).resolve().parent
tr, te, meta = load_all()
mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}
em = tr[tr.category == "emotion"].copy()
em["label"] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
em["group"] = em.label.map(mgroup)
em["blk"] = list(zip(em.user, em.aa, em.bb))
bf_rows = {}
for b, g in em.groupby("blk"):
    g = g.sort_values("cc"); paths = list(g.path)
    for path, d in zip(paths, block_features(paths, mfeat, len(paths))):
        bf_rows[path] = d
bf_df = pd.DataFrame.from_dict(bf_rows, orient="index")
bf_cols = [c for c in bf_df.columns if c not in ("pos", "k", "posfrac")]
bf_df = bf_df.reindex(em.path).reset_index(drop=True)
bf_df[["user", "group", "path"]] = em[["user", "group", "path"]].to_numpy()

z = np.load(HERE / "motionbert_embeddings.npz")
mb_idx = z["index"]; joint_emb = z["joint_pooled"]           # (n, 17, 512)
n = joint_emb.shape[0]
joint_flat = joint_emb.reshape(n, -1)                        # (n, 8704)

aligned = pd.read_csv(HERE / "aligned_phases.csv").loc[mb_idx].reset_index(drop=True)
Xc = joint_flat.copy()
for (sess, act), g in aligned.groupby(["session", "action"]):
    idx = g.index.to_numpy(); Xc[idx] = joint_flat[idx] - joint_flat[idx].mean(0)
aligned["clip_key"] = aligned.parent_qa_path
clip_keys = list(aligned.groupby("clip_key").groups.keys())
A_rel = np.stack([Xc[g.index.to_numpy()].mean(0) for _, g in aligned.groupby("clip_key")])
mb_rel_df = pd.DataFrame(A_rel, index=clip_keys)

common = [p for p in bf_df.path if p in mb_rel_df.index]
bf_df2 = bf_df.set_index("path").loc[common].reset_index()
Xphys = bf_df2[bf_cols].fillna(bf_df2[bf_cols].median()).to_numpy(float)
Xjoint = mb_rel_df.loc[common].to_numpy()
y = bf_df2.group.to_numpy(); users = bf_df2.user.to_numpy()
ufold = {}
for f in range(5):
    for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique(): ufold[u] = f
fold = np.array([ufold.get(u, -1) for u in users])
print(f"clips: {len(common)}  joint-flat dim: {Xjoint.shape[1]}")

def ev_pca(Xextra, ncomp, label, combine_with_phys=True):
    accs = []
    for f in range(5):
        trm, tem = fold != f, fold == f
        pca = PCA(n_components=ncomp, random_state=0).fit(Xextra[trm])
        Ztr, Zte = pca.transform(Xextra[trm]), pca.transform(Xextra[tem])
        if combine_with_phys:
            Xtr = np.concatenate([Xphys[trm], Ztr], 1); Xte = np.concatenate([Xphys[tem], Zte], 1)
        else:
            Xtr, Xte = Ztr, Zte
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                           l2_regularization=1.0, random_state=0)
        c.fit(Xtr, y[trm]); accs.append(c.score(Xte, y[tem]))
    print(f"  {label:56s} {np.mean(accs):.4f}  folds {[round(a,3) for a in accs]}")
    return np.mean(accs)

def ev_phys():
    accs = []
    for f in range(5):
        trm, tem = fold != f, fold == f
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                           l2_regularization=1.0, random_state=0)
        c.fit(Xphys[trm], y[trm]); accs.append(c.score(Xphys[tem], y[tem]))
    print(f"  {'champion baseline (reference)':56s} {np.mean(accs):.4f}  folds {[round(a,3) for a in accs]}")

print("=== per-joint (body-part-local) pooling: PCA-reduced, fold-internal fit ===")
ev_phys()
for nc in (20, 50, 100):
    ev_pca(Xjoint, nc, f"joint-local-relative alone, PCA={nc}", combine_with_phys=False)
for nc in (20, 50, 100):
    ev_pca(Xjoint, nc, f"champion + joint-local-relative [COMBINED], PCA={nc}")
