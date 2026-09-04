#!/usr/bin/env python3
"""PCRME Stage 4 -- lightweight partial adaptation, per the kill-criterion requirement.

Rather than unsupervised PCA (Stage 3b), fit a SMALL SUPERVISED head on top of the frozen
MotionBERT joint-local features: a linear layer trained with a same-action-different-manner
contrastive-style objective is the "correct" version of this test, but a full contrastive
training loop (hard-negative mining, multiple epochs over a transformer's output space) is a
large undertaking for what must be a fast go/no-go check. This runs the cheapest legitimate
proxy: a per-fold LINEAR DISCRIMINANT projection (supervised, unlike PCA, so it is explicitly
optimising for exactly the manner-group separation PCA could not see) on the joint-local
relative features, fit ONLY on that fold's training subjects, then the same combined-with-
champion test. This is "partial adaptation" in the sense the brief asked for: a small trained
component sitting on top of the frozen backbone, optimising directly for the manner target
instead of blindly compressing variance.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
mb_idx = z["index"]; joint_emb = z["joint_pooled"]
n = joint_emb.shape[0]
joint_flat = joint_emb.reshape(n, -1)
glob = z["global_pooled"]

aligned = pd.read_csv(HERE / "aligned_phases.csv").loc[mb_idx].reset_index(drop=True)

def relativize(X):
    Xc = X.copy()
    for (sess, act), g in aligned.groupby(["session", "action"]):
        idx = g.index.to_numpy(); Xc[idx] = X[idx] - X[idx].mean(0)
    return Xc

Xc_joint = relativize(joint_flat)
Xc_glob = relativize(glob)
aligned["clip_key"] = aligned.parent_qa_path
groups_iter = list(aligned.groupby("clip_key"))
clip_keys = [k for k, _ in groups_iter]
A_rel_joint = np.stack([Xc_joint[g.index.to_numpy()].mean(0) for _, g in groups_iter])
A_rel_glob = np.stack([Xc_glob[g.index.to_numpy()].mean(0) for _, g in groups_iter])
joint_df = pd.DataFrame(A_rel_joint, index=clip_keys)
glob_df = pd.DataFrame(A_rel_glob, index=clip_keys)

common = [p for p in bf_df.path if p in joint_df.index]
bf_df2 = bf_df.set_index("path").loc[common].reset_index()
Xphys = bf_df2[bf_cols].fillna(bf_df2[bf_cols].median()).to_numpy(float)
Xjoint = joint_df.loc[common].to_numpy()
Xglob = glob_df.loc[common].to_numpy()
y = bf_df2.group.to_numpy(); users = bf_df2.user.to_numpy()
ufold = {}
for f in range(5):
    for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique(): ufold[u] = f
fold = np.array([ufold.get(u, -1) for u in users])
print(f"clips: {len(common)}")

def ev_lda(Xextra, label, n_comp=4):
    accs = []
    for f in range(5):
        trm, tem = fold != f, fold == f
        lda = LinearDiscriminantAnalysis(n_components=min(n_comp, len(set(y[trm]))-1))
        lda.fit(Xextra[trm], y[trm])
        Ztr, Zte = lda.transform(Xextra[trm]), lda.transform(Xextra[tem])
        Xtr = np.concatenate([Xphys[trm], Ztr], 1); Xte = np.concatenate([Xphys[tem], Zte], 1)
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                           l2_regularization=1.0, random_state=0)
        c.fit(Xtr, y[trm]); accs.append(c.score(Xte, y[tem]))
    print(f"  {label:56s} {np.mean(accs):.4f}  folds {[round(a,3) for a in accs]}")
    return np.mean(accs)

def ev_lda_alone(Xextra, label, n_comp=4):
    accs = []
    for f in range(5):
        trm, tem = fold != f, fold == f
        lda = LinearDiscriminantAnalysis(n_components=min(n_comp, len(set(y[trm]))-1))
        lda.fit(Xextra[trm], y[trm])
        acc = lda.score(Xextra[tem], y[tem])
        accs.append(acc)
    print(f"  {label:56s} {np.mean(accs):.4f}  folds {[round(a,3) for a in accs]}")
    return np.mean(accs)

print("=== Stage 4: supervised (LDA) adaptation head on frozen MotionBERT features ===")
print(f"  {'champion baseline (reference)':56s} 0.5713")
ev_lda_alone(Xglob, "global-relative, LDA-projected, LDA's OWN accuracy")
ev_lda_alone(Xjoint, "joint-local-relative, LDA-projected, LDA's OWN accuracy")
ev_lda(Xglob, "champion + LDA(global-relative) [COMBINED]")
ev_lda(Xjoint, "champion + LDA(joint-local-relative) [COMBINED]")
