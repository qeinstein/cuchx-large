import os, sys, itertools, numpy as np, pandas as pd
from collections import Counter

ROOT = "/home/fluxx/Workspace/cuchx-large"
sys.path.insert(0, os.path.join(ROOT, "champ"))

from core import load_all, fit_manner, mgroup, GROUPS, opts, gt_letters, block_features, solve_emotion
from pseudotest import folds

tr, te, meta = load_all()
hau = tr[(tr.source == "HAU") & (tr.category == "emotion")].copy()
hau["user"] = hau.path.str.extract(r"(user\d+)")
hau["trial"] = hau.path.map(lambda p: int(p.split("/")[-1].split("-")[-1]) if len(p.split("/")[-1].split("-")) == 3 else -1)
hau["sess"] = hau.path.map(lambda p: "-".join(p.split("/")[-1].split("-")[:2]) if len(p.split("/")[-1].split("-")) == 3 else "")

users = sorted(hau.user.dropna().unique(), key=lambda s: int(s[4:]))
print(f"Total users: {len(users)}")

meta_indexed = meta.set_index("qa_path")

results = []
for fi, hold in enumerate(folds(users, 5)):
    train_users = [u for u in users if u not in hold]
    trn = tr[tr.user.isin(train_users)]
    mm = fit_manner(trn, meta)
    
    for u in hold:
        u_hau = hau[hau.user == u]
        for sess, g in u_hau.groupby("sess"):
            if len(g) != 3 or set(g.trial) != {1, 2, 3}:
                continue
            g_sorted = g.sort_values("trial")
            paths = list(g_sorted.path)
            true_manners = [g_sorted.iloc[i][g_sorted.iloc[i].answer] for i in range(3)]
            true_letters = [g_sorted.iloc[i].answer for i in range(3)]
            qids = list(g_sorted.qa_id)
            
            opts_list = [set(opts(g_sorted.iloc[i])) for i in range(3)]
            inter = set.intersection(*opts_list)
            if len(inter) < 3:
                continue
                
            regimes = {
                "(0, 2)": (0, 2, 1),
                "(0, 1)": (0, 1, 2),
                "(1, 2)": (1, 2, 0)
            }
            
            for reg_name, (i0, i1, i_drop) in regimes.items():
                blk = [paths[i0], paths[i1]]
                mini_vis = pd.DataFrame([
                    {"idx": 0, "qa_id": f"{qids[i0]}_eval", "category": "emotion", "true_path": paths[i0],
                     "A": g_sorted.iloc[i0].A, "B": g_sorted.iloc[i0].B, "C": g_sorted.iloc[i0].C, "D": g_sorted.iloc[i0].D},
                    {"idx": 1, "qa_id": f"{qids[i1]}_eval", "category": "emotion", "true_path": paths[i1],
                     "A": g_sorted.iloc[i1].A, "B": g_sorted.iloc[i1].B, "C": g_sorted.iloc[i1].C, "D": g_sorted.iloc[i1].D}
                ])
                pred_dict, _ = solve_emotion(mini_vis, [[0, 1]], mm, 1.0, 1.0)
                champ0 = pred_dict.get(f"{qids[i0]}_eval")
                champ1 = pred_dict.get(f"{qids[i1]}_eval")
                
                truth0 = true_letters[i0]
                truth1 = true_letters[i1]
                
                t0_0 = meta_indexed.loc[paths[i0], "t0"] if paths[i0] in meta_indexed.index else np.nan
                t1_0 = meta_indexed.loc[paths[i0], "t1"] if paths[i0] in meta_indexed.index else np.nan
                t0_1 = meta_indexed.loc[paths[i1], "t0"] if paths[i1] in meta_indexed.index else np.nan
                t1_1 = meta_indexed.loc[paths[i1], "t1"] if paths[i1] in meta_indexed.index else np.nan
                dur0 = (t1_0 - t0_0) if np.isfinite(t0_0) and np.isfinite(t1_0) else np.nan
                dur1 = (t1_1 - t0_1) if np.isfinite(t0_1) and np.isfinite(t1_1) else np.nan
                
                results.append({
                    "fold": fi, "user": u, "sess": sess, "regime": reg_name,
                    "champ0": champ0, "truth0": truth0, "champ_ok0": int(champ0 == truth0),
                    "champ1": champ1, "truth1": truth1, "champ_ok1": int(champ1 == truth1),
                    "dur0": dur0, "dur1": dur1, "dur_ratio": dur0 / max(1e-6, dur1) if np.isfinite(dur0) and np.isfinite(dur1) else np.nan,
                    "manners": [true_manners[i0], true_manners[i1]],
                    "groups": [mgroup(true_manners[i0]), mgroup(true_manners[i1])]
                })

df = pd.DataFrame(results)
out_csv = os.path.join(ROOT, "research", "withholding_oof_results_20260907.csv")
df.to_csv(out_csv, index=False)
print(f"Saved OOF results to {out_csv}")

print("\n=======================================================")
print("CHAMPION ACCURACY BY WITHHOLDING REGIME:")
for reg, g in df.groupby("regime"):
    n_q = len(g) * 2
    ok = g.champ_ok0.sum() + g.champ_ok1.sum()
    acc = ok / n_q
    print(f"Regime {reg:8s}: {ok}/{n_q} = {acc:.4f} ({acc*100:.2f}%)")

print("\n=======================================================")
print("REGIME (0, 2) ANALYSIS (TRIAL 2 WITHHELD):")
df_02 = df[df.regime == "(0, 2)"].copy()
print(f"Total sessions in (0, 2): {len(df_02)}")
print(f"Clip 0 accuracy: {df_02.champ_ok0.sum()}/{len(df_02)} = {df_02.champ_ok0.mean():.4f}")
print(f"Clip 1 accuracy: {df_02.champ_ok1.sum()}/{len(df_02)} = {df_02.champ_ok1.mean():.4f}")
print(f"Both correct: {((df_02.champ_ok0 == 1) & (df_02.champ_ok1 == 1)).sum()}/{len(df_02)} = {((df_02.champ_ok0 == 1) & (df_02.champ_ok1 == 1)).mean():.4f}")
print(f"Both wrong: {((df_02.champ_ok0 == 0) & (df_02.champ_ok1 == 0)).sum()}/{len(df_02)} = {((df_02.champ_ok0 == 0) & (df_02.champ_ok1 == 0)).mean():.4f}")

err_count = (df_02.champ_ok0 == 0).sum() + (df_02.champ_ok1 == 0).sum()
print(f"Total champion errors under regime (0, 2): {err_count}/{len(df_02)*2} = {err_count/(len(df_02)*2):.4f}")

print("\nPer-fold champion accuracy on regime (0, 2):")
for fi, g in df_02.groupby("fold"):
    ok = g.champ_ok0.sum() + g.champ_ok1.sum()
    print(f"  Fold {fi}: {ok}/{len(g)*2} = {ok/(len(g)*2):.4f}")
