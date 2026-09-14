import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

train_df = pd.read_csv('splits/fold_0_train.csv')
val_df = pd.read_csv('splits/fold_0_val.csv')

cache = np.load('multimodal_features_all.npz', allow_pickle=True)
all_paths = list(cache['train_paths'])
X_all = cache['X_train']
p_to_idx = {p: i for i, p in enumerate(all_paths)}

print("=== BENCHMARKING CATBOOST ON EMOTION ===")
tr_emo = train_df[train_df.category == "emotion"].copy()
val_emo = val_df[val_df.category == "emotion"].copy()

tr_emo["target"] = tr_emo.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)
val_emo["target"] = val_emo.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)

X_tr_emo = np.array([X_all[p_to_idx[p]][:120] for p in tr_emo.path])
X_val_emo = np.array([X_all[p_to_idx[p]][:120] for p in val_emo.path])

# Fit CatBoost
clf_cat = CatBoostClassifier(
    iterations=250,
    learning_rate=0.08,
    depth=5,
    random_seed=42,
    verbose=0
)
clf_cat.fit(X_tr_emo, tr_emo.target)
classes_cat = list(clf_cat.classes_)
probs_cat = clf_cat.predict_proba(X_val_emo)

corr_cat = 0
for i, (_, row) in enumerate(val_emo.iterrows()):
    scores = [probs_cat[i, classes_cat.index(str(row[l]).strip().lower())] if str(row[l]).strip().lower() in classes_cat else 0.0 for l in "ABCD"]
    if "ABCD"[int(np.argmax(scores))] == row.answer:
        corr_cat += 1
print(f"CatBoost Emotion MC Acc: {corr_cat}/{len(val_emo)} ({corr_cat/len(val_emo)*100:.2f}%)")

# Ensemble RF-k60 + Logistic Regression + LightGBM + CatBoost
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(np.nan_to_num(X_tr_emo))
X_val_s = scaler.transform(np.nan_to_num(X_val_emo))

sel = SelectKBest(f_classif, k=60)
X_tr_sel = sel.fit_transform(X_tr_s, tr_emo.target)
X_val_sel = sel.transform(X_val_s)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_tr_sel, tr_emo.target)
p_rf = rf.predict_proba(X_val_sel)
classes_rf = list(rf.classes_)

lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
lr.fit(X_tr_s, tr_emo.target)
p_lr = lr.predict_proba(X_val_s)
classes_lr = list(lr.classes_)

for w_cat in [0.0, 0.1, 0.2, 0.3, 0.4]:
    corr_ens = 0
    for i, (_, row) in enumerate(val_emo.iterrows()):
        scores = []
        for l in "ABCD":
            w = str(row[l]).strip().lower()
            s_rf = p_rf[i, classes_rf.index(w)] if w in classes_rf else 0.0
            s_lr = p_lr[i, classes_lr.index(w)] if w in classes_lr else 0.0
            s_cb = probs_cat[i, classes_cat.index(w)] if w in classes_cat else 0.0

            score = (1.0 - w_cat) * (0.60 * s_rf + 0.40 * s_lr) + w_cat * s_cb
            scores.append(score)
        if "ABCD"[int(np.argmax(scores))] == row.answer:
            corr_ens += 1
    print(f"Weight w_cat={w_cat:.2f} -> Ensemble Emotion Acc: {corr_ens}/{len(val_emo)} ({corr_ens/len(val_emo)*100:.2f}%)")
