"""Independent frame classifier for selective sequence corrections.

Exact HARn intervals label HAU frames.  A class-balanced ExtraTrees model is trained with
whole held-out users and decoded by probability centroids.  The decision ledger compares it
to shipped mechanism S and reports confidence gates without touching test labels.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
import dense as D  # noqa: E402
from pseudotest import folds  # noqa: E402


VOC = json.load(open(os.path.join(ROOT, "champ", "vocab.json")))["HARN2HAU"]
OPT2CLS = {VOC[action]: D.A2I[action] for action in D.ACTIONS}


def opts(row):
    return [str(row[x]).strip() for x in "ABCD"]


def path_key(path):
    if str(path).startswith("HAU/"):
        return str(path)
    match = re.search(r"(LM_test_\d+)", str(path))
    return match.group(1) if match else str(path)


def make_item(row, skeleton, segments):
    key = row.unit_dir + "|K"
    if key not in skeleton:
        return None
    k = skeleton[key]
    frames = skeleton[row.unit_dir + "|F"]
    x = np.concatenate([D.frame_feats(k), D.imu_feats(row.unit_dir, len(k))], 1)
    y = np.full(len(frames), D.BG, np.int16)
    for action, f0, f1 in segments.get((row.user, row.trial), []):
        if action in D.A2I and np.isfinite(f0):
            y[(frames >= f0) & (frames <= f1)] = D.A2I[action]
    return x, y


def decode(prob, classes, option_texts):
    t = np.arange(len(prob), dtype=float)
    centroids, peaks, means = [], [], []
    for option in option_texts:
        cls = OPT2CLS.get(option)
        if cls is None or cls not in classes:
            p = np.zeros(len(prob))
        else:
            p = prob[:, classes.index(cls)]
        if len(p) > 7:
            p = np.convolve(p, np.ones(7) / 7.0, mode="same")
        centroids.append(float((p * t).sum() / (p.sum() + 1e-12) / max(1, len(p) - 1)))
        peaks.append(float(p.max()) if len(p) else 0.0)
        means.append(float(p.mean()) if len(p) else 0.0)
    order = np.argsort(centroids)
    sorted_centroids = np.sort(centroids)
    separation = float(np.min(np.diff(sorted_centroids))) if len(sorted_centroids) > 1 else 0.0
    return ("".join("ABCD"[int(i)] for i in order), separation,
            float(np.mean(peaks)), float(np.min(peaks)), centroids, means)


def main():
    tr = pd.read_csv(os.path.join(ROOT, "training_qa.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test_qa.csv"))
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    skeleton = np.load(os.path.join(ROOT, "champ", "skel_seq.npz"))
    segments = defaultdict(list)
    for r in meta[meta.kind == "train_harn"].itertuples():
        segments[(r.user, r.trial)].append((r.action, r.f0, r.f1))

    sequence = tr[(tr.source == "HAU") & (tr.category == "sequence")]
    sequence_paths = set(sequence.path)
    rng = np.random.default_rng(20260906)
    sample_x, sample_y, sample_user = [], [], []
    seq_x = {}
    for i, r in enumerate(meta[meta.kind == "train_hau"].itertuples()):
        item = make_item(r, skeleton, segments)
        if item is None:
            continue
        x, y = item
        # Fixed stratified sample: equal cap for every action/background class in a clip.
        chosen = []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            chosen.extend(rng.choice(idx, size=min(16, len(idx)), replace=False).tolist())
        sample_x.append(x[chosen])
        sample_y.append(y[chosen])
        sample_user.extend([r.user] * len(chosen))
        if r.qa_path in sequence_paths:
            seq_x[r.qa_path] = x
        if (i + 1) % 200 == 0:
            print("features", i + 1, flush=True)
    x_train = np.concatenate(sample_x)
    y_train = np.concatenate(sample_y)
    u_train = np.asarray(sample_user)
    print("sampled frames", x_train.shape, "sequence clips", len(seq_x), flush=True)

    s = pd.read_csv(os.path.join(ROOT, "seqlab", "audit_run18.csv")).set_index("qa")
    users = sorted(meta.loc[meta.kind == "train_hau", "user"].dropna().unique())
    rows = []
    from sklearn.ensemble import ExtraTreesClassifier
    for fi, hold in enumerate(folds(users, 5)):
        mask = ~np.isin(u_train, hold)
        clf = ExtraTreesClassifier(
            n_estimators=96, max_features="sqrt", min_samples_leaf=2,
            max_leaf_nodes=512, class_weight="balanced", n_jobs=1,
            random_state=100 + fi,
        )
        clf.fit(x_train[mask], y_train[mask])
        target = sequence[sequence.path.isin(seq_x) & sequence.path.str.extract(
            r"(user\d+)", expand=False).isin(hold)]
        for path, group in target.groupby("path"):
            prob = clf.predict_proba(seq_x[path])
            classes = list(clf.classes_)
            for _, r in group.iterrows():
                if r.qa_id not in s.index:
                    continue
                pred, sep, peak_mean, peak_min, centroids, means = decode(prob, classes, opts(r))
                truth = "".join(x for x in str(r.answer) if x in "ABCD")
                base = str(s.loc[r.qa_id, "base"])
                rows.append(dict(
                    qa_id=r.qa_id, fold=fi, user=path.split("/")[1], path=path,
                    truth=truth, base=base, forest=pred, changed=int(base != pred),
                    base_correct=int(base == truth), forest_correct=int(pred == truth),
                    separation=sep, peak_mean=peak_mean, peak_min=peak_min,
                    centroids=json.dumps(centroids), means=json.dumps(means),
                ))
        print("fold", fi, "done", flush=True)
    out = pd.DataFrame(rows)
    out["transition"] = "same"
    out.loc[(out.changed == 1) & (out.base_correct == 0) & (out.forest_correct == 1), "transition"] = "W->R"
    out.loc[(out.changed == 1) & (out.base_correct == 1) & (out.forest_correct == 0), "transition"] = "R->W"
    out.loc[(out.changed == 1) & (out.base_correct == 0) & (out.forest_correct == 0), "transition"] = "W->W"
    op = os.path.join(ROOT, "research", "frame_forest_sequence_oof_20260906.csv")
    out.to_csv(op, index=False)
    print("raw", len(out), "base", int(out.base_correct.sum()),
          "forest", int(out.forest_correct.sum()),
          out[out.changed == 1].transition.value_counts().to_dict())
    for threshold in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        g = out[(out.changed == 1) & (out.separation >= threshold)]
        print("sep >=", threshold, "n", len(g), g.transition.value_counts().to_dict(),
              "fold_net", {int(f): int(((x.transition == "W->R").sum() -
                                         (x.transition == "R->W").sum()))
                           for f, x in g.groupby("fold")})
    print("wrote", op)


if __name__ == "__main__":
    main()
