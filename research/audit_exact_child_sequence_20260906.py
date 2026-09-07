"""Use exact-template HARn child identities as high-precision HAU order constraints."""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
sys.path.insert(0, os.path.join(ROOT, "research"))
import harn as H  # noqa: E402
import evaluate_user_template_pairs_20260904 as U  # noqa: E402


def opts(row):
    return [str(row[x]).strip() for x in "ABCD"]


def letters(answer):
    return [x for x in str(answer) if x in "ABCD"]


def meta_key(path):
    if str(path).startswith(("HAU/", "HARn/")):
        return str(path)
    match = re.search(r"(LM_test_\d+)", str(path))
    return match.group(1) if match else str(path)


def build_nest(meta, train):
    pairs = (("train_hau", "train_harn") if train else ("test", "test"))
    hau = []
    for r in meta[meta.kind == pairs[0]].itertuples():
        if not np.isfinite(r.t0):
            continue
        if not train and int(r.qa_path.split("_")[-1]) < 65:
            continue
        hau.append((r.qa_path, r.t0, r.t1, r.f0, r.f1))
    nest = {}
    for r in meta[meta.kind == pairs[1]].itertuples():
        if not np.isfinite(r.t0):
            continue
        if not train and int(r.qa_path.split("_")[-1]) >= 65:
            continue
        candidates = [h for h in hau if h[1] <= r.t0 + 0.5 and h[2] >= r.t1 - 0.5
                      and h[3] <= r.f0 and h[4] >= r.f1]
        if len(candidates) == 1:
            nest[r.qa_path] = candidates[0][0]
    return nest


def predict_from_points(question_options, points):
    counts = Counter()
    for i, (on_i, act_i) in enumerate(points):
        for j, (on_j, act_j) in enumerate(points):
            if i != j and act_i != act_j and on_i < on_j:
                counts[(act_i, act_j)] += 1
    graph = defaultdict(set)
    for x, y in itertools.permutations(set(question_options), 2):
        a, b = counts[(x, y)], counts[(y, x)]
        if a > b:
            graph[x].add(y)
        elif b > a:
            graph[y].add(x)
    reach = defaultdict(set)
    for x in question_options:
        todo = list(graph[x])
        while todo:
            y = todo.pop()
            if y in reach[x]:
                continue
            reach[x].add(y)
            todo.extend(graph[y])
    coverage = sum((y in reach[x]) or (x in reach[y])
                   for x, y in itertools.combinations(question_options, 2))
    ranked = []
    for perm in itertools.permutations(question_options):
        score = sum(y in reach[x] for i, x in enumerate(perm) for y in perm[i + 1:])
        ranked.append((score, perm))
    ranked.sort(reverse=True)
    margin = ranked[0][0] - ranked[1][0]
    pred = "".join("ABCD"[question_options.index(x)] for x in ranked[0][1])
    return pred, coverage, margin, counts


def main():
    tr = pd.read_csv(os.path.join(ROOT, "training_qa.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test_qa.csv"))
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    mi = meta.set_index("qa_path")
    oof_templates = pd.read_csv(os.path.join(
        ROOT, "research", "harn_template_sensor_gate_oof_20260905.csv"
    )).set_index("qa_id")
    test_templates = pd.read_csv(os.path.join(
        ROOT, "research", "harn_template_sensor_gate_test_20260905.csv"
    )).set_index("qa_id")
    s = pd.read_csv(os.path.join(ROOT, "seqlab", "audit_run18.csv")).set_index("qa")

    nest_train = build_nest(meta, True)
    obs_train = defaultdict(list)
    hs = tr[(tr.source == "HARn") & (tr.category == "single")]
    for _, r in hs.iterrows():
        if r.qa_id not in oof_templates.index:
            continue
        child = r.path
        parent = nest_train.get(child)
        if parent is None:
            continue
        letter = str(oof_templates.loc[r.qa_id, "template"])
        phrase = str(r[letter]).strip()
        action_id = H.S2A.get(phrase)
        action = H.H2H.get(action_id)
        if action is None:
            continue
        p, c = mi.loc[parent], mi.loc[child]
        span = p.f1 - p.f0
        if np.isfinite(span) and span > 0:
            block = tuple(parent.split("/")[1:2] + parent.split("/")[-1].split("-")[:2])
            obs_train[block].append((float((c.f0 - p.f0) / span), action))

    rows = []
    seq = tr[(tr.source == "HAU") & (tr.category == "sequence")]
    for _, r in seq.iterrows():
        if r.qa_id not in s.index:
            continue
        bits = r.path.split("/")
        block = (bits[1], *bits[-1].split("-")[:2])
        points = obs_train.get(block, [])
        if len(points) < 2:
            continue
        pred, coverage, margin, _ = predict_from_points(opts(r), points)
        truth = "".join(letters(r.answer))
        base = str(s.loc[r.qa_id, "base"])
        rows.append(dict(
            qa_id=r.qa_id, fold=int(s.loc[r.qa_id, "fold"]), block=str(block),
            n_children=len(points), coverage=coverage, margin=margin,
            base=base, new=pred, truth=truth, changed=int(base != pred),
            base_correct=int(base == truth), new_correct=int(pred == truth),
        ))
    oof = pd.DataFrame(rows)
    oof["transition"] = "same"
    oof.loc[(oof.changed == 1) & (oof.base_correct == 0) & (oof.new_correct == 1), "transition"] = "W->R"
    oof.loc[(oof.changed == 1) & (oof.base_correct == 1) & (oof.new_correct == 0), "transition"] = "R->W"
    oof.loc[(oof.changed == 1) & (oof.base_correct == 0) & (oof.new_correct == 0), "transition"] = "W->W"
    op = os.path.join(ROOT, "research", "exact_child_sequence_oof_20260906.csv")
    oof.to_csv(op, index=False)

    nest_test = build_nest(meta, False)
    obs_test = defaultdict(list)
    hs_test = te[(te.source == "HARn") & (te.category == "single")]
    for _, r in hs_test.iterrows():
        if r.qa_id not in test_templates.index:
            continue
        child = meta_key(r.path)
        parent = nest_test.get(child)
        if parent is None:
            continue
        letter = str(test_templates.loc[r.qa_id, "template"])
        phrase = str(r[letter]).strip()
        action_id = H.S2A.get(phrase)
        action = H.H2H.get(action_id)
        if action is None:
            continue
        p, c = mi.loc[parent], mi.loc[child]
        span = p.f1 - p.f0
        if np.isfinite(span) and span > 0:
            obs_test[parent].append((float((c.f0 - p.f0) / span), action))

    champion = pd.read_csv(os.path.join(
        ROOT, "submission_095614_327of342_CHAMPION.csv"
    )).set_index("qa_id")
    test_block_for = {int(idx): tuple(map(int, block))
                      for block in U.TEST_BLOCKS for idx in block}
    test_rows = []
    seq_test = te[te.category == "sequence"]
    for _, r in seq_test.iterrows():
        idx = int(re.search(r"LM_test_(\d+)", str(r.path)).group(1))
        block = test_block_for[idx]
        points = [pt for parent_idx in block
                  for pt in obs_test.get(f"LM_test_{parent_idx:04d}", [])]
        if len(points) < 2:
            continue
        pred, coverage, margin, _ = predict_from_points(opts(r), points)
        base = str(champion.loc[r.qa_id, "prediction"])
        test_rows.append(dict(
            qa_id=r.qa_id, idx=idx, block=str(block), n_children=len(points),
            coverage=coverage, margin=margin, champion=base, new=pred,
            changed=int(base != pred),
        ))
    test = pd.DataFrame(test_rows)
    tp = os.path.join(ROOT, "research", "exact_child_sequence_test_20260906.csv")
    test.to_csv(tp, index=False)

    for cov in (3, 4, 5, 6):
        g = oof[oof.coverage >= cov]
        changed = g[g.changed == 1]
        print("coverage >=", cov, "rows", len(g), "base", int(g.base_correct.sum()),
              "new", int(g.new_correct.sum()), "changes", len(changed),
              changed.transition.value_counts().to_dict())
        print(" fold net", (g.groupby("fold").new_correct.sum() -
                             g.groupby("fold").base_correct.sum()).to_dict())
    print("\nTEST coverage=6 changes")
    if len(test):
        print(test[(test.coverage == 6) & (test.changed == 1)].to_string(index=False))
    print("observations", sum(map(len, obs_train.values())), sum(map(len, obs_test.values())))
    print("wrote", op, tp)


if __name__ == "__main__":
    main()
