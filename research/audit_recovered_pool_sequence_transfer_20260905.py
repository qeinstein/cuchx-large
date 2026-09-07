"""Leakage-free sequence-order transfer selected by recovered session pools.

Held-out donor selection uses the production pool recovered from test-visible questions in
the pair-ends atlas, never the target's answer-derived pool.  Source orders and source pools
come only from users outside the held-out fold.  The baseline is shipped mechanism S
(`audit_run18.base`), not the experimental Y (`audit_run18.new`).
"""
from __future__ import annotations

import ast
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
import evaluate_user_template_pairs_20260904 as U  # noqa: E402


def letters(answer):
    return [x for x in str(answer) if x in "ABCD"]


def options(row):
    return [str(row[x]).strip() for x in "ABCD"]


TR = pd.read_csv(os.path.join(ROOT, "training_qa.csv"))
TR["user"] = TR.path.str.extract(r"(user\d+)")
parts = TR.path.str.extract(r"/(\d+)-(\d+)-(\d+)")
TR["aa"], TR["bb"] = parts[0], parts[1]
TR["blk"] = list(zip(TR.user, TR.aa, TR.bb))
ACT = TR[(TR.source == "HAU") & TR.category.isin(["single", "multi", "combination", "sequence"])]
SEQ = TR[(TR.source == "HAU") & (TR.category == "sequence")]
BLOCKS = {b: g for b, g in SEQ.groupby("blk")}


def true_pool(block):
    out = set()
    for _, r in ACT[ACT.blk.map(lambda x: x == block)].iterrows():
        for letter in letters(r.answer):
            out.update(x.strip() for x in str(r[letter]).split(","))
    return frozenset(out)


POOLS = {b: true_pool(b) for b in BLOCKS}
EDGES = {}
for block, group in BLOCKS.items():
    edges = Counter()
    for _, r in group.iterrows():
        oo = options(r)
        ans = letters(r.answer)
        for i in range(len(ans)):
            for j in range(i + 1, len(ans)):
                edges[(oo[ord(ans[i]) - 65], oo[ord(ans[j]) - 65])] += 1
    EDGES[block] = edges


def recovered_oof():
    atlas = pd.read_csv(os.path.join(
        ROOT, "research", "multi_pair_atlas_20260904", "block_atlas_oof_ends.csv"
    ))
    by_path = {}
    for _, r in atlas.iterrows():
        pool = frozenset(x for x in str(r.selected_pool).split("||") if x and x != "nan")
        for path in json.loads(r.block_paths):
            by_path[path] = (pool, int(r.fold), int(r.block_size), str(r.block_id))
    return by_path


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def sources_for(pool, excluded_users, threshold=0.75):
    candidates = [(b, jaccard(pool, POOLS[b])) for b in BLOCKS if b[0] not in excluded_users]
    best = max((score for _, score in candidates), default=0.0)
    if best < threshold:
        return [], best
    return [b for b, score in candidates if abs(score - best) < 1e-12], best


def closure_prediction(question_options, sources, recovered_pool):
    universe = set(recovered_pool) | set(question_options)
    score = Counter()
    for source in sources:
        edges = EDGES[source]
        for x, y in itertools.permutations(universe, 2):
            score[(x, y)] += edges.get((x, y), 0) - edges.get((y, x), 0)
    graph = defaultdict(set)
    for (x, y), value in score.items():
        if value > 0:
            graph[x].add(y)
        elif value < 0:
            graph[y].add(x)
    reach = defaultdict(set)
    for x in universe:
        todo = list(graph[x])
        while todo:
            y = todo.pop()
            if y in reach[x]:
                continue
            reach[x].add(y)
            todo.extend(graph[y])
    pair = Counter()
    for x, y in itertools.permutations(question_options, 2):
        if y in reach[x]:
            pair[(x, y)] = 1
        elif x in reach[y]:
            pair[(y, x)] = 1
    coverage = sum(bool(pair[(x, y)] or pair[(y, x)])
                   for x, y in itertools.combinations(question_options, 2))
    ranked = []
    for perm in itertools.permutations(question_options):
        value = sum(pair[(perm[i], perm[j])] for i in range(4) for j in range(i + 1, 4))
        ranked.append((value, perm))
    ranked.sort(reverse=True)
    margin = ranked[0][0] - ranked[1][0]
    prediction = "".join("ABCD"[question_options.index(x)] for x in ranked[0][1])
    return prediction, coverage, margin


def main():
    recovered = recovered_oof()
    s = pd.read_csv(os.path.join(ROOT, "seqlab", "audit_run18.csv")).set_index("qa")
    users = sorted(TR.user.dropna().unique(), key=lambda x: int(x[4:]))
    # Match the atlas/pseudotest folds exactly.
    sys.path.insert(0, os.path.join(ROOT, "champ"))
    from pseudotest import folds
    held = {fi: set(group) for fi, group in enumerate(folds(users, 5))}

    rows = []
    for _, r in SEQ.iterrows():
        if r.qa_id not in s.index or r.path not in recovered:
            continue
        pool, fold, block_size, block_id = recovered[r.path]
        sources, similarity = sources_for(pool, held[fold])
        if not sources:
            continue
        pred, coverage, margin = closure_prediction(options(r), sources, pool)
        truth = "".join(letters(r.answer))
        base = str(s.loc[r.qa_id, "base"])
        rows.append(dict(
            qa_id=r.qa_id, fold=fold, user=r.user, block_id=block_id,
            block_size=block_size, recovered_pool="||".join(sorted(pool)),
            best_similarity=similarity, n_sources=len(sources), coverage=coverage,
            margin=margin, base=base, new=pred, truth=truth,
            changed=int(base != pred), base_correct=int(base == truth),
            new_correct=int(pred == truth), sources=";".join(map(str, sources)),
        ))
    oof = pd.DataFrame(rows)
    oof["transition"] = "same"
    oof.loc[(oof.changed == 1) & (oof.base_correct == 0) & (oof.new_correct == 1), "transition"] = "W->R"
    oof.loc[(oof.changed == 1) & (oof.base_correct == 1) & (oof.new_correct == 0), "transition"] = "R->W"
    oof.loc[(oof.changed == 1) & (oof.base_correct == 0) & (oof.new_correct == 0), "transition"] = "W->W"
    op = os.path.join(ROOT, "research", "recovered_pool_sequence_transfer_oof_20260905.csv")
    oof.to_csv(op, index=False)

    te = pd.read_csv(os.path.join(ROOT, "test_qa.csv"))
    champion = pd.read_csv(os.path.join(ROOT, "submission_095614_327of342_CHAMPION.csv")).set_index("qa_id")
    test_rows = []
    block_for = {int(i): tuple(map(int, block)) for block in U.TEST_BLOCKS for i in block}
    for _, r in te[te.category == "sequence"].iterrows():
        idx = int(re.search(r"LM_test_(\d+)", str(r.path)).group(1))
        block = block_for.get(idx)
        if block is None:
            continue
        pool = frozenset(U.SNAP["pool_of"].get(block[0], []))
        sources, similarity = sources_for(pool, set())
        if not sources:
            continue
        pred, coverage, margin = closure_prediction(options(r), sources, pool)
        base = str(champion.loc[r.qa_id, "prediction"])
        test_rows.append(dict(
            qa_id=r.qa_id, idx=idx, block=str(block), recovered_pool="||".join(sorted(pool)),
            best_similarity=similarity, n_sources=len(sources), coverage=coverage,
            margin=margin, champion=base, new=pred, changed=int(base != pred),
            sources=";".join(map(str, sources)),
        ))
    test = pd.DataFrame(test_rows)
    tp = os.path.join(ROOT, "research", "recovered_pool_sequence_transfer_test_20260905.csv")
    test.to_csv(tp, index=False)

    for cov in (4, 5, 6):
        g = oof[oof.coverage >= cov]
        changed = g[g.changed == 1]
        print("coverage >=", cov, "rows", len(g), "base", int(g.base_correct.sum()),
              "new", int(g.new_correct.sum()), "changed", len(changed),
              changed.transition.value_counts().to_dict())
        print("fold net", (g.groupby("fold").new_correct.sum() -
                           g.groupby("fold").base_correct.sum()).to_dict())
    print("\nTEST coverage=6 changes vs frozen champion")
    print(test[(test.coverage == 6) & (test.changed == 1)].to_string(index=False))
    print("wrote", op, tp)


if __name__ == "__main__":
    main()
