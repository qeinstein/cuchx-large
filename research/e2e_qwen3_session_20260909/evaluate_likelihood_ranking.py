"""Offline residual-RANKING evaluation for fold likelihood screens.

Inputs: fold likelihood records (.jsonl) + champ/oof_e2e_baseline_20260909.csv (+ training_qa.csv).
Decodes constrained Qwen decisions (argmax / exact-set / 24-permutation), joins the
incumbent counterpart by qa_id, and evaluates on the disagreement set:
  AUROC/AUPRC of the soft score for W->R vs R->W, risk-coverage, precision/net at
  top-k (row-counted and unique-session-counted), category breakdown, oracle headroom,
  winner clustering. No submissions, no test data.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def auroc(scores, labels):
    # Mann-Whitney U; labels in {0,1}, higher score = more likely positive.
    pos = sorted((s for s, y in zip(scores, labels) if y == 1))
    neg = sorted((s for s, y in zip(scores, labels) if y == 0))
    if not pos or not neg:
        return float("nan")
    import bisect
    wins = 0.0
    for s in pos:
        lo = bisect.bisect_left(neg, s)
        hi = bisect.bisect_right(neg, s)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def auprc(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    total = sum(labels)
    if total == 0:
        return float("nan")
    area, prev_rec = 0.0, 0.0
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / total
        area += prec * (rec - prev_rec)
        prev_rec = rec
    return area


def norm_set(s):
    return "".join(sorted(s))


def decode_choice(entry):
    lp = entry.get("letter_logp") or {}
    if not lp:
        return None, None, None
    ranked = sorted(lp.items(), key=lambda kv: kv[1], reverse=True)
    pred = ranked[0][0]
    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else -1e9)
    probs = {k: math.exp(v) for k, v in lp.items()}
    total = sum(probs.values())
    ent = -sum((p / total) * math.log(p / total + 1e-12) for p in probs.values())
    return pred, margin, ent


def decode_multi(entry, card_logprior):
    mem = entry.get("member_logp") or {}
    if not mem:
        return None, None, None
    letters = sorted(mem)
    p = {}
    for letter in letters:
        y, n = mem[letter]["Y"], mem[letter]["N"]
        m = max(y, n)
        ey, en = math.exp(y - m), math.exp(n - m)
        p[letter] = ey / (ey + en)
    sets = []
    for r in range(1, len(letters) + 1):
        for combo in itertools.combinations(letters, r):
            combo = "".join(combo)
            s = sum(math.log(p[L] + 1e-12) for L in combo)
            s += sum(math.log(1 - p[L] + 1e-12) for L in letters if L not in combo)
            s += card_logprior.get(len(combo), -1e9)
            sets.append((s, combo))
    sets.sort(reverse=True)
    pred = sets[0][1]
    margin = sets[0][0] - (sets[1][0] if len(sets) > 1 else -1e9)
    return pred, margin, None


def decode_sequence(entry):
    pairs = entry.get("pair_logp") or {}
    if not pairs:
        return None, None, None
    letters = sorted({c for key in pairs for c in key})
    if len(letters) != 4:
        return None, None, None
    scored = []
    for perm in itertools.permutations(letters):
        pos = {L: i for i, L in enumerate(perm)}
        s = 0.0
        for a, b in itertools.combinations(letters, 2):
            key = a + b if a + b in pairs else b + a
            first, second = (a, b) if pos[a] < pos[b] else (b, a)
            d = pairs[key]
            s += d[first] - math.log(math.exp(d[first]) + math.exp(d[second]) + 1e-12)
        scored.append((s, "".join(perm)))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][0] - (scored[1][0] if len(scored) > 1 else -1e9), None


def main(records_path, oof_path="champ/oof_e2e_baseline_20260909.csv",
         train_path="training_qa.csv"):
    records = load_jsonl(records_path)
    base = {r["qa_id"]: r for r in csv.DictReader(open(oof_path))}
    train = {}
    with open(train_path) as handle:
        reader = csv.DictReader(handle)
        if "\ufeffqa_id" in (reader.fieldnames or []):
            for row in reader:
                row["qa_id"] = row.pop("\ufeffqa_id")
                train[row["qa_id"]] = row
        else:
            for row in reader:
                train[row["qa_id"]] = row
    card_counts: Counter[int] = Counter()
    for row in train.values():
        if row["category"] == "multi" and row["source"] == "HAU":
            card_counts[len(str(row["answer"]).strip())] += 1
    total_cards = sum(card_counts.values())
    card_logprior = {k: math.log(v / total_cards) for k, v in card_counts.items()}

    disagreements = []
    stats = Counter()
    for entry in records:
        qa_id = entry["qa_id"]
        if qa_id not in base:
            stats["no_baseline_counterpart"] += 1
            continue
        truth = entry["truth"].strip()
        brow = base[qa_id]
        b_pred = brow["pred"].strip()
        cat = entry["category"]
        if entry.get("kind") in ("choice",):
            q_pred, margin, ent = decode_choice(entry)
        elif entry.get("kind") == "multi":
            q_pred, margin, ent = decode_multi(entry, card_logprior)
            if q_pred is not None:
                q_pred = norm_set(q_pred)
                truth = norm_set(truth)
                b_pred = norm_set(b_pred)
        elif entry.get("kind") == "pairs":
            q_pred, margin, ent = decode_sequence(entry)
        else:
            stats["undecodable"] += 1
            continue
        if q_pred is None:
            stats["undecodable"] += 1
            continue
        stats["decoded"] += 1
        b_ok = (b_pred == truth)
        q_ok = (q_pred == truth)
        if q_pred == b_pred:
            stats["agree"] += 1
            continue
        stats["disagree"] += 1
        if q_ok and not b_ok:
            outcome = "W2R"
        elif b_ok and not q_ok:
            outcome = "R2W"
        else:
            outcome = "BOTH_WRONG"
        disagreements.append({
            "qa_id": qa_id, "regime": entry["regime"], "session": entry["session"],
            "user": entry["user"], "category": cat, "incumbent": b_pred,
            "truth": truth, "qwen": q_pred, "margin": margin,
            "entropy": ent if ent is not None else float("nan"),
            "outcome": outcome,
        })
    total = len(disagreements)
    w2r = sum(1 for d in disagreements if d["outcome"] == "W2R")
    r2w = sum(1 for d in disagreements if d["outcome"] == "R2W")
    print(json.dumps({"records": len(records), "stats": dict(stats),
                      "disagreements": total, "W2R": w2r, "R2W": r2w,
                      "oracle_net": w2r - r2w}, indent=2))
    if not disagreements:
        return
    for regime in ("full", "pair"):
        subset = [d for d in disagreements if d["regime"] == regime]
        if subset:
            print(f"===== regime={regime} n={len(subset)} =====")
            report_set(subset)
    print("===== pooled n=%d =====" % total)
    report_set(disagreements)


def report_set(rows):
    w = [d for d in rows if d["outcome"] == "W2R"]
    r = [d for d in rows if d["outcome"] == "R2W"]
    print(f"W2R={len(w)} R2W={len(r)} both_wrong={len(rows) - len(w) - len(r)}")
    scores = [d["margin"] for d in rows]
    labels = [1 if d["outcome"] == "W2R" else 0 for d in rows]
    print(f"margin AUROC={auroc(scores, labels):.3f} "
          f"AUPRC={auprc(scores, labels):.3f} (positives={sum(labels)})")
    ranked = sorted(rows, key=lambda d: d["margin"], reverse=True)
    print("risk-coverage (top-k by margin):")
    for k in (1, 3, 5, 8, 10, 15, 20):
        if k > len(ranked):
            break
        top = ranked[:k]
        pw = sum(1 for d in top if d["outcome"] == "W2R")
        pr = sum(1 for d in top if d["outcome"] == "R2W")
        print(f"  k={k}: W2R={pw} R2W={pr} net={pw - pr} "
              f"precision={pw / max(pw + pr, 1):.3f}")
    print("unique-session top-k (first row per session by margin):")
    seen, sess_ranked = set(), []
    for d in ranked:
        if d["session"] not in seen:
            seen.add(d["session"])
            sess_ranked.append(d)
    for k in (1, 3, 5, 8, 10):
        if k > len(sess_ranked):
            break
        top = sess_ranked[:k]
        pw = sum(1 for d in top if d["outcome"] == "W2R")
        pr = sum(1 for d in top if d["outcome"] == "R2W")
        print(f"  sessions k={k}: W2R={pw} R2W={pr} net={pw - pr} "
              f"precision={pw / max(pw + pr, 1):.3f}")
    print("by category:",
          dict(Counter((d["category"], d["outcome"]) for d in rows)))
    print("winners by user:", dict(Counter(d["user"] for d in w)))
    print("winners by category:", dict(Counter(d["category"] for d in w)))
    print("regressions by user:", dict(Counter(d["user"] for d in r)))
    print("regressions by category:", dict(Counter(d["category"] for d in r)))


if __name__ == "__main__":
    main(sys.argv[1])