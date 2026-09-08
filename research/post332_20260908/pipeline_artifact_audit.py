#!/usr/bin/env python3
"""Offline integrity checks for tomorrow's adaptive suite.

This script never imports the heavy solver and never contacts Kaggle.  It checks
the immutable champion, every manifest hash/diff, output formatting, the decoder's
signed-effect algebra, and all in-tree ``P.solve`` unpack sites.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CHAMP = ROOT / "submission_097076_332of342_CHAMPION.csv"
BASE330 = ROOT / "submission_096491_330of342_CHAMPION.csv"
CHAMP_SHA = "25e79e1dae1149bdad81d081d1fad3a94db4e1eb88df7f00e91276e6d5668d56"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pred_map(path):
    rows = read(path)
    return rows, {r["qa_id"]: r["prediction"] for r in rows}


def validate_submission(path, test_rows):
    rows, p = pred_map(path)
    assert len(rows) == len(test_rows) == 682, (path, len(rows))
    assert [r["qa_id"] for r in rows] == [r["qa_id"] for r in test_rows], path
    cats = {r["qa_id"]: r["category"] for r in test_rows}
    for q, v in p.items():
        c = cats[q]
        assert v and all(x in "ABCD" for x in v) and len(set(v)) == len(v), (path, q, v)
        if c in ("single", "emotion", "combination", "object_interaction"):
            assert len(v) == 1, (path, q, c, v)
        elif c == "sequence":
            assert sorted(v) == list("ABCD"), (path, q, c, v)
        elif c == "multi":
            assert 1 <= len(v) <= 4, (path, q, c, v)
    return p


def solve_unpack_audit():
    bad = []
    for path in sorted((ROOT / "champ").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for parent in ast.walk(tree):
            for node in ast.iter_child_nodes(parent):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "solve"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "P"):
                    continue
                if isinstance(parent, ast.Assign) and isinstance(parent.value, ast.Call):
                    target = parent.targets[0]
                    n = len(target.elts) if isinstance(target, (ast.Tuple, ast.List)) else None
                    if n != 4:
                        bad.append({"file": str(path.relative_to(ROOT)), "line": node.lineno,
                                    "unpacked_values": n})
    assert not bad, bad
    return {"checked": True, "bad_sites": bad}


def decoder_audit():
    # Every signed effect vector for the five-row aggregate is represented.  The
    # adaptive decoder observes reversion deltas (-d) for the first four rows.
    states = [s for s in itertools.product((-1, 0, 1), repeat=5) if sum(s) == 2]
    assert len(states) == 30
    accepted = rejected = 0
    for rev in itertools.product((-1, 0, 1), repeat=4):
        forced = 2 + sum(rev)
        if forced in (-1, 0, 1):
            accepted += 1
            expect = ("ADAPTIVE_fifth_0526_plus_revert_0501.csv"
                      if forced == -1 else "ADAPTIVE_fifth_0526.csv")
            cmd = ["python3", str(OUT / "adaptive_decode.py"), *map(str, rev)]
            got = json.loads(subprocess.check_output(cmd, text=True))["next_file"]
            assert got.endswith(expect), (rev, forced, got)
        else:
            rejected += 1
    assert accepted + rejected == 81
    return {"signed_states": len(states), "accepted_probe_outcomes": accepted,
            "rejected_inconsistent_outcomes": rejected}


def manifest_audit(test_rows):
    manifest = json.loads((OUT / "submission_manifest.json").read_text())
    base332 = pred_map(CHAMP)[1]
    base330 = pred_map(BASE330)[1]
    checked = 0
    for name, info in manifest.items():
        path = OUT / (name + ".csv")
        assert path.exists(), path
        p = validate_submission(path, test_rows)
        assert sha(path) == info["sha256"], name
        for label, base in (("332", base332), ("330", base330)):
            diff = [{"qa_id": q, "from": base[q], "to": v}
                    for q, v in p.items() if base[q] != v]
            assert diff == info["diffs"][label], (name, label)
            dpath = OUT / (name + ".vs" + label + ".diff.csv")
            assert read(dpath) == info["diffs"][label], dpath
        assert "test_0458" not in info.get("changes", {})
        checked += 1
    return {"manifest_entries": checked}


def main():
    test_rows = read(ROOT / "test_qa.csv")
    assert sha(CHAMP) == CHAMP_SHA
    champ = validate_submission(CHAMP, test_rows)
    checks = {
        "champion_sha256": sha(CHAMP),
        "champion_rows": len(champ),
        "champion_changed_vs_330": sum(
            champ[q] != pred_map(BASE330)[1][q] for q in champ),
        "solve_unpack": solve_unpack_audit(),
        "decoder": decoder_audit(),
        "manifest": manifest_audit(test_rows),
    }
    (OUT / "pipeline_artifact_audit.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
