"""Minimal Kaggle-side inventory; emits no competition data, only structure."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


root = Path("/kaggle/input")
out = {"root": str(root), "entries": [], "suffix_counts": {}, "name_hits": []}
total = 0
for base, dirs, files in os.walk(root):
    dirs.sort()
    files.sort()
    for name in files:
        p = Path(base) / name
        try:
            size = p.stat().st_size
        except OSError:
            continue
        total += size
        rel = str(p.relative_to(root))
        out["suffix_counts"][p.suffix.lower() or "<none>"] = out["suffix_counts"].get(p.suffix.lower() or "<none>", 0) + 1
        if len(out["entries"]) < 200:
            out["entries"].append({"path": rel, "bytes": size})
        low = rel.lower()
        if any(x in low for x in ("train_qa", "test_qa", "meta.csv", "depth", "skeleton", "imu", "radar")):
            if len(out["name_hits"]) < 500:
                out["name_hits"].append({"path": rel, "bytes": size})
out["total_bytes_seen"] = total
out["suffix_counts"] = dict(sorted(out["suffix_counts"].items()))
print(json.dumps(out, indent=2))
Path("inventory.json").write_text(json.dumps(out, indent=2))
