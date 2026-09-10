"""Stage a compact Kaggle dataset using hard links, preserving the source corpus."""
from __future__ import annotations

import json
import os
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "research/takeover_20260908/kaggle_exact_dataset"
STAGE.mkdir(parents=True, exist_ok=True)


def link(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if dst.stat().st_ino == src.stat().st_ino:
            return
        raise FileExistsError(dst)
    os.link(src, dst)


for name, src in {
    "training_qa.csv": ROOT / "training_qa.csv",
    "test_qa.csv": ROOT / "test_qa.csv",
    "meta.csv": ROOT / "champ/meta.csv",
    "vocab.json": ROOT / "champ/vocab.json",
    "champion.csv": ROOT / "submission_097076_332of342_CHAMPION.csv",
}.items():
    link(src, STAGE / name)

with (ROOT / "champ/meta.csv").open(newline="", encoding="utf-8") as handle:
    meta = list(csv.DictReader(handle))
for row in (row for row in meta if row["kind"] == "train_hau"):
    src = ROOT / "hf_data_manual/HAU" / row["user"] / row["trial"] / "Depth_Color/Depth_Color.mp4"
    safe = f"hau__{row['user']}__{row['trial']}.mp4"
    link(src, STAGE / safe)

# The original exact-interval dataset contained only the short HARn-style test clips.
# The end-to-end video challenger needs identical train/inference handling, so stage every
# distributed test Depth_Color video (HAU and HARn) that is present locally.
for i in range(1, 209):
    src = ROOT / "hf_data_manual/large_model_track_test" / f"LM_test_{i:04d}" / "Depth_Color/Depth_Color.mp4"
    if src.exists():
        link(src, STAGE / f"test__LM_test_{i:04d}.mp4")

manifest = {
    "dataset": "toheebogunade/cuchx-exact-supervision-20260908",
    "parent_videos": len(list(STAGE.glob("hau__*.mp4"))),
    "test_videos": len(list(STAGE.glob("test__*.mp4"))),
    "bytes": sum(p.stat().st_size for p in STAGE.glob("*.mp4")),
}
(STAGE / "staging_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
