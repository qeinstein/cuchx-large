#!/usr/bin/env python3
"""Pack the hard-linked HAU training videos into one upload-efficient tar archive."""

import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/takeover_20260908/kaggle_exact_dataset"
OUT = Path(__file__).with_name("dataset_stage") / "hau_videos.tar"
videos = sorted(SOURCE.glob("hau__*.mp4"))
if len(videos) != 810:
    raise SystemExit(f"expected 810 HAU videos, found {len(videos)}")
with tarfile.open(OUT, "w") as archive:
    for index, video in enumerate(videos, 1):
        archive.add(video, arcname=video.name, recursive=False)
        if index % 100 == 0:
            print(f"packed {index}/{len(videos)}", flush=True)
print(OUT)
print(OUT.stat().st_size)
