# No-LMT HARn ablation

This isolated experiment tests a direct-Depth video specialist on the 13 real-test HARn clips that lack an LMT directory. It intentionally preserves all existing caches and submissions.

The protocol is subject-disjoint and uses only `Depth/Depth.mp4`, the one visual modality present on every affected test clip. It compares its held-out predictions with `oof_v8_final.csv`, because `champ/final.py` explicitly fills production `harn_no_evidence` rows from `submission_v8.csv`.

Run from the repository root:

```sh
./venv/bin/python experiments/no_lmt_harn_20260904/run_ablation.py --scope all --extract --evaluate --resume
```

`--scope qa` is a fast diagnostic; `--scope all` is the primary experiment and trains on all direct HARn training videos while holding every validation subject out.

Outputs are written only in this directory. The script never writes a submission or calls Kaggle.
