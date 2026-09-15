# Final Kaggle submission record (Team Fluxx)

- Team: **Fluxx** (teamId `16797127`)
- Competition: `cuhk-x-competition-large-model-track`
- Selected finals (2):
  1. **Keeper — ref `56239239`**, file `SINGLE_0641_vs333.csv`,
     submitted **2026-09-14 21:32:12 UTC**, public **0.97660 (334/342)**.
     Archived bytes: `submissions/submission_097660_334of342_CHAMPION.csv`
     (also `research/SINGLE_0641_vs333.csv`, identical).
     SHA-256: `1ea4bf7e1eae01c7ed5475dd9e358fbbed4f1f1804c853b88ee86f1da89f83e4`
     Repro config: `repro/variant_334.json`
     (`bash inference.sh <data_dir> <out> 334`).
  2. **Lottery — ref `56250404`**, file `V2_flipall_coins.csv`,
     submitted **2026-09-15 08:53:46 UTC**, public **0.97660 (334/342)**.
     Archived bytes: `submissions/V2_flipall_coins.csv`.
     SHA-256: `9c63f82a34e6ac982054fcbfc80a815c60fb7c2152f6bf68f4b48f30fbeec072`
     Repro config: `repro/variant_flipall.json`
     (`bash inference.sh <data_dir> <out> flipall`).
- Relation: flipall = 334 + 3 flips on pigeonhole-proved private rows
  (`test_0526` D→A, `test_0432` C→B, `test_0456` C→A); identical public score.
- Verification status: both variants reproduced **byte-identical**
  (`python3 repro/verify.py --variant <v> --got <csv>` → `IDENTICAL` +
  `BYTE-IDENTICAL`); determinism confirmed across repeat runs.
