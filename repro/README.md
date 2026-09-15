# Reproduction package (Team Fluxx)

Frozen solution for the two selected Kaggle final submissions. Verified
byte-identical end to end (see `repro/SUBMISSION_RECORD.md`).

## What reproduces what

| Variant | Kaggle ref | Archived artifact | Role |
|---|---|---|---|
| `334` | `56239239` | `submissions/submission_097660_334of342_CHAMPION.csv` | keeper |
| `flipall` | `56250404` | `submissions/V2_flipall_coins.csv` | private lottery ticket |

Pipeline: `champ/make_submission.py` (solver on committed frozen caches) →
`repro/apply_lineage.py` (51-row frozen correction lineage from
`repro/patches_334.json`, every entry asserting the pipeline value it
corrects) → variant extra flips (`repro/variant_*.json`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy/pandas/sklearn/scipy/torch-CPU, pinned
```

Working directory: package root (where `inference.sh` lives).

## Run

```bash
bash inference.sh <data_dir> <output_csv> [variant]
# variant: 334 (default) | flipall
```

`<data_dir>` is the official label-free Testing data; `test_qa.csv` is located
inside it (top level preferred, else up to 4 deep). No other file from
`<data_dir>` is read: all model features are frozen in committed per-clip
caches keyed by clip id. Output is `qa_id,prediction`, 682 rows, in the input
`test_qa.csv` order.

Verify against the archived bytes:

```bash
python3 repro/verify.py --variant 334 --got <output_csv>
python3 repro/verify.py --variant flipall --got <output_csv>
```

## Requirements

- Hardware: any x86-64 CPU box; 4GB RAM suffices (2GB verified tight but OK).
  No GPU needed (torch pinned to CPU/MPS; no `.cuda()` in the solver path).
- Runtime: ~1–3 minutes single-threaded (`OMP_NUM_THREADS=1` forced).
- Network: none. No downloads at inference (all weights-as-features are
  committed caches; see `repro/EXTERNAL_DATA.md`).
- Determinism: `PYTHONHASHSEED=0` forced; two runs are byte-identical
  (verified; the patch applier aborts on any pipeline drift).

## Method (one paragraph)

Per-clip sensor features (skeleton kinematics, IMU/radar statistics, dense
temporal action logits — all precomputed and committed) feed a session-block
solver:
clips are grouped into recording sessions, a shared action pool is solved per
session under category invariants (single = 1 action, combination = 2, …),
and specialist repairs (object↔single consistency, manner/emotion conditioning,
sequence total-order decode) adjust categories with OOF-gated weights. Small
sklearn heads (gradient boosting / log-reg) are fit on the frozen training
features at run time; missing-evidence rows fall back to the committed 332
checkpoint. The 51-row lineage then applies documented corrections (36
provenance mechanisms, 13 pinned validated values where the rebuilt pipeline
regressed vs the validated base, 2 banked singleton probes).

## Notes and limits

- Training inputs (`training_qa.csv`, feature caches) are committed frozen
  artifacts; only `test_qa.csv` is taken from `<data_dir>`. Bulk official
  videos/sensors are not included and not needed.
- `repro/HONOR_DECLARATION.md` is a placeholder: the signed template arrives
  from the organizers after the freeze.
