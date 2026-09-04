# Emotion/manner physics map — 2026-09-04

This is an analysis-only, training-only audit of the preserved champion emotion OOF snapshot.
It does not train a model, modify the shared pipeline, or create a Kaggle submission.

Start with [RESULTS.md](RESULTS.md). It records the snapshot contract, residual confusion map,
physical signal interpretation, action/script context, and the limits on the conclusions.

Reproduce from the repository root:

```sh
./venv/bin/python research/emotion_manner_physics_map_20260904/audit_emotion_physics.py
```

Key outputs under `tables/`:

* `input_provenance.csv` — exact input hashes.
* `emotion_residual_ledger.csv` — row-level champion OOF ledger and within-session features.
* `residual_confusion_*.csv` and `recurrent_confusion_pair_summary.csv` — directed, symmetric, and recurrent-pair residuals.
* `raw_skeleton_dynamics.csv`, `physical_signal_screen_within_session.csv`, and `residual_pair_physical_map.csv` — transparent temporal measurements and action-normalised comparisons.
* `action_context_*.csv` and `script_context_residuals.csv` — descriptive action/script slices.

The audit is intentionally pinned to `champ/audit_champ.csv` (734/809 emotion OOF), rather
than silently mixing it with the differently scored `champ/oof_final.csv` artifact.
