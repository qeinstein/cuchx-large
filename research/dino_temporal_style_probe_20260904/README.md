# EXP-EMO-DINO-TRAJ-001

This directory is an isolated probe of temporal DINOv2-Depth trajectories for
HAU Emotion/manner recognition. It does not change `champ/`, create a
submission, or use IMU, radar, skeleton, or DTW features.

`probe_dino_trajectory.py` converts each native-rate `(T, 384)` DINO sequence
into frame-delta trajectory features: native duration, path length, velocity,
acceleration, curvature, rhythm, ordered contiguous phase summaries, and
fixed-random-projection delta directions. It then compares only sibling clips
within inferred sessions, so the action script and subject are controlled as
nuisance factors.

The evaluation is intentionally two-layered:

- A subject-disjoint orientation diagnostic asks whether a held-out pair's true
  group ordering is preferred to its swap.
- A separate decision ledger proposes only option-valid transpositions of the
  immutable champion OOF prediction, then reports W->R, R->W, precision, net,
  and every fold at fixed confidence gates.

The second result is the promotion criterion. A high orientation score does not
by itself justify an override.

Run it with:

```sh
./venv/bin/python research/dino_temporal_style_probe_20260904/probe_dino_trajectory.py
```

Results are written only to `results/`; reruns refuse to overwrite that folder.
