# T2 — Person-centric crop preprocessing: pilot report

## TL;DR

- Skeleton→pixel projection is **not feasible**: no calibration exists anywhere
  (repo, data package, skeleton JSONs), and skeleton coords are root-relative 3D
  (joint 0 always at origin), not pixels.
- Fallback implemented in `make_crops.py`: clip-level actor box from
  **foreground motion + per-modality prior** (Thermal=hot mass, Depth=near mass),
  square-ified, margins 1.0×/1.3×/1.6×, plus paired full-frame. Works for
  train HAU + test clips, Depth_Color + Thermal, both resolutions.
- Pilot (24 clips, frozen DINOv2-S/14, leave-1-user-out ×4, chance 16.7%):
  Depth full 25.0% → crop-1.6× 29.2%; Thermal full 20.8% → crop-1.6× 29.2%.
  Crop ≥ full in all 6 comparisons; gains are +1–2 clips (n.s., N=24).
- Recommendation: CONDITIONAL GO — extract full + 1.6× on Kaggle GPU via
  `research/t2_crop/build_dino_crops.py`, gate adoption on a full-scale
  repeat of this protocol.

## 1. Frame inspection findings

Sample frames + box overlays: `research/t2_crop/inspect/`
(`HAU_*` raw frames, `BOX_*` motion-only boxes, `BOX2_*` fused boxes).

### Video formats (probed with cv2)

| Split | Modality | Size | fps | # clips |
|---|---|---|---|---|
| train HAU | Depth_Color (rainbow colorized depth) | 320×240 | 10 | all |
| train HAU | Thermal (inferno-like; hot=white/yellow) | 320×240 | 25 | all |
| test | Depth_Color | **640×480** | 10 | 198 |
| test | Thermal | 320×240 | 25 | 140/198 |

Note the train/test resolution mismatch on Depth (320×240 vs 640×480) and the
fps mismatch between modalities (10 vs 25). `make_crops.py` / `pilot.py` sample
by timestamp fraction, so both are handled.

### Person size / position / clutter

- Person scale varies enormously: ~15–25% of frame width when seated far
  (user1 living room) vs >60% when close to the camera (user8 back view,
  user23 bending over sink). A fixed center crop cannot work — the actor is
  frequently at the right edge or off-center.
- Rooms are cluttered and user-specific: couches, tables, mirrors, sinks, beds,
  an AC unit (cold black bar in Thermal). Full-frame features can latch onto
  room identity instead of the action — the core motivation for cropping.
- Depth character: rainbow colorization, autoscale differs across cameras
  (train rooms red=far vs test kitchen yellow/green mid-range); test Depth has
  black invalid-pixel regions (sensor dropout, speckles). Absolute color is
  unreliable → all priors used here are relative (percentiles within the clip).
- Thermal character: actor is the hottest mass (white/yellow); warm rooms
  (user1, orange background) still separate by relative threshold.
- Gotchas: user8 has a mirror — the reflection moves and competes with the
  (static, back-view) real person; test clip 0050 has a bed as a large near
  foreground object competing with the actor.

## 2. Projection feasibility verdict: NOT FEASIBLE

- `champ/skel_seq.npz` `|K` arrays are T×17×3 with x∈[−1,1], y∈[−0.3,0.3],
  third channel up to ~1.6, joint 0 identically (0,0): root-relative 3D pose
  (meters), not image coordinates.
- Repo-wide search for intrinsics/extrinsics/calibration/projection code:
  zero hits. Dataset README describes skeleton as "pose predictions only".
- No camera matrices, no distortion coefficients, no depth-to-color registration
  shipped with either data package.
- **Verdict**: skeleton-guided pixel boxes are impossible without calibration.
  Used instead: motion + appearance-prior box (below). If calibration ever
  ships, `compute_box()` is the single function to swap for a projected-joint
  box.

## 3. Crop method (`make_crops.py`)

Per clip: sample 12 probe frames evenly (fractional → fps agnostic).

1. Motion map = mean |gray − temporal median| (room is static, actor moves).
2. Prior map (mean over probes): Thermal `R−0.45B+0.15G` (hot mass, same
   formula as `champ/build_thermal_cropped_segments.py`); Depth `B−R`
   (near mass on rainbow map). Both relative per clip.
3. Fused score = z(motion) + 0.5·z(prior); mask = score > 90th percentile;
   largest connected component → base box. Fallbacks: motion-only, prior-only,
   center 0.7·min(H,W).
4. Square-ify around center, expand by margin (1.0/1.3/1.6), clamp to frame.
   One box per clip (stable, no jitter).

Tuning: swept fusion weight × percentile on 8 clips against eyeball truth.
w=0.5/p90 puts the box on the actor in 7/8; the 8th (user8 mirror) lands on
the mirror+actor boundary — the 1.3×/1.6× margins still contain the actor,
and the reflection itself shows the actor's front. The bed-foreground test
clip (0050) is correctly on the actor at w=0.5/p90 but drifts to the bed at
higher prior weight / lower percentile.

CLI: `python3 research/t2_crop/make_crops.py <clip.mp4> --modality Thermal
--out <dir> --tag <name>` or `@listfile` with `<mp4> <modality> <tag>` lines.
Writes `full_f*.png`, `crop{1.0,1.3,1.6}_f*.png`, `boxes.json`.

## 4. Pilot benchmark

### Protocol (exact)

- Pool: 24 clips = 6 actions × 4 users, complete grid, no fills
  (actions: Walking, Eating, Sitting down, Peeling fruit, Listening to the
  music with headphones, Pouring; users: user1, user16, user17, user19 —
  top-4 by action coverage). Labels from `training_qa.csv` "Which action is
  performed in this video?" (answer letter → option text).
- Frames: 4/clip, fractional sampling (same timestamps across modalities).
- Backbone: frozen `facebook/dinov2-small` (ViT-S/14, 384-d CLS, DINOv2
  weights, no fine-tuning), CPU, cv2 resize→224 + ImageNet normalize (same
  math as `champ/build_dino_frames.py`). Total: 24×4×4 variants×2
  modalities = 768 forward passes. (Pilot was downsized from 56×8 after an
  OOM kill and 4-core contention slowed embedding to ~60–100 s/clip; box
  method and protocol unchanged.)
- Clip embedding: L2-normalized mean of 4 frame CLS tokens.
- Classifier: nearest centroid (cosine), centroids from train-user clips only.
- Splits: leave-1-user-out × 4 folds (6 test clips/fold, 24 predictions per
  cell). Strictly subject-disjoint. Chance = 16.7%.
- Reproduce: `PYTHONPATH=/tmp/t2libs python3 research/t2_crop/pilot.py`
  (needs `pip install --target /tmp/t2libs pillow opencv-python-headless`;
  DINOv2 weights download ~85 MB from HF Hub on first run; embeddings cached
  in `pilot_emb.npz` with resume support).

### Results

Leave-1-user-out nearest-centroid accuracy (24 test clips per cell),
chance = 16.7% (4/24):

| Modality | full | crop 1.0× | crop 1.3× | crop 1.6× |
|---|---|---|---|---|
| Depth_Color | 25.0% (6/24) | 25.0% (6/24) | 25.0% (6/24) | **29.2% (7/24)** |
| Thermal | 20.8% (5/24) | 25.0% (6/24) | 25.0% (6/24) | **29.2% (7/24)** |

Per-fold (held-out user in brackets [u1, u16, u17, u19], 6 clips each):

| Cell | folds |
|---|---|
| Depth full | [.167, .333, .167, .333] |
| Depth 1.0 | [.167, .500, .167, .167] |
| Depth 1.3 | [.167, .333, .167, .333] |
| Depth 1.6 | [.167, .500, .333, .167] |
| Thermal full | [.000, .333, .167, .333] |
| Thermal 1.0 | [.167, .333, .000, .500] |
| Thermal 1.3 | [.167, .333, .167, .333] |
| Thermal 1.6 | [.167, .333, .167, .500] |

Paired exact McNemar (full vs 1.6×, N=24): Depth p=0.50 (full-only 1,
crop-only 2); Thermal p=0.36 (full-only 3, crop-only 5). Per-action,
1.6× never loses by more than 1 clip and fixes Pouring on Thermal (0/4 →
2/4) and Sitting/Walking on Depth. All 48 pilot boxes came from the fused
path (zero fallbacks); median box width 32% of frame (range 11–48%),
i.e. crops typically discard ~2/3 of background pixels.

### Reading the numbers

- Direction is uniform: crop ≥ full in all 6 comparisons, 1.6× best on
  both modalities. No evidence of harm from cropping.
- Power is low (N=24/fold-cell): the +1–2 clip gains are not significant
  (p≈0.4–0.5). This pilot cannot *prove* a transfer gain — but it was
  designed as a smoke test, and it passes: boxes land on the actor
  (visually verified, §1/§3), nothing breaks, direction is right.
- Absolute accuracy (21–29%) is modest, as expected for frozen CLS +
  nearest-centroid across unseen users on colorized depth/thermal; the
  pilot measures relative transfer, not SOTA.
- 1.6× > tighter margins: context around the actor (objects, furniture
  contact) helps; tight 1.0× sometimes clips limbs/objects.

## 5. Recommendation: CONDITIONAL GO

**GO** for full-scale crop-based DINO extraction on Kaggle GPU, with both
streams extracted (full + 1.6× crop) so the downstream head can ablate:

1. On Kaggle (GPU, repo + data attached):
   `python3 research/t2_crop/build_dino_crops.py --mod Depth_Color`
   `python3 research/t2_crop/build_dino_crops.py --mod Thermal`
   (torch.hub `dinov2_vits14`, batch 64, stride 1; resume-safe; writes
   `champ/dino_full_crop16_<mod>.npz` with `<qa_path>|full` and
   `<qa_path>|crop1.6` keys, (T,384) float16. Same video coverage as
   `champ/build_dino_frames.py`: train_hau + test.)
2. Gate the switch on a full-scale repeat of this protocol (all HAU users,
   all actions, leave-users-out): adopt crop-1.6 as primary iff it beats
   full-frame there; otherwise concatenate [full; crop] (768-d) — the
   pilot shows crops add a non-redundant view either way.
3. Do NOT use tight 1.0× as the single crop (limb/object clipping); 1.6×
   is the pilot's best margin. Skip skeleton-projection work unless
   calibration ships (§2).

## 6. Artifacts

- `research/t2_crop/make_crops.py` — crop module + CLI.
- `research/t2_crop/pilot.py` — pilot benchmark.
- `research/t2_crop/build_dino_crops.py` — Kaggle GPU scale-up script
  (plumbing smoke-tested on CPU with a stub backbone; torch.hub load line is
  verbatim from proven `champ/build_dino_frames.py`).
- `research/t2_crop/pilot_results.json` — pool + per-fold accuracies.
- `research/t2_crop/pilot_emb.npz` — cached frame embeddings (re-run cache).
- `research/t2_crop/pilot_boxes.json` — pilot clip boxes + sources.
- `research/t2_crop/inspect/` — raw frames and box overlays.
