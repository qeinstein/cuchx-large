# T2-DINO dataset spec: NO new upload needed

The kernel attaches two existing datasets. Verified 2026-09-14 via
`kaggle datasets files` (full pagination).

## Inputs (attached, read-only)

| Dataset | Size | Verified contents |
|---|---|---|
| `toheebogunade/cuchx-videos-20260914` | 2.7 GB | 1623 train-HAU mp4s (`HAU/<user>/<trial>/{Depth_Color,Thermal}/*.mp4`) + 337 test mp4s (`large_model_track_test/LM_test_*`). Matches local tree exactly (1623/337). No HARn videos exist anywhere (train-HARn descriptors come from parent slicing). |
| `toheebogunade/cuchx-dense-inputs-20260914` | 102 MB | `meta.csv`, `training_qa.csv`, `test_qa.csv`, `vocab.json`, `skel_seq.npz` (49 MB), `imu_seq.npz` (29 MB), `dense_logits.npz` (24 MB), `harn_clf.npz`, `feats.csv`, `make_crops.py` (t2_crop box module, cv2+numpy only — imported by the kernel), plus champ `.py` sources. |

Kernel input paths: `/kaggle/input/cuchx-videos-20260914`,
`/kaggle/input/cuchx-dense-inputs-20260914` (see `kernel-metadata.json`).

## Outputs (kernel `/kaggle/working`, downloaded after run)

Frame totals (from `meta.csv` `nf`): Depth ≈ 165.6k frames
(train-HAU 143,598 + test 22,011); Thermal ≈ 2.5x ≈ 414k frames.

| File | Contents | Est. size |
|---|---|---|
| `dino_s_depth.npz` | ViT-S/14 Depth, keys `<qa_path>\|full`, `<qa_path>\|crop1.6`, (T,384) float16 | ~150 MB compressed |
| `dino_s_thermal.npz` | ViT-S/14 Thermal, same key scheme (25 fps; per-clip n/fps in manifest) | ~350 MB compressed |
| `dino_b_depth.npz` | ViT-B/14 Depth full-stream only, (T,768) float16 (skipped if budget guard trips) | ~140 MB compressed |
| `oof_harn_action.npz` | Per-stream HARn OOF log-probs (`<stream>\|oof\|<path>`), `classes` | < 30 MB |
| `oof_hau_pool.npz` | HAU presence OOF probs/labels/paths per modality, `actions` | < 5 MB |
| `harn_action_report.json` | Exact top-1 + per-action table per stream | tiny |
| `hau_pool_report.json` | macro/micro AP + per-action AP per modality | tiny |
| `fusion_curve.csv` | DINO+skeleton+IMU weight sweep (HARn top-1, same clips) | tiny |
| `manifest.json` | Backbone availability (incl. DINOv3 probe), fps, coverage, Depth-frames==nf alignment audit, timings, fallbacks | tiny |
| `dino_pilot.npz` | Pilot-stage only (10 clips x 2 streams) | tiny |

Total full-stage download ≈ 0.6–0.7 GB. No dataset push required;
the owner downloads outputs with `kaggle kernels output ...` (see REPORT.md).
