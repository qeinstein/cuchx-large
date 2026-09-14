# T2 modality / sync / metadata audit (modality-auditor)

Date: 2026-09-14. Repo: CUHK-X Large Model Track. Scope: read-only except this file.
Constraint note: CPU-only, no cv2/ffprobe/PIL on PATH; video probed via existing
`champ/*.npz` caches + mp4 file sizes (no frame decode). All counts verified by execution.

## 1. Modality-coverage table

### 1a. Train HAU (sessions)

| Modality | Present | Missing | Note |
|---|---|---|---|
| Depth_Color mp4 | 810/814 | 4 visual-only trials have no LMT row at all: `user22/1-1-1`, `user22/3-1-3`, `user7/3-3-2`, `user7/6-3-2` | |
| Thermal mp4 | 813/814 | `user1/1-2-3` only | |
| IR | 0 | n/a | **No IR modality exists anywhere on disk** (only substring hits in action names, e.g. `2_Comb_hair`, `10_Stir_drinks`) |
| LMT Skeleton dir | 807/810 | 3 genuinely absent: `user1/1-2-2`, `user2/1-1-3`, `user2/3-1-1` | +10 more with short filenames dropped by regex (Bug 1) |
| LMT IMU dir | 802/810 | 8: `user5/7-2-{1,2,3}`, `user22/4-3-2`, `user21/1-2-2`, `user17/3-3-{1,2,3}` | |
| LMT Radar non-empty | 795/810 | 8 missing file + 7 header-only (43 B) | missing: `user2/1-1-{1,2,3}`, `user3/1-1-{1,2,3}`, `user8/3-3-3`, `user1/1-2-3` |
| meta.csv f0 finite | 797/810 | 13 NaN | 3 no-dir + 10 short-name (recoverable, Bug 1) |
| feats.csv sk non-null | 797/810 | 13 | same 13 |
| feats.csv imu non-null | 789/810 | 21 | 8 missing dir + rest parse/shape failures |
| feats.csv rad non-null | 795/810 | 15 | = non-empty radar files exactly |

### 1b. Train HARn (segments, 2927 LMT units, **no HARn mp4s on disk** — HARn are sub-intervals of HAU)

| Modality | Present | Missing |
|---|---|---|
| Skeleton (all long filenames) | 2927/2927 | 0 |
| IMU dir | 2903/2927 | 24 (top: `6_Drink_water` 4, `9_Pour_drinks` 4, `36_Walk` 4) |
| Radar non-empty | 1409/2927 | 13 missing + **1505 header-only empty (51.4%)** |
| meta f0 finite | 2927/2927 | 0 |
| feats rad non-null | 1409/2927 | 1518 (= missing+empty exactly) |

### 1c. Test (visual 198 clips vs LMT 195 units — **not the same set**)

| Modality | Present | Missing |
|---|---|---|
| Depth_Color mp4 | 198/198 visual dirs | `LM_test_0030` has LMT but **no visual dir at all** |
| Thermal mp4 | 140/198 | **58 lack Thermal**: `LM_test_0001`–`0006`, `0008`–`0022`, … (all test clips < ~0065 that are HARn-like carry Depth only) |
| IR | 0 | n/a — does not exist |
| In LMT but not visual (8) | `LM_test_0030/0034/0038/0039/0103/0104/0105/0106` | no mp4 to verify against |
| Visual but not in LMT (11) | `LM_test_0007/0023/0027/0046/0050/0051/0054/0055/0056/0059/0062` | no skeleton/IMU/radar at all |
| LMT IMU dir | 190/195 | `LM_test_0034/0103/0104/0105/0106` |
| LMT Radar non-empty | 136/195 | 6 missing + 53 header-only |
| meta f0 finite | 193/195 | `LM_test_0034` (23 skel files), `LM_test_0106` (55 skel files) — both short-name, recoverable (Bug 1) |

Path-staleness note: every `test_qa.csv` path says `large_model_track_test/LM_test_XXXX/Depth/Depth.mp4`
but disk has `Depth_Color/Depth_Color.mp4` — the QA path never resolves (`os.path.exists` False).

Formats/sizes (typical): Depth mp4 ~0.3–3.5 MB, Thermal mp4 ~1.6 MB; IMU 2 CSVs
(`up(LA+RA+C).csv`, `down(LL+RL).csv`, 21 cols incl. 5 sensors WTC/WTRA/WTLA/WTRL/WTLL);
Radar.csv 9 cols (`timestamp,frame,DetObj#,x,y,z,v,snr,noise`); skeleton 17-joint
`keypoints` (x, y, depth-like z in ~0–1.46 m) + `keypoint_scores`.

## 2. Unused-signal audit

### 2a. Exactly what enters `feats.csv` (`champ/build_feats.py`)

- Skeleton → 31 cols: speed/accel/jerk stats, wrist+ankle speed, `sk_conf` (mean score),
  `sk_T`, `sk_scale`, hip path/net, cadence (`sk_cad_hz/pow`, spectral centroid),
  active fraction, IQR. Multi-person frames collapsed to max-score person.
- IMU → 16 cols: norms of acc + gyro **concatenated across all 5 sensors and both files**,
  plus |Δacc| stats and two energies.
- Radar → 7 cols (prior claim "ONE scalar" is **wrong**): `rad_v_{mean,std,p50,p90,max}`
  over |v|, plus `rad_n`, `rad_frames`. But all seven derive from the `v` column + row counts.
- Downstream (`harn.py feat_cols`) also adds `nf`, `f0`. Sequence models (`dense.py`,
  `harn_clf.py`) additionally use per-frame skeleton K + per-frame IMU (same acc/gyro norms).

### 2b. Raw signals NEVER used

1. **Radar spatial + quality**: `x,y,z` (per-point 3-D position), `snr`, `noise`,
   `DetObj#`, `timestamp`/`frame` temporal structure, point-count dynamics. 90.6% of `v==0`
   in sampled clip, so |v| stats sit on a near-degenerate column while geometry is ignored.
2. **IMU per-sensor identity**: 5 sensors (chest C, arms LA/RA, legs LL/RL) are pooled into
   one bag; arm-vs-leg energy ratio (highly action-discriminative) is unrecoverable downstream.
3. **IMU magnetometer (3 cols), euler angles (3), quaternion (4), temperature, battery**: parsed
   never. (Angles/quaternion are orientation — complementary to acc/gyro norms.)
4. **Thermal temporal**: `Thermal.mp4` exists for 813/814 train + 140/198 test but no frozen
   pipeline file reads it (`grep thermal champ/pipeline.py champ/core.py champ/dense.py
   champ/decode.py` = empty; only research-only `build_thermal_*` scripts reference it).
5. **Depth appearance at inference**: `pipeline.py` supports DINO fusion (`W_HDINO=0.3`) but
   `champ/harn_dino.npz` and `champ/dino_frames.npz` **do not exist** → `has_dino` is always
   False → the Depth modality currently contributes **zero** to every test prediction.
6. **Skeleton confidence**: `keypoint_scores` are 1.0 for 100% of sampled joints (2 clips,
   ~1000 joints) — the column is degenerate; `sk_conf`=1.0 everywhere. Not a usable signal
   (no bug — just no information).
7. **Radar-empty indicator**: 51% of HARn radar files are header-only; emptiness itself may
   correlate with action/recording condition but is never encoded (becomes NaN → imputed).

### 2c. Pilot diagnostics (subject-disjoint, protocol fixed before running)

Task: 3-way Walk (`36_Walk`) vs Eat (`7_Eat_food`) vs Sit-down (`34_Sit_down`) on HARn
segments; train users 1–7+16–24 pool, held-out users `user8,user9`; LogisticRegression.
N reported after modality-availability filtering.

| Pilot | Features | n_train/n_test | Acc | Majority |
|---|---|---|---|---|
| Radar-spatial (unused: x/y/z/snr/noise/drift/range/29 feats) | full | 265/60 | **0.583** | 0.650 |
| Radar v-only (pipeline-like: v stats + counts) | 6 | 265/60 | 0.517 | 0.650 |
| IMU-full (unused: mag/angles/per-sensor acc, 40 feats) | full | 567/62 | 0.532 | 0.661 |
| IMU-pipe (acc/gyro norms, 12 feats) | 12 | 567/62 | **0.548** | 0.661 |

Honest reading: **both pilots are negative vs majority** on this 3-way slice (small-N,
n_test≈60; radar missingness 48.8% on the slice further restricts the sample). The only
positive delta is radar-spatial over v-only (**+6.6 pp**, 0.583 vs 0.517), consistent with
"v is 90% zeros; geometry carries the residual signal" — but neither is usable alone.
IMU magnetometer/angles/per-sensor add nothing here (−1.6 pp, noise). Scripts: `/tmp/radar_pilot2.py`,
`/tmp/imu_pilot.py` (kept outside the repo per scratch policy). Do **not** over-claim: these
pilots test single-modality separability of 3 actions, not marginal value inside the fusion.

## 3. Sync / alignment audit

Checked: skeleton filename ts vs frame id (2 train clips, gap-free 10.000 fps, monotone,
zero missing frames); IMU vs skeleton wall-clock (IMU span within ±0.1 s of skeleton span
on `user1/1-1-1`, `user1/1-1-2`); radar vs skeleton same-session timestamps (radar frames
93–1211 over the same wall-clock span — radar `frame` is its own clock, not the global
video axis, correctly never joined on by the pipeline).

### HARn ⊂ HAU nesting (global frame axis + timestamps): PASS with prefix bug

- 2905/2927 (99.2%) nest strictly inside `HAU/<user>/<trial>` on both `[f0,f1]` and `[t0,t1]±0.5 s`.
- All 22 non-nested rows are explained by **parent f0 NaN** (child skeleton valid in every case).
  Zero frame-containment violations among finite rows.
- Child skeleton JSONs are byte-identical copies of parent frames (verified
  `HARn/0_Wash_face/user16/1-1-1` first frame ∈ parent predictions, `a==b` True).

### Bug 1 (HIGH — recoverable data loss, affects test): short-filename skeleton dropped by regex

- 10 train HAU + 2 test clips store valid skeleton as `Color_%08d.json` (no timestamp):
  train `user1/{1-2-3,2-1-1}`, `user2/{1-1-1,1-1-2}`, `user3/{1-1-1,1-1-2,1-1-3,6-2-1,6-2-2,6-2-3}`;
  test `LM_test_0034` (23 files), `LM_test_0106` (55 files).
- `build_meta.py` regex requires the timestamp → `f0/f1/t0/t1` = NaN; `build_feats.py` regex
  `Color_.*_(\d{8})` also fails on short names → all `sk_*` NaN despite data on disk.
- Pipeline consequence: `clip_has_sensor=False` on these clips triggers the NO_SENSOR action
  restriction (4 actions) in `pipeline.py` — a **wrong biconditional branch for 2 live test clips**.
  (3 further train clips genuinely lack a Skeleton dir: `user1/1-2-2`, `user2/1-1-3`, `user2/3-1-1`.)
- Fix: accept `Color_(\d{8}).json` in both regexes (frame ids kept, `t0/t1/fps` NaN or inherited
  from HARn children/long-name siblings); rebuild `meta.csv`/`feats.csv`/`skel_seq.npz`.

### Bug 2 (LOW — doc claim false, minor numeric effect): `depth_n == nskel` off-by-one

- `build_dino_frames.py` header claims "verified: depth_n == nskel"; the `depth_mnv3_frames.npz`
  cache (stride-2, 0-based video indices) shows video_n − skel_n = **−1 on 43/80 clips, 0 on 37/80**.
  The mapping "video frame i == global frame f0+i" holds but video runs one frame short half the time;
  `harn_dino.py` slicing (`index = global − parent_f0`) can address one past the end on the last frame.
- 2 depth-cache clips (`HAU/user2/1-1-1`, `1-1-2`) have no `skel_seq.npz` entry — same Bug 1 root cause.

## 4. Object-category evidence verdict

Test object rows: 21 (`test_0522`–`test_0542`), 21 unique clips (`LM_test_0009`…`0061`, all <0065).

| Evidence | Count |
|---|---|
| Depth mp4 exists (0.3–2.7 MB, shows the manipulated object) | 20/21 (`LM_test_0030` has no visual dir) |
| Thermal mp4 | 0/21 (all in the 58 Thermal-less test clips) |
| LMT skeleton+IMU | 16/21; 5 clips have **no LMT at all** (`0023/0027/0054/0055/0056`) |
| Radar non-empty | 3/21 (`0024/0030/0035`); rest header-only |

Pipeline object branch (`pipeline.py` L317–336 + `harn.fit_object_prior`): picks argmax action
from sensor classifiers, then answers from the **train action→object co-occurrence table**
(`pri[action][object]`); no-sensor clips marginalise the prior over 4 actions. No visual read,
no detector, no depth/thermal feature enters the object decision anywhere.

**Verdict: YES — the object decision is currently sensor-blind AND vision-blind.**
The object token is never observed; it is imputed from the action prior. Depth frames depicting
the object exist for 20/21 clips but are never read by this branch (and finding §2b.5 means no
visual path exists at inference at all). The 5 no-LMT object clips are answered by prior alone.

## 5. Top-3 actionable recommendations (ranked by expected value)

1. **Fix the short-filename skeleton regex (Bug 1) and rebuild caches.** ~2 lines
   (`build_meta.py`, `build_feats.py`) + `skel_seq.npz`/`harn_clf.npz` refresh. Recovers 10 train
   clips for training AND fixes the sensor-biconditional branch on live test clips
   `LM_test_0034`/`LM_test_0106` (currently forced into the wrong 4-action set). Highest EV:
   directly changes test predictions at near-zero risk.
2. **Give the object branch eyes: depth-crop visual rerank on the 20/21 object clips with mp4s.**
   Even a frozen CLIP/DINO 4-way similarity between the question's object nouns and 3–5 sampled
   depth frames, fused 50/50 with the current prior, attacks the most blind decision in the system
   (21 questions, currently prior-only). Thermal is unavailable for all 21 — use Depth only.
3. **Encode radar emptiness + add 4 spatial stats; stop trusting |v| alone.**
   Add `rad_empty` indicator (51% HARn / 27% test-LMT are header-only — currently silent NaN) plus
   `x/y/z` spread and centroid drift (pilot: +6.6 pp over v-only, the only positive unused-signal
   delta measured). Cheap (extends `radar_feats`), removes a missingness-as-noise source from the
   HARn aggregate classifier. Do NOT invest in IMU mag/angles (pilot-negative) or skeleton scores
   (degenerate 1.0).

## Appendix — reproduction pointers

- Coverage: `hf_data_manual/HAU/`, `hf_data_manual/large_model_track_test/`,
  `hf_data_manual/LMT_(IMU,Radar,Skeleton)/{Training/{HAU,HARn},Testing/large_model_track_test}`.
- Pipeline reads: [build_feats.py](/home/fluxx/Workspace/cuchx-large/champ/build_feats.py),
  [build_meta.py](/home/fluxx/Workspace/cuchx-large/champ/build_meta.py),
  [pipeline.py](/home/fluxx/Workspace/cuchx-large/champ/pipeline.py),
  [harn.py](/home/fluxx/Workspace/cuchx-large/champ/harn.py).
- Pilots: `/tmp/radar_pilot2.py`, `/tmp/imu_pilot.py` (protocol in §2c).
- Key numbers re-verified in-session: radar-empty 1505/2927 HARn; video−skel −1 on 43/80;
  `harn_dino.npz`/`dino_frames.npz` absent; keypoint_scores 100% 1.0 (2-clip sample).
