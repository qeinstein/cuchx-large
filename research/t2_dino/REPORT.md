# T2-DINO: DINO-family action-perception specialist — experiment package

Status: PREPARED, not pushed. Owner runs one command per stage (below).

## 1. Prior-DINO audit verdict: 2 real bugs + 1 methodological caveat

Read: `champ/build_dino_frames.py`, `champ/build_dino_clipfeats.py`,
`champ/harn_dino.py`, `champ/run_loc_dino.py`, plus `champ/build_depth.py`,
`champ/pipeline.py` (fusion), `champ/dense.py` (DINO_K hook),
`champ/run_dense_dino.py`, `research/t2_crop/*`.

**BUG 1 (blocking): `build_dino_frames.py` cannot regenerate `dino_frames.npz`
today.** Its `video_path()` defaults to `mod='Depth'` → `HAU/<u>/<t>/Depth/Depth.mp4`,
but the data layout is `Depth_Color/Depth_Color.mp4` + `Thermal/Thermal.mp4`
(verified on disk; zero bare `Depth/` dirs exist). Missing files hit a silent
`continue`, so the committed script encodes 0 clips and then crashes at
`next(iter(store.values()))`. The missing npz can only be rebuilt with a path
fix. The new kernel uses the correct layout.

**BUG 2 (quality): grayscale conversion discards the depth signal's encoding.**
`build_dino_frames.py` does `cvtColor(BGR2GRAY)` then channel-replication —
but `Depth_Color` is a *rainbow colormap* where depth lives in hue, and rainbow
luminance is non-monotonic in depth. The old features threw away hue. The new
kernel keeps COLOR (BGR→RGB, as proven in `research/t2_crop/build_dino_crops.py`
and its pilot). Expect the old 35.4% standalone number to be pessimistic vs
color input; the kernel re-measures it.

**CAVEAT (not a bug): the OOF protocol itself is clean, but the fusion weight
is circular.** `harn_dino.py` is genuinely subject-disjoint: `folds(allu,5)`
splits by user, `StandardScaler` is inside the per-fold pipeline (fit on train
users only), descriptors use no labels. `run_loc_dino.py` likewise (per-clip
smoothing cannot leak across clips). The `409/429 → 417/429` HARn-single claim
in `pipeline.py` consumes subject-disjoint OOF caches — valid as far as it
goes — BUT `W_HDINO=0.3` is described as "the held-out optimum", i.e. the
weight was picked on the same OOF it is reported on (~+8 questions, plausibly
partly selection noise). The kernel's fusion test + `oof_eval_cpu.py` include a
split-half nested weight check (pick w on even users, report on odd and vice
versa) to de-bias this.

Minor notes: (a) `champ/run_dense_dino.py` contains zero DINO references despite
its name — it is the plain dense-OOF script (misnomer only). (b) `run_loc_dino`
resampling `idx=(arange(T)*len(A))//T` is a no-op when the alignment claim
holds, nearest-fill otherwise — fine. (c) `build_dino_clipfeats` FFT assumes
10 fps — correct for Depth only, not Thermal (25 fps). (d) The `depth_n==nskel`
alignment claim traces to a docstring in `build_depth.py` with no surviving
verification script; encoding metadata (320×240@10fps Depth vs 10 Hz skeleton)
makes it plausible. The kernel re-verifies it per clip
(`CAP_PROP_FRAME_COUNT` vs `meta.nf`) and records mismatches in the manifest.
(e) 224×224 squash of 4:3 frames distorts aspect; kept (proven recipe), flagged
as a follow-up ablation, not changed blind.

## 2. Backbone decision

| Candidate | Load string | Verdict |
|---|---|---|
| DINOv2 ViT-S/14 | `torch.hub.load('facebookresearch/dinov2','dinov2_vits14',...)` | PRIMARY. Proven in-repo; 384-d; full extraction Depth+Thermal × full+crop. |
| DINOv2 ViT-B/14 | `...,'dinov2_vitb14',...` | ABLATION in full stage (Depth full-stream only, budget-guarded). Same proven repo family. |
| DINOv3 | unknown entrypoint (HF-first release; torch.hub support unverified, no internet probing done) | PROBE ONLY: kernel records `torch.hub.list('facebookresearch/dinov3')` in manifest; no full encode gated on it. Promote to a follow-up kernel iff the probe shows a usable entrypoint. |
| CLIP (`openai/clip`) | needs internet too | DEPRIORITIZED: language-supervised, weaker for colorized depth; DINO first. |

**Internet decision: `enable_internet=true`.** Kernels have no internet unless
enabled, and torch.hub must fetch the repo + weights (~85 MB S, ~330 MB B).
Vendoring weights into a dataset would need the hub repo vendored too (or
custom checkpoint surgery) for zero benefit here — outputs are downloaded
artifacts for local science, not a competition submission, so internet-on is
unrestricted. Precedent metadata in-repo uses `enable_internet=false` only for
CPU/inventory kernels; this is the first GPU backbone kernel.

## 3. Experiment design (what the kernel does)

- **Extraction (GPU):** stride-1 COLOR DINOv2-S on train-HAU (810 clips) +
  test (195 meta + 11 extra dirs on disk), Depth_Color + Thermal, full-frame +
  t2_crop 1.6× actor box (imported from the attached dense-inputs dataset;
  center-crop fallback). ViT-B/14 Depth-full ablation if budget allows.
  Time-guard degrades gracefully (drop B first) and always saves + manifests.
- **Heads (subject-disjoint 5-fold, same `folds()` seed as champ):**
  (a) HARn action identity — pooled-LogReg per stream (exact `harn_dino.pool`
  replica for comparability) + full/crop concat + torch attention-pooling MLP
  on the best stream; exact top-1 + per-action tables.
  (b) HAU clip action-set presence — multilabel LogReg OOF, macro/micro AP +
  per-action AP (labels from single/multi/combination answers).
  (c) Fusion test vs skeleton+IMU — weight sweep combining DINO OOF with the
  shipped `harn_clf.npz` OOF on identical clips (clip top-1) 
...[truncated 1694 chars]