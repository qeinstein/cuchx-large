# T2 VLM Scout Report — frontier multimodal capacity & routed-ensemble feasibility

Date (UTC): 2026-09-14 ~19:00. Deadline: 2026-09-15 15:55 UTC (~21 h remaining).
Scope: 682 test questions; OOF validation = 4,087 train questions.

## 1. Rules verdict: YES, external/closed VLMs and APIs are allowed

Competition-specific rules (§2.6, fetched via `kaggle competitions pages`):

> "a. **External data and models are permitted in the Large Model Track**,
> including large vision-language models and external/closed-source APIs,
> provided they are reasonably accessible to all Participants at minimal cost
> (the "Reasonableness Standard") and you comply with all other obligations
> in these Rules (including §2.5 Winner License and §2.8 Winner's Obligations)."

Competition Description page confirms:

> "This is the **Large Model Track**: any model is allowed, including large
> vision-language models and external APIs."
> "Any model size, architecture, or pretrained weights are allowed; large
> vision-language models and external APIs are permitted."

Caveats (do not block a submission, but bind winners):

- §2.8: winners must deliver code + `inference.sh` + checkpoint + README that
  **reproduce** the submission, and pass a **live (Zoom) inference run** on
  fresh data. A >10% gap vs the private-leaderboard score = disqualification.
  An API-dependent solution is reproducible only while the API/model persists —
  pin exact model IDs and keep full response logs.
- "No using test-set answers in training; no manual labeling of test questions."
- Data is non-redistributable (CUHK-X License v2.0) — do not upload videos to
  third parties beyond inference calls; contact-sheet crops via API are a gray
  area the parent should consciously accept (prior work already did this).

## 2. Availability inventory (probed today, this box)

| Route | Status | Detail |
|---|---|---|
| Local GPU | NONE | CPU-only, 2.6 GB RAM, no `nvidia-smi` |
| Ollama / vLLM / LM Studio / gcloud | NONE | binaries missing |
| Local checkpoints | NONE useful | HF cache = 85 MB (`dinov2-small` only); no Qwen/LLAVA weights; no `.safetensors`/`.gguf` in repo |
| Python stack | CPU torch 2.11, transformers 5.1.0 | no diffusers/cv2/PIL stock; installed `pillow`+`opencv-headless` into `/tmp/pilotlibs` for frame extraction (works) |
| Env API keys | NONE | no `*_API_KEY`/`*_TOKEN` in env; no `~/.anthropic`, `~/.openai`, `~/.openrouter` configs |
| Repo-embedded OpenRouter key | **VALID, working** | in `benchmark_vlm_judge.py` (75-char literal, not placeholder); `/auth/key` → HTTP 200 today; usage ~$8.71, no cap set. Value never printed/copied; pilot loads it at runtime |
| Internet to providers | ALL REACHABLE | api.anthropic.com 404 (expected for `/`), openrouter.ai 200, api.openai.com 421 (expected), huggingface.co 200, kaggle.com 404-for-`/` (expected); PyPI 200 |
| Frontier models on OpenRouter | AVAILABLE | `anthropic/claude-sonnet-5` ($2/$10 per MTok), `claude-opus-5` ($5/$25), `google/gemini-2.5-pro` ($1.25/$10), `qwen/qwen3-vl-32b-instruct` (~$0.10/$0.42), `openai/gpt-5.5` ($5/$30), `x-ai/grok-4.6`, `deepseek-v4-pro` vision |
| Kaggle GPU kernels (spec only, nothing pushed) | FEASIBLE but SLOW | Free tier: T4×2 (16 GB each, ~30 h/wk) or P100; 9 h/session cap; internet toggle allows HF pulls. Qwen3-VL-32B fp16 (~65 GB) does NOT fit; 4-bit (~22 GB + video ctx) marginal on 2×T4; 8B fits. Pull ~15 GB ≈ 10–20 min. Full-OOF video inference on T4s would take a large fraction of the remaining window — strictly dominated by the API route |
| Train videos local | HAU yes / HARn NO | `hf_data_manual/HAU/*` has Depth_Color+Thermal mp4s; HARn/object rows have only skeleton/IMU (no mp4 anywhere in `hf_data_manual` or `hf_archives`) → object_interaction NOT vision-testable locally |

**Bottom line: exactly one strong-VLM route is open — OpenRouter API with the
repo's existing key. Setup time was ~15 min, not 2 h.**

## 3. Prior art (what "retired VLMs" actually scored)

- `vlm_falsification_cache.json` (87 train Q, older API-VLM zero-shot, contact sheets):
  **21/87 = 24.1%** — single 6/20, multi 4/20, combination 7/17,
  emotion 4/20, sequence **0/10**.
- Local fine-tune kernels (Kaggle, prior): Qwen2.5-VL-3B and Qwen3-VL-2B
  (LoRA) — session-bundle challengers; fusion vs structural baseline was
  net −2 and net 0 on fold audits (see `research/e2e_qwen3_session_20260909/*.summary.json`).
- `vlm_oracle_engine.py`: Qwen2.5-VL-72B on OpenRouter over all 682 test Q
  (`vlm_predictions_cache.json`) — test answers unknown, so unvalidatable; no accuracy.
- README summary: "VLM prompting … accuracy < 45%", "likelihood ranking …
  zero correlation with held-out correctness".

## 4. Pilot (new, today): claude-sonnet-5 × 30 diagnostic train questions

Protocol (`research/t2_vlm/pilot_vlm.py`, seed 20260914, temp 0):

- 6 Q each from single / multi / combination / sequence / emotion
  (object_interaction substituted by combination — no HARn video locally).
- 16-frame 4×4 Depth contact sheet per Q; category-specific prompt demanding a
  machine-readable `FINAL: <letters>` last line; multi required per-option
  PRESENT/ABSENT judgments first.
- 4/6 multi outputs initially unparseable (format drift) → repaired with a
  strict letter-only re-query (`pilot_repair_multi.py`); numbers below are
  POST-repair (favorable to the VLM). Raw outputs in `pilot_results.csv`.

Results vs chance and vs structural OOF baseline (`champ/oof_e2e_baseline_20260909.csv`, n=3689):

| Category | Pilot (sonnet-5) | Chance | Structural OOF |
|---|---|---|---|
| single | 2/6 = 33% (2/3 on rows with video; 3 HARn rows had no local video) | 25% | 1120/1147 = 97.6% |
| multi | 2/6 = 33% | ~7–15% | 687/718 = 95.7% |
| combination | 1/6 = 17% | 25% | 695/699 = 99.4% |
| sequence | **3/6 = 50%** | 4% | 105/274 = **38.3%** |
| emotion | 1/6 = 17% | 25% | 619/718 = 86.2% |
| OVERALL | 9/30 = 30% (9/27 = 33% excl. no-video) | — | 3337/3689 = 90.5% |

Cost/time measured: ~1,470 in-tokens + ~100 out-tokens per Q; 30 Q ≈ $0.12,
~8 s/Q at 3-way concurrency.

## 5. Feasibility verdict

**A category-routed VLM ensemble as a primary/challenger system: NOT viable.**
Frontier zero-shot (30%) is ~60 pp below the structural OOF baseline (90.5%)
and ~67 pp below the 97.1% champion. Four of five categories are at/below
chance; routing cannot fix that.

**One narrow hypothesis survives: sequence-only auxiliary.**
Sequence is the structural solver's worst channel (38.3% OOF) and the VLM's
best (3/6 = 50%, far above the 4% chance rate — the model genuinely extracts
temporal order from contact sheets). But n=6 is far too small to conclude
50% > 38% (95% CI ≈ 12–88%).

Recommended next step (only VLM spend justified before the deadline):

1. Sequence confirmation probe: ~40 train sequence Q (subject-disjoint,
   Depth contact sheets, same strict protocol) on ONE model
   (`claude-sonnet-5`; optionally also `qwen3-vl-32b` as a cheap second).
   Cost ≈ $0.20, wall time ≈ 10 min at 6-way concurrency.
2. Promote to fusion work ONLY if probe ≥ ~55% (i.e., beats 38% with margin);
   expected ceiling even then is small (sequence = 39/682 test Q; 38→55%
   ≈ +7 test Q ≈ +1 pp).
3. Otherwise: retire the VLM track entirely; spend remaining hours on
   structural sequence/emotion work.

### Category × model × modality matrix (conditional plan)

| Category | VLM role | Model | Modality | Rationale |
|---|---|---|---|---|
| sequence | candidate auxiliary (probe first) | claude-sonnet-5 | Depth 16-frame sheet | only VLM>chance + baseline-weak cell |
| single/multi/combination/emotion/object | NO VLM | — (structural) | — | VLM at/below chance; baseline 83–99% |

## 6. Full-run cost/time estimates (if the probe passes)

Volume: 4,087 OOF + 682 test = 4,769 Q (test-only = 682 Q).

| Model | $/MTok in/out | Full OOF+test | Test-only | Wall (6 workers, ~8 s/Q) |
|---|---|---|---|---|
| claude-sonnet-5 | 2 / 10 | ~$21 | ~$3 | ~1.8 h / ~15 min |
| gemini-2.5-pro | 1.25 / 10 | ~$13 | ~$2 | same |
| qwen3-vl-32b | 0.10 / 0.42 | ~$1 | ~$0.15 | same |
| claude-opus-5 | 5 / 25 | ~$55 | ~$8 | same |

Sequence-only full run (274 train + 39 test = 313 Q, sonnet-5): ≈ $1.40, ≈ 7 min.

## 7. Blocking facts / risks

1. No local/HF/Kaggle-GPU path can host a frontier VLM in-window; the API key
   is the single point of failure (valid today, no cap; have a backup key ready).
2. HARn videos are absent locally — object_interaction (21 test Q) and HARn
   singles cannot be vision-piloted without an HF download.
3. Winner verification requires a live reproducible run — an API solution must
   pin model IDs and archive prompts + raw responses.
4. Emotion/combination/single/multi show no VLM signal; do not burn deadline
   hours re-probing them with bigger models — the failure is depth-video
   perception + fine-grained manner distinctions, not scale (72B already failed).

## Artifacts

- `research/t2_vlm/pilot_vlm.py` — pilot (loads key at runtime, prints none).
- `research/t2_vlm/pilot_repair_multi.py` — strict letter-only repair pass.
- `research/t2_vlm/pilot_results.csv` — 30 rows with raw outputs, timings, tokens.
- `research/t2_vlm/REPORT.md` — this file.
