# Direct-Depth missing-LMT ablation

This run uses raw `Depth/Depth.mp4` only. It does not use LMT, skeleton, IMU, radar, IR, Depth_Color, `champ/meta.csv`, parent-frame alignment, or existing feature caches.

- Scope: `all`; train descriptors: 3098; action classes: 44.
- Real target: 13 no-LMT clips / 15 questions ({'single': np.int64(10), 'object_interaction': np.int64(5)}).
- Comparator: held-out `oof_v8_final.csv` predictions, the same v8 vector used by `champ/final.py` for real no-evidence fallbacks.
- This directory contains diagnostics only; no Kaggle submission was created.

## Pooled and per-fold result

```text
                 slice   n  video_correct  video_accuracy  v8_correct  v8_accuracy  disagreements  w_to_r  r_to_w  neutral_disagreements  flip_precision  net  net_per_100_flips
all HARn single+object 562            447        0.795374       446.0     0.793594          146.0    65.0    64.0                   17.0        0.503876  1.0           0.684932
 test-style actionable 447            355        0.794183       343.0     0.767338          120.0    59.0    47.0                   14.0        0.556604 12.0          10.000000
    object_interaction 133            117        0.879699       117.0     0.879699           17.0     8.0     8.0                    1.0        0.500000  0.0           0.000000
                single 429            330        0.769231       329.0     0.766900          129.0    57.0    56.0                   16.0        0.504425  1.0           0.775194
                fold_0 128            100        0.781250        98.0     0.765625           38.0    16.0    14.0                    8.0        0.533333  2.0           5.263158
                fold_1 110             88        0.800000        83.0     0.754545           29.0    15.0    10.0                    4.0        0.600000  5.0          17.241379
                fold_2 154            120        0.779221       122.0     0.792208           38.0    16.0    18.0                    4.0        0.470588 -2.0          -5.263158
                fold_3  82             73        0.890244        72.0     0.878049           15.0     8.0     7.0                    0.0        0.533333  1.0           6.666667
                fold_4  88             66        0.750000        71.0     0.806818           26.0    10.0    15.0                    1.0        0.400000 -5.0         -19.230769
           action_top1 562            156        0.277580         NaN          NaN            NaN     NaN     NaN                    NaN             NaN  NaN                NaN
```

## Fixed confidence-gate audit

```text
                                  slice   n  video_correct  video_accuracy  v8_correct  v8_accuracy  disagreements  w_to_r  r_to_w  neutral_disagreements  flip_precision  net  net_per_100_flips  gate             selection  proposed_overrides
                  all; confidence>=0.00 562            447        0.795374         446     0.793594            146      65      64                     17        0.503876    1           0.684932   0.0                   all                 146
test-style actionable; confidence>=0.00 447            355        0.794183         343     0.767338            120      59      47                     14        0.556604   12          10.000000   0.0 test-style actionable                 120
                  all; confidence>=0.70 562            458        0.814947         446     0.793594            108      55      43                     10        0.561224   12          11.111111   0.7                   all                 108
test-style actionable; confidence>=0.70 447            364        0.814318         343     0.767338             92      53      32                      7        0.623529   21          22.826087   0.7 test-style actionable                  92
                  all; confidence>=0.80 562            460        0.818505         446     0.793594             98      51      37                     10        0.579545   14          14.285714   0.8                   all                  98
test-style actionable; confidence>=0.80 447            364        0.814318         343     0.767338             84      49      28                      7        0.636364   21          25.000000   0.8 test-style actionable                  84
                  all; confidence>=0.90 562            461        0.820285         446     0.793594             85      46      31                      8        0.597403   15          17.647059   0.9                   all                  85
test-style actionable; confidence>=0.90 447            363        0.812081         343     0.767338             74      44      24                      6        0.647059   20          27.027027   0.9 test-style actionable                  74
```

## Limits

- This is subject-disjoint, but the real 13 missing-LMT clips have unknown action mix; the training ablation cannot guarantee a matched hidden-label distribution.
- The v8 comparison is exact as a fallback vector, but v8 itself was trained before this ablation and is not a video-only model.
- Any test diagnostic must remain non-production unless a predeclared gate shows at least 0.80 W-to-R / (W-to-R + R-to-W) precision with enough support and fold consistency.

## Verdict: KILLED (2026-09-04)

The best pre-registered gate (`confidence>=0.90`, `test-style actionable`) reaches
**0.647 flip precision** on 74 disagreements (44 W→R / 24 R→W, net +20). That is a real,
positive, non-trivial signal -- the direct-Depth specialist is a legitimate improvement over
v8 in aggregate (0.795 vs 0.794 on the full ablation cohort, and clears v8 at every gate) --
but it falls short of the pre-declared **0.80** minimum by a wide margin, and even lowering
the bar is not obviously safe: the *ungated* per-fold breakdown has two of five folds net
negative (fold_2 -2, fold_4 -5) before any confidence filtering is applied, so the aggregate
positive is not fold-consistent at the level this repository requires for a production
override (see `FINDINGS_session2_residual_surface.md`'s established noise floor and the
mechanism-U precedent: a good-looking pooled number with fold failures underneath does not
survive contact with the real 13-clip, 15-question target, whose action mix is itself unknown
and cannot be assumed to match the training distribution).

No override is proposed for the 15 real no-LMT questions. The champion's existing
`harn_no_evidence` -> v8 fallback remains in place. This closes out the secondary objective
for this session; the `depth_dino_direct_all.npz` cache and `run_ablation.py` are preserved so
a future session does not need to re-extract if a different comparator or gate is worth
testing later.
