# Direct-Depth missing-LMT ablation

This run uses raw `Depth/Depth.mp4` only. It does not use LMT, skeleton, IMU, radar, IR, Depth_Color, `champ/meta.csv`, parent-frame alignment, or existing feature caches.

- Scope: `qa`; train descriptors: 524; action classes: 44.
- Real target: 13 no-LMT clips / 15 questions ({'single': np.int64(10), 'object_interaction': np.int64(5)}).
- Comparator: held-out `oof_v8_final.csv` predictions, the same v8 vector used by `champ/final.py` for real no-evidence fallbacks.
- This directory contains diagnostics only; no Kaggle submission was created.

## Pooled and per-fold result

```text
                 slice   n  video_correct  video_accuracy  v8_correct  v8_accuracy  disagreements  w_to_r  r_to_w  neutral_disagreements  flip_precision   net  net_per_100_flips
all HARn single+object 562            406        0.722420       446.0     0.793594          175.0    57.0    97.0                   21.0        0.370130 -40.0         -22.857143
    object_interaction 133            115        0.864662       117.0     0.879699           13.0     5.0     7.0                    1.0        0.416667  -2.0         -15.384615
                single 429            291        0.678322       329.0     0.766900          162.0    52.0    90.0                   20.0        0.366197 -38.0         -23.456790
                fold_0 128             82        0.640625        98.0     0.765625           49.0    13.0    29.0                    7.0        0.309524 -16.0         -32.653061
                fold_1 110             86        0.781818        83.0     0.754545           31.0    14.0    11.0                    6.0        0.560000   3.0           9.677419
                fold_2 154            111        0.720779       122.0     0.792208           50.0    17.0    28.0                    5.0        0.377778 -11.0         -22.000000
                fold_3  82             66        0.804878        72.0     0.878049           20.0     7.0    13.0                    0.0        0.350000  -6.0         -30.000000
                fold_4  88             61        0.693182        71.0     0.806818           25.0     6.0    16.0                    3.0        0.272727 -10.0         -40.000000
           action_top1 562            122        0.217082         NaN          NaN            NaN     NaN     NaN                    NaN             NaN   NaN                NaN
```

## Fixed confidence-gate audit

```text
           slice   n  video_correct  video_accuracy  v8_correct  v8_accuracy  disagreements  w_to_r  r_to_w  neutral_disagreements  flip_precision  net  net_per_100_flips  gate  proposed_overrides
confidence>=0.00 562            406        0.722420         446     0.793594            175      57      97                     21        0.370130  -40         -22.857143   0.0                 175
confidence>=0.70 562            443        0.788256         446     0.793594             97      41      44                     12        0.482353   -3          -3.092784   0.7                  97
confidence>=0.80 562            450        0.800712         446     0.793594             77      36      32                      9        0.529412    4           5.194805   0.8                  77
confidence>=0.90 562            454        0.807829         446     0.793594             51      27      19                      5        0.586957    8          15.686275   0.9                  51
```

## Limits

- This is subject-disjoint, but the real 13 missing-LMT clips have unknown action mix; the training ablation cannot guarantee a matched hidden-label distribution.
- The v8 comparison is exact as a fallback vector, but v8 itself was trained before this ablation and is not a video-only model.
- Any test diagnostic must remain non-production unless a predeclared gate shows at least 0.80 W-to-R / (W-to-R + R-to-W) precision with enough support and fold consistency.
