#!/usr/bin/env python3
"""PCRME Stage 3 -- frozen MotionBERT feature extraction for every Stage-1 aligned phase.

Model: MotionBERT DSTformer backbone, NTU60-xsub action-finetuned checkpoint
(`walterzhu/MotionBERT`, `checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin`).
Chosen over the raw pretrain checkpoint because its config (`configs/action/MB_ft_NTU60_xsub
.yaml`) uses `num_joints: 17` on the `ntu60_hrnet` dataset variant -- i.e. it was fine-tuned on
COCO-17 2D poses (from HRNet), the SAME joint layout our own skeleton cache already uses (see
`champ/dense.py:frame_feats`, which indexes 5/6 as shoulders and 11/12 as hips -- the COCO
convention). The raw pretrain checkpoint's native task is 2D-to-3D lifting on H36M/AMASS and is
a worse match for "does this encoder represent EXECUTION STYLE", which is closer to what an
action-recognition fine-tune has already learned to preserve.

Input adaptation (audited numerically below before any bulk extraction)
-------------------------------------------------------------------------
MotionBERT's own NTU60-HRNet loader (`lib/data/dataset_action.py:ActionDataset.__init__`)
builds its input as: 2D pixel coords normalised by image size (`make_cam`) -> re-ordered
COCO->H36M (`coco2h36m`) -> concatenated with a per-joint detection confidence -> `crop_scale`
to [-1, 1] using the CLIP'S OWN bounding box. We have true 3D skeletons, not a 2D detector, so:

  * we DROP the z (depth) coordinate and use (x, y) as if it were the detector's 2D projection
    -- an orthographic projection onto the camera's image plane, which is the plane the raw
    Skeleton/predictions/*.json keypoints were almost certainly estimated in to begin with
    (a depth-camera pose estimator still outputs an x,y pixel/normalised location per joint);
  * confidence is set to a constant 1.0 for every joint (we have no per-joint detection score
    in the cache, and our clips do not report missing joints at the source);
  * `coco2h36m` is used verbatim, copied from MotionBERT's own source, not reimplemented;
  * `crop_scale` is used verbatim (self-normalising: the bounding box comes from the clip's own
    coordinates, so no external "image size" is needed and none is invented);
  * NO temporal resampling to a fixed length is applied -- MotionBERT's positional/temporal
    embeddings are sliced to the sequence's own length (`temp_embed[:, :F, :, :]`), so a phase
    segment of its natural duration is fed directly, preserving true relative timing within the
    segment. Segments longer than `maxlen=243` frames (about 24s at 10Hz) are centre-cropped;
    none of our phase segments are anywhere near that long (median duration ~2-4s).

This is an adaptation of a pretrained encoder to a foreign input distribution and is reported
as exactly that, not as a like-for-like replication of NTU60's evaluation. Stage 3's own
falsification protocol (frozen-embedding accuracy vs the fair champion baseline, plus the
mandatory shuffle-label control) is what decides whether this adaptation is usable, not this
docstring.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
CHAMP = ROOT / "champ"
MB_SRC = HERE / "motionbert_src"
sys.path.insert(0, str(CHAMP))
sys.path.insert(0, str(MB_SRC))

from lib.model.DSTformer import DSTformer  # noqa: E402
from lib.utils.utils_data import crop_scale  # noqa: E402


def coco2h36m(x):
    """Verbatim copy of MotionBERT's lib/data/dataset_action.py:coco2h36m (not reimplemented
    from scratch): COCO-17 -> H36M-17 joint reorder with 4 derived joints (root, belly, neck,
    head). Copied inline rather than imported because dataset_action.py pulls in unrelated
    heavy dataset-loading dependencies (easydict, torch Dataset plumbing) not needed here.
    Input: x (M, T, V, C).
    """
    y = np.zeros(x.shape)
    y[:, :, 0, :] = (x[:, :, 11, :] + x[:, :, 12, :]) * 0.5
    y[:, :, 1, :] = x[:, :, 12, :]
    y[:, :, 2, :] = x[:, :, 14, :]
    y[:, :, 3, :] = x[:, :, 16, :]
    y[:, :, 4, :] = x[:, :, 11, :]
    y[:, :, 5, :] = x[:, :, 13, :]
    y[:, :, 6, :] = x[:, :, 15, :]
    y[:, :, 8, :] = (x[:, :, 5, :] + x[:, :, 6, :]) * 0.5
    y[:, :, 7, :] = (y[:, :, 0, :] + y[:, :, 8, :]) * 0.5
    y[:, :, 9, :] = x[:, :, 0, :]
    y[:, :, 10, :] = (x[:, :, 1, :] + x[:, :, 2, :]) * 0.5
    y[:, :, 11, :] = x[:, :, 5, :]
    y[:, :, 12, :] = x[:, :, 7, :]
    y[:, :, 13, :] = x[:, :, 9, :]
    y[:, :, 14, :] = x[:, :, 6, :]
    y[:, :, 15, :] = x[:, :, 8, :]
    y[:, :, 16, :] = x[:, :, 10, :]
    return y

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT = HERE / "motionbert_weights" / "ntu60_xsub_best.bin"


def build_model():
    # dim_out=3 matches the checkpoint's vestigial internal 3D-regression head, which
    # get_representation() never touches (it returns pre_logits' output, before self.head).
    m = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8,
                  mlp_ratio=2, num_joints=17, maxlen=243, att_fuse=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    backbone_sd = {k[len("module.backbone."):]: v for k, v in sd.items()
                  if k.startswith("module.backbone.")}
    missing, unexpected = m.load_state_dict(backbone_sd, strict=False)
    print(f"loaded backbone: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 5):", missing[:5])
    if unexpected:
        print("  unexpected (first 5):", unexpected[:5])
    m.eval()
    return m.to(DEVICE)


def to_mb_input(K):
    """K: (T, 17, 3) COCO-order 3D skeleton -> (1, T, 17, 3) MotionBERT input (x, y, conf)."""
    xy = K[:, :, :2].astype(np.float32)                       # drop z, orthographic projection
    conf = np.ones((xy.shape[0], xy.shape[1], 1), dtype=np.float32)
    coco = np.concatenate([xy, conf], axis=-1)                # (T, 17, 3)
    coco = coco[None, :, :, :]                                 # (M=1, T, 17, 3) for coco2h36m
    h36m = coco2h36m(coco)                                     # (1, T, 17, 3)
    h36m = crop_scale(h36m[0], scale_range=[1, 1])              # (T, 17, 3) normalised to [-1,1]
    return h36m.astype(np.float32)


@torch.no_grad()
def embed_segment(model, K):
    """K: (T, 17, 3) raw 3D skeleton segment -> (T, 17, 512) per-frame per-joint representation."""
    x = to_mb_input(K)
    T = len(x)
    if T > 243:
        s = (T - 243) // 2
        x = x[s:s + 243]
    xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)[None]   # (1, T, 17, 3)
    rep = model.get_representation(xt)                                # (1, T, 17, 512)
    return rep[0].cpu().numpy()


def numeric_audit(model, n=4):
    """Print concrete before/after numbers for a handful of real segments -- required before
    any bulk extraction, per the brief. Not a label check; a data-transform sanity check."""
    SKEL = np.load(CHAMP / "skel_seq.npz")
    aligned = pd.read_csv(HERE / "aligned_phases.csv")
    print("=== numeric audit of the skeleton -> MotionBERT-input transform ===")
    shown = 0
    for r in aligned.itertuples():
        kkey, fkey = f"{r.parent_unit_dir}|K", f"{r.parent_unit_dir}|F"
        if kkey not in SKEL or fkey not in SKEL:
            continue
        F = SKEL[fkey]
        lo = int(np.searchsorted(F, r.seg_f0, side="left"))
        hi = int(np.searchsorted(F, r.seg_f1, side="right"))
        hi = max(hi, lo + 1)
        K = SKEL[kkey][lo:hi]
        if len(K) < 4:
            continue
        print(f"\n  row {r.Index}: {r.action} ({r.session}, cc={r.cc}), "
              f"raw frames={len(K)}")
        print(f"    raw skeleton x range [{K[:,:,0].min():.3f}, {K[:,:,0].max():.3f}]  "
              f"y range [{K[:,:,1].min():.3f}, {K[:,:,1].max():.3f}]  "
              f"z range [{K[:,:,2].min():.3f}, {K[:,:,2].max():.3f}]")
        x = to_mb_input(K)
        print(f"    MB input shape {x.shape}  xy range [{x[...,:2].min():.3f}, "
              f"{x[...,:2].max():.3f}]  conf range [{x[...,2].min():.3f}, {x[...,2].max():.3f}]")
        assert x[..., :2].min() >= -1.0001 and x[..., :2].max() <= 1.0001, "crop_scale bound violated"
        rep = embed_segment(model, K)
        print(f"    representation shape {rep.shape}  mean={rep.mean():.4f}  "
              f"std={rep.std():.4f}  finite={np.isfinite(rep).all()}")
        shown += 1
        if shown >= n:
            break
    print(f"\naudited {shown} segments; all in-bounds and finite: proceeding to bulk extraction.")


def main():
    model = build_model()
    numeric_audit(model)

    SKEL = np.load(CHAMP / "skel_seq.npz")
    aligned = pd.read_csv(HERE / "aligned_phases.csv")
    pooled = {}          # row index -> (dim_rep,) global mean-pooled embedding
    per_joint = {}        # row index -> (17, dim_rep) joint-local, time-pooled embedding
    n_ok = 0
    for i, r in enumerate(aligned.itertuples()):
        kkey, fkey = f"{r.parent_unit_dir}|K", f"{r.parent_unit_dir}|F"
        if kkey not in SKEL or fkey not in SKEL:
            continue
        F = SKEL[fkey]
        lo = int(np.searchsorted(F, r.seg_f0, side="left"))
        hi = int(np.searchsorted(F, r.seg_f1, side="right"))
        hi = max(hi, lo + 1)
        K = SKEL[kkey][lo:hi]
        if len(K) < 4:
            continue
        rep = embed_segment(model, K)              # (T, 17, 512)
        pooled[r.Index] = rep.mean(axis=(0, 1))     # global: mean over time AND joints
        per_joint[r.Index] = rep.mean(axis=0)       # joint-local: mean over time only, (17,512)
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(aligned)} rows scanned, {n_ok} embedded", flush=True)

    print(f"\nembedded {n_ok} of {len(aligned)} aligned phase rows")
    idx = np.array(sorted(pooled.keys()))
    global_mat = np.stack([pooled[i] for i in idx])
    joint_mat = np.stack([per_joint[i] for i in idx])
    np.savez_compressed(HERE / "motionbert_embeddings.npz",
                        index=idx, global_pooled=global_mat, joint_pooled=joint_mat)
    print(f"wrote {HERE / 'motionbert_embeddings.npz'}  "
          f"global={global_mat.shape}  joint={joint_mat.shape}")


if __name__ == "__main__":
    main()
