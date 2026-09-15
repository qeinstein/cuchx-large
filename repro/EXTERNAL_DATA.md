# External data and artifacts (Team Fluxx)

## Pretrained backbones (features frozen; weights NOT needed at inference)

No weight file is loaded by the inference path. Both backbones below were used
only to precompute features, which are committed as per-clip caches; the
solver reads the caches, never the weights. Reproduction therefore does not
depend on any checkpoint download, latest or otherwise.

| Backbone | Exact version | Used for | At inference? |
|---|---|---|---|
| DINOv2 ViT-S/14 | `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)` (Meta, CC-BY-NC / Apache-2.0 code) | research embeddings + historical `harn_dino` signal | **Not loaded**: `CHAMP_DENSE_DINO=0` default skips `dino_frames.npz`; `champ/harn_dino.npz` absent → fusion branch skipped |
| MobileNetV3-Small | torchvision pretrained, weights id `mobilenet_v3_small-047dcff4` (hash-pinned filename, BSD) | `champ/depth_mnv3_frames.npz` (built, then unused) | **Not loaded**: no solver code path reads it; file excluded from package |

For regeneration (not required): DINOv2 via torch.hub as above (hub snapshot
as of Sep 2026); MobileNetV3-Small via
`torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')`, whose
checkpoint filename embeds hash `047dcff4`.

## Training data

- Official challenge data only. No external training data of any kind.
- Committed frozen artifacts derived from official data (required: organizers
  mount Testing data only, so training features must ship with the package):
  `training_qa.csv` (700KB index+labels), `champ/meta.csv`,
  `champ/feats.csv`, `champ/skel_seq.npz` (50MB),
  `champ/dense_logits.npz` (24MB), `champ/harn_clf.npz` (1.2MB),
  `champ/vocab.json`. Regenerable via `champ/build_*.py` (needs official bulk
  data + pose/IMU toolchain; not part of the inference path).
- Explicitly excluded (verified unused: byte-identical output with the files
  removed): `champ/imu_seq.npz` (IMU enters via `feats.csv` stats and baked-in
  committed logits instead) and `champ/depth_mnv3_frames.npz` (no solver code
  path reads it).
- `test_qa.csv` is committed AND read from `<data_dir>` at run time
  (`<data_dir>` wins when present). Bulk official test videos/sensors are not
  included and not needed (features keyed by clip id).
