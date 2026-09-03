"""Frame-level frozen DINOv2 (ViT-S/14) features on the Depth video.

Research basis: the standard strong recipe for temporal action localization (ActionFormer,
TemporalMaxer, TSP) is frozen pretrained features + a light temporal head, not end-to-end
backbone training; and DINOv2 frozen features are documented to transfer to depth imagery
and to linearly separate depth structure.  Our own probe showed 34 hand-made depth
statistics carry no complementary signal, and that the residual failure is fine-grained
ACTION IDENTITY - which is exactly what an appearance representation supplies and motion
statistics cannot.

Frame alignment is exact (verified: depth_n == nskel, video frame i == global frame f0+i),
so these features drop straight into the existing localizer alongside skeleton+IMU.

Output champ/dino_frames.npz: per clip a (T, 384) float16 array of CLS tokens.
"""
import os, sys, time
import numpy as np, pandas as pd
import cv2
import torch
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
BATCH = 64
RES = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model():
    m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                       pretrained=True, trust_repo=True, verbose=False)
    return m.eval().to(DEV)


def video_path(qa_path, kind, mod='Depth'):
    if kind == 'train_hau':
        _, u, t = qa_path.split('/')
        return os.path.join(ROOT, 'hf_data_manual', 'HAU', u, t, mod, f'{mod}.mp4')
    if kind == 'train_harn':
        _, a, u, t = qa_path.split('/')
        return os.path.join(ROOT, 'hf_data_manual', 'HARn', a, u, t, mod, f'{mod}.mp4')
    return os.path.join(ROOT, 'hf_data_manual', 'large_model_track_test', qa_path,
                        mod, f'{mod}.mp4')


@torch.no_grad()
def encode_video(model, path, stride=1):
    cap = cv2.VideoCapture(path)
    mean = torch.tensor(MEAN, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=DEV).view(1, 3, 1, 1)
    feats, buf, kept = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if kept % stride:
            kept += 1
            continue
        kept += 1
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        g = cv2.resize(g, (RES, RES), interpolation=cv2.INTER_AREA)
        buf.append(g)
        if len(buf) == BATCH:
            feats.append(_flush(model, buf, mean, std)); buf = []
    if buf:
        feats.append(_flush(model, buf, mean, std))
    cap.release()
    if not feats:
        return None
    return np.concatenate(feats, 0).astype(np.float16)


def _flush(model, buf, mean, std):
    x = torch.from_numpy(np.stack(buf)).to(DEV).float().div_(255.0)
    x = x[:, None].repeat(1, 3, 1, 1)          # depth is single-channel; replicate to RGB
    x = (x - mean) / std
    return model(x).float().cpu().numpy()


def main(kinds=('train_hau', 'test'), stride=1):
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    todo = [(r.qa_path, r.kind) for r in meta.itertuples() if r.kind in kinds]
    out_p = os.path.join(ROOT, 'champ', 'dino_frames.npz')
    store = {}
    if os.path.exists(out_p):
        z = np.load(out_p)
        store = {k: z[k] for k in z.files}
        print('resuming with', len(store), 'clips already done')
    model = load_model()
    t0 = time.time(); nfr = 0
    for i, (p, kind) in enumerate(todo):
        if p in store:
            continue
        vp = video_path(p, kind)
        if not os.path.exists(vp):
            continue
        try:
            F = encode_video(model, vp, stride)
        except Exception as e:
            print('  ERR', p, e, flush=True); continue
        if F is None:
            continue
        store[p] = F
        nfr += len(F)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{len(todo)}  frames {nfr}  {nfr/max(1e-9,el):.1f} fps  '
                  f'elapsed {el/60:.1f}m', flush=True)
        if (i + 1) % 200 == 0:
            np.savez(out_p, **store)
    np.savez(out_p, **store)
    print('done:', len(store), 'clips,', nfr, 'frames encoded in %.1f min'
          % ((time.time() - t0) / 60))
    print('feature dim:', next(iter(store.values())).shape)


if __name__ == '__main__':
    main(stride=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
