"""Is the gray-mush reconstruction a float32-vs-autocast(bf16) mismatch?

Reconstruct REVEALED-only frames with (a) autocast bf16 (training path) and (b) plain float32
(the viewer's path). Report MSE + pred stats and save a side-by-side montage for eyeballing.
"""
import sys
from pathlib import Path
from dataclasses import fields

import numpy as np
import torch
import cv2

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

device = "cuda"
ckpt = _SRC / "autoencoder_occluded.pt"
frames = _SRC.parent.parent / "occluded.npy"
actions = _SRC.parent.parent / "occluded_actions.npy"

payload = torch.load(ckpt, map_location=device, weights_only=False)
allowed = {f.name for f in fields(AutoEncoderConfig)}
cfg = AutoEncoderConfig(**{k: v for k, v in payload["config"].items() if k in allowed})
print("config dtype:", cfg.dtype, "| mae_max_mask:", cfg.mae_max_mask)
model = AutoEncoder(cfg).to(device).eval()
model.load_state_dict(payload["model_state_dict"])

# param dtypes
dt = {p.dtype for p in model.parameters()}
print("param dtypes present:", dt)

L = cfg.max_temporal_length
raw = np.load(frames, mmap_mode="r")
act = np.load(actions, mmap_mode="r")

# find a clip and keep only revealed frames (action==0)
rng = np.random.default_rng(3)
clips, acts = [], []
while len(clips) < 8:
    ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - L)
    a = np.asarray(act[ep, s:s + L])
    if (a == 0).sum() >= 4:
        clips.append(np.asarray(raw[ep, s:s + L])); acts.append(a)
x = torch.from_numpy(np.stack(clips).astype(np.float32) / 255.0).to(device)
a = torch.from_numpy(np.stack(acts)).to(device)
revealed = (a == 0)  # (B, L)


def recon(use_amp):
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            p = model(x)
    return p.float()


for use_amp in (True, False):
    p = recon(use_amp)
    rm = revealed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(p)
    mse_rev = ((p - x)[rm] ** 2).mean().item()
    pr = p[revealed]
    print(f"autocast={use_amp!s:5} | revealed MSE {mse_rev:.5f} | "
          f"pred[min {pr.min():.2f} max {pr.max():.2f} mean {pr.mean():.3f} std {pr.std():.3f}]")

# montage: pick first revealed frame of each clip -> rows: input | amp recon | fp32 recon
p_amp = recon(True); p_f32 = recon(False)
tiles = []
for b in range(len(clips)):
    t = int(torch.argmax(revealed[b].float()))  # first revealed frame
    def to_bgr(img):
        u8 = (img.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    col = np.vstack([to_bgr(x[b, t]), to_bgr(p_amp[b, t]), to_bgr(p_f32[b, t])])
    tiles.append(col)
montage = np.hstack(tiles)
montage = cv2.resize(montage, (montage.shape[1] * 3, montage.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
out = _SRC / "_recon_montage.png"
cv2.imwrite(str(out), montage)
print(f"saved montage (rows: input / autocast-recon / fp32-recon) -> {out}")
