"""Is the gray-mush a bottleneck collapse (all images -> same latents) or a dead decoder?"""
import sys
from pathlib import Path
from dataclasses import fields

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

device = "cuda"
payload = torch.load(_SRC / "autoencoder_occluded.pt", map_location=device, weights_only=False)
allowed = {f.name for f in fields(AutoEncoderConfig)}
cfg = AutoEncoderConfig(**{k: v for k, v in payload["config"].items() if k in allowed})
model = AutoEncoder(cfg).to(device).eval()
model.load_state_dict(payload["model_state_dict"])

raw = np.load(_SRC.parent.parent / "occluded.npy", mmap_mode="r")
act = np.load(_SRC.parent.parent / "occluded_actions.npy", mmap_mode="r")
L = cfg.max_temporal_length
rng = np.random.default_rng(1)

# gather revealed frames from many episodes
rev_frames = []
while len(rev_frames) < 32:
    ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - L)
    a = np.asarray(act[ep, s:s+L])
    idx = np.where(a == 0)[0]
    if len(idx):
        rev_frames.append(np.asarray(raw[ep, s+idx[0]]))
x = torch.from_numpy(np.stack(rev_frames).astype(np.float32)/255.0).unsqueeze(1).to(device)  # (32,1,64,64,3)

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.encoder(x).float()        # (32,1,n_lat,btl)
        out = model.decoder(z.to(torch.bfloat16)).float()

z_flat = z.reshape(z.shape[0], -1)          # (32, n_lat*btl)
print("latents per image:", tuple(z.shape[1:]))
print(f"latent value range [{z.min():.3f}, {z.max():.3f}]  (Tanh-bounded)")
print(f"latent std ACROSS images (mean over dims): {z_flat.std(0).mean():.4f}")
print(f"latent std WITHIN a latent vector       : {z_flat.std(1).mean():.4f}")
# pairwise cosine similarity of latents across images: ~1 => collapsed
zc = z_flat / (z_flat.norm(dim=1, keepdim=True) + 1e-6)
cos = (zc @ zc.T)
off = cos[~torch.eye(len(zc), dtype=bool, device=device)]
print(f"pairwise cosine sim across images: mean {off.mean():.3f}  (->1.0 == bottleneck collapse)")
print(f"decoder output std per image (mean): {out.reshape(32,-1).std(1).mean():.4f}")

# Decoder sensitivity: does output change if we perturb the latents hard?
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z2 = torch.tanh(torch.randn_like(z) * 3)  # very different valid latents
        out2 = model.decoder(z2.to(torch.bfloat16)).float()
diff = (out2 - out).abs().mean().item()
print(f"output change when latents fully randomized: {diff:.4f}  (~0 == decoder ignores latents)")
