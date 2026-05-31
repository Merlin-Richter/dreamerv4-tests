"""Localize WHERE per-image signal dies in the encoder.

Feed N distinct dense images through a FRESH (untrained) encoder and, after each block,
measure the cross-image variation of the latent token representations:
  - std across images (per dim, averaged) -- 0 == identical across images
  - mean pairwise cosine of the flattened latent block (->1.0 == collapsed)
Also do the same for the patch tokens, to see if patches keep per-image info while latents lose it.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from video_auto_encoder import Encoder, AutoEncoderConfig

device = "cuda"
raw = np.load(_SRC.parent.parent / "occluded.npy", mmap_mode="r")
act = np.load(_SRC.parent.parent / "occluded_actions.npy", mmap_mode="r")
rng = np.random.default_rng(2)

N = 6
frames = []
while len(frames) < N:
    ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - 16)
    a = np.asarray(act[ep, s:s+16]); idx = np.where(a == 0)[0]
    if len(idx):
        frames.append(np.asarray(raw[ep, s+idx[0]]))
x = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.).unsqueeze(1).to(device)  # (N,1,64,64,3)
print(f"input cross-image std (pixels): {x.reshape(N,-1).std(0).mean():.4f}\n")

torch.manual_seed(0)
cfg = AutoEncoderConfig(dtype=torch.float32, img_input_H=64, img_input_W=64,
                        max_temporal_length=1, mae_min_mask=0.0, mae_max_mask=0.0,
                        drop_rate=0.0, att_drop_rate=0.0)
enc = Encoder(cfg).to(device).eval()
n_lat = cfg.n_latents


def stats(tok):  # tok: (N,1,k,C) -> cross-image std, mean pairwise cosine
    f = tok.reshape(N, -1).float()
    s = f.std(0).mean().item()
    fc = f / (f.norm(dim=1, keepdim=True) + 1e-6)
    cos = (fc @ fc.T)[~torch.eye(N, dtype=bool, device=device)].mean().item()
    return s, cos


with torch.no_grad():
    xp = enc.patchify(x)
    xp = enc.patch_proj(xp)
    xp = xp + enc.learned_position_embedding
    latents = enc.learned_latents.unsqueeze(0).unsqueeze(0).expand(N, 1, -1, -1)
    h = torch.concat((xp, latents), dim=2)

    ps, pc = stats(h[:, :, :-n_lat])
    ls, lc = stats(h[:, :, -n_lat:])
    print(f"{'after patch_proj+pos':28s} | patch std {ps:.4f} cos {pc:.3f} | latent std {ls:.4f} cos {lc:.3f}")

    for i, block in enumerate(enc.blocks):
        is_temporal = (i + 1) % 4 == 0
        h = block(h)
        ps, pc = stats(h[:, :, :-n_lat])
        ls, lc = stats(h[:, :, -n_lat:])
        tag = f"block{i} {'(temporal)' if is_temporal else '(spatial) '}"
        print(f"{tag:28s} | patch std {ps:.4f} cos {pc:.3f} | latent std {ls:.4f} cos {lc:.3f}")

    z = enc.norm(h[:, :, -n_lat:, :])
    zs, zc = stats(z)
    print(f"{'after norm':28s} | latent std {zs:.4f} cos {zc:.3f}")
    z = enc.act(enc.bottleneck_proj(z))
    zs, zc = stats(z)
    print(f"{'after bottleneck+tanh':28s} | latent std {zs:.4f} cos {zc:.3f}")
