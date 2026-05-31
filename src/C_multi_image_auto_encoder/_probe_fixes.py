"""Test candidate fixes for latent collapse by hard-overfitting 8 distinct dense images.

Hypothesis: latent cross-attention is near-uniform (tiny logits from unit-norm q/k + small
scale) -> mean pooling -> image-invariant latents. Test sharpening + capacity variants.
A fix should drive eval MSE toward 0 and latent cos well below 1.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import video_auto_encoder as vae
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

device = "cuda"
raw = np.load(_SRC.parent.parent / "occluded.npy", mmap_mode="r")
act = np.load(_SRC.parent.parent / "occluded_actions.npy", mmap_mode="r")


def grab(n, seed=2):
    rng = np.random.default_rng(seed)
    fr = []
    while len(fr) < n:
        ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - 16)
        a = np.asarray(act[ep, s:s+16]); idx = np.where(a == 0)[0]
        if len(idx):
            fr.append(np.asarray(raw[ep, s+idx[0]]))
    return torch.from_numpy(np.stack(fr).astype(np.float32) / 255.).to(device)


N = 8
x = grab(N).unsqueeze(0)  # (1,N,64,64,3) one clip, N distinct frames


def run(name, soft_cap=50.0, kill_qknorm=False, n_latents=4, bottleneck=64, scale_mult=1.0):
    torch.manual_seed(0)
    cfg = AutoEncoderConfig(dtype=torch.float32, img_input_H=64, img_input_W=64,
                            max_temporal_length=N, mae_min_mask=0.0, mae_max_mask=0.0,
                            drop_rate=0.0, att_drop_rate=0.0,
                            att_logit_soft_cap=soft_cap, n_latents=n_latents,
                            bottleneck_dim=bottleneck)
    m = AutoEncoder(cfg).to(device)
    if kill_qknorm or scale_mult != 1.0:
        for mod in m.modules():
            if isinstance(mod, vae.Attention):
                if kill_qknorm:
                    mod.q_norm = nn.Identity(); mod.k_norm = nn.Identity()
                mod.scale = mod.scale * scale_mult
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m.train()
    for _ in range(1500):
        loss = ((m(x) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(x).float()
    emse = ((pred - x) ** 2).mean().item()
    z = m.encoder(x).float().reshape(N, -1)
    zc = z / (z.norm(dim=1, keepdim=True) + 1e-6)
    cos = (zc @ zc.T)[~torch.eye(N, dtype=bool, device=device)].mean().item()
    print(f"{name:24s} | eval MSE {emse:.5f} | pred std/img {pred.reshape(N,-1).std(1).mean():.4f} | latent cos {cos:.3f}")


print(f"Overfit ONE clip of {N} distinct dense frames, 1500 steps, lr 1e-3\n")
run("baseline")
run("kill_qknorm", kill_qknorm=True)
run("sharper_scale x4", scale_mult=4.0)
run("kill_qknorm + scale x4", kill_qknorm=True, scale_mult=4.0)
run("big_lat16_btl256", n_latents=16, bottleneck=256)
run("big + kill_qknorm", n_latents=16, bottleneck=256, kill_qknorm=True)
