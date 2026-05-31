"""Regime is innocent; isolated revealed frames train fine. Does the real CLIP structure
(16-frame clips with ~40% trivial curtain frames + static within-clip background) re-trigger
the collapse? Overfit K full clips and measure latent separation on REVEALED frames only.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

device = "cuda"
raw = np.load(_SRC.parent.parent / "occluded.npy", mmap_mode="r")
act = np.load(_SRC.parent.parent / "occluded_actions.npy", mmap_mode="r")
L = 16


def grab_clips(k, seed=3):
    rng = np.random.default_rng(seed)
    clips, acts = [], []
    for _ in range(k):
        ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - L)
        clips.append(np.asarray(raw[ep, s:s+L])); acts.append(np.asarray(act[ep, s:s+L]))
    x = torch.from_numpy(np.stack(clips).astype(np.float32) / 255.).to(device)  # (k,L,64,64,3)
    a = torch.from_numpy(np.stack(acts)).to(device)  # (k,L) 0=revealed 1=curtain
    return x, a


def run(name, k, lr=3e-4, steps=1500, drop=0.1, bf16=True, max_seconds=120):
    torch.manual_seed(0)
    x, a = grab_clips(k)
    cfg = AutoEncoderConfig(dtype=torch.float32, img_input_H=64, img_input_W=64,
                            max_temporal_length=L, mae_min_mask=0.0, mae_max_mask=0.0,
                            drop_rate=drop, att_drop_rate=drop)
    m = AutoEncoder(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    t0 = time.time(); done = 0
    for _ in range(steps):
        if bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ((m(x) - x) ** 2).mean()
        else:
            loss = ((m(x) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        done += 1
        if time.time() - t0 > max_seconds:
            break
    m.eval()
    with torch.no_grad():
        pred = m(x).float()
        z = m.encoder(x).float()  # (k,L,n_lat,btl)
    rev = (a == 0)  # (k,L)
    full_mse = ((pred - x) ** 2).mean().item()
    rev_mse = ((pred - x) ** 2).mean(dim=(2, 3, 4))[rev].mean().item()
    # latent cosine across all REVEALED frames
    zr = z[rev].reshape(rev.sum(), -1); zr = zr / (zr.norm(dim=1, keepdim=True) + 1e-6)
    n = zr.shape[0]
    cos = (zr @ zr.T)[~torch.eye(n, dtype=bool, device=device)].mean().item()
    rev_pred_std = pred[rev].reshape(rev.sum(), -1).std(1).mean().item()
    print(f"{name:22s} | steps {done:4d} | full MSE {full_mse:.5f} | revealed MSE {rev_mse:.5f} | "
          f"rev pred std/img {rev_pred_std:.4f} | revealed latent cos {cos:.3f}  (n_rev={n})", flush=True)


print(f"Overfit K full {L}-frame clips (with curtain frames), real regime, <=1500 steps / 120s each\n", flush=True)
run("K=4 clips", 4)
run("K=16 clips", 16)
run("K=32 clips", 32)
