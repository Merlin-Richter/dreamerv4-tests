"""K=32 clips collapses at lr3e-4/log4. Does escape-at-scale need more steps, higher lr, or a
sharper attention-temperature init? Each variant capped by wall-clock so the script terminates.
"""
import sys
import math
import time
from pathlib import Path
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import video_auto_encoder as vae
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

device = "cuda"
raw = np.load(_SRC.parent.parent / "occluded.npy", mmap_mode="r")
act = np.load(_SRC.parent.parent / "occluded_actions.npy", mmap_mode="r")
L = 16


def grab_clips(k, seed=3):
    rng = np.random.default_rng(seed); clips, acts = [], []
    for _ in range(k):
        ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - L)
        clips.append(np.asarray(raw[ep, s:s+L])); acts.append(np.asarray(act[ep, s:s+L]))
    x = torch.from_numpy(np.stack(clips).astype(np.float32) / 255.).to(device)
    a = torch.from_numpy(np.stack(acts)).to(device)
    return x, a


def run(name, k=32, lr=3e-4, drop=0.1, init_logscale=math.log(4.0), max_seconds=240, max_steps=6000):
    torch.manual_seed(0)
    x, a = grab_clips(k)
    cfg = AutoEncoderConfig(dtype=torch.float32, img_input_H=64, img_input_W=64,
                            max_temporal_length=L, mae_min_mask=0.0, mae_max_mask=0.0,
                            drop_rate=drop, att_drop_rate=drop)
    m = AutoEncoder(cfg).to(device)
    for mod in m.modules():
        if isinstance(mod, vae.Attention):
            mod.logit_scale.data.fill_(init_logscale)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    t0 = time.time(); done = 0
    for _ in range(max_steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((m(x) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        done += 1
        if time.time() - t0 > max_seconds:
            break
    m.eval()
    with torch.no_grad():
        pred = m(x).float(); z = m.encoder(x).float()
    rev = (a == 0)
    rev_mse = ((pred - x) ** 2).mean(dim=(2, 3, 4))[rev].mean().item()
    zr = z[rev].reshape(rev.sum(), -1); zr = zr / (zr.norm(dim=1, keepdim=True) + 1e-6)
    n = zr.shape[0]
    cos = (zr @ zr.T)[~torch.eye(n, dtype=bool, device=device)].mean().item()
    rps = pred[rev].reshape(rev.sum(), -1).std(1).mean().item()
    print(f"{name:34s} | steps {done:4d} | revealed MSE {rev_mse:.5f} | rev pred std {rps:.4f} | latent cos {cos:.3f}", flush=True)


print("K=32 clips: what enables escape at scale? (<=240s / 6000 steps each)\n", flush=True)
run("baseline log4, lr3e-4")
run("log4, lr1e-3", lr=1e-3)
run("log8 init, lr3e-4", init_logscale=math.log(8.0))
run("log16 init, lr1e-3", init_logscale=math.log(16.0), lr=1e-3)
