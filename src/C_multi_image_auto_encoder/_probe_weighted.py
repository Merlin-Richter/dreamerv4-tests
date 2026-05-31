"""Escape is slow (~2200 steps) because trivial content (curtain + global-mean background)
dominates plain MSE. Test whether a content-weighted MSE (upweight pixels that deviate from the
global mean image) makes the latent-collapse escape FAST under fresh-batch training.
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
POOL = 3000
BATCH = 32
MAX_SECONDS = 300
LOG_EVERY = 200

rng = np.random.default_rng(0)
clips = np.empty((POOL, L, 64, 64, 3), np.uint8)
acts = np.empty((POOL, L), np.int64)
for i in range(POOL):
    ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - L)
    clips[i] = raw[ep, s:s+L]; acts[i] = act[ep, s:s+L]
clips_t = torch.from_numpy(clips).to(device)
acts_t = torch.from_numpy(acts).to(device)
global_mean = (clips_t.float() / 255.).mean(dim=(0, 1))   # (64,64,3)

pr_x, seen = [], 0
for i in range(POOL - 1, POOL - 600, -1):
    idx = torch.where(acts_t[i] == 0)[0]
    if len(idx):
        pr_x.append(clips_t[i, idx[0]]); seen += 1
    if seen >= 64:
        break
probe = torch.stack(pr_x).float().unsqueeze(1) / 255.


def train(alpha):
    torch.manual_seed(0)
    cfg = AutoEncoderConfig(dtype=torch.float32, img_input_H=64, img_input_W=64,
                            max_temporal_length=L, mae_min_mask=0.0, mae_max_mask=0.0,
                            drop_rate=0.1, att_drop_rate=0.1)
    m = AutoEncoder(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

    def probe_cos():
        m.eval()
        with torch.no_grad():
            z = m.encoder(probe).float().reshape(64, -1); pred = m(probe).float()
        zc = z / (z.norm(dim=1, keepdim=True) + 1e-6)
        cos = (zc @ zc.T)[~torch.eye(64, dtype=bool, device=device)].mean().item()
        m.train(); return cos, pred.reshape(64, -1).std(1).mean().item()

    print(f"\n=== alpha={alpha} (weight = 1 + {alpha}*|x - global_mean|) ===", flush=True)
    print(f"{'step':>6} | {'probe latent cos':>16} | {'probe pred std':>14}", flush=True)
    t0 = time.time(); step = 0; first_escape = None
    while time.time() - t0 < MAX_SECONDS:
        bi = torch.randint(0, POOL, (BATCH,), device=device)
        x = clips_t[bi].float() / 255.
        w = 1.0 + alpha * (x - global_mean).abs()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = m(x)
            loss = (w * (pred - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % LOG_EVERY == 0:
            cos, rps = probe_cos()
            print(f"{step:6d} | {cos:16.3f} | {rps:14.4f}", flush=True)
            if first_escape is None and cos < 0.9:
                first_escape = step
    cos, rps = probe_cos()
    print(f"FINAL step {step} | cos {cos:.3f} | pred std {rps:.4f} | first<0.9 at step {first_escape}", flush=True)


print(f"Content-weighted MSE, fresh batches, pool={POOL}, <= {MAX_SECONDS}s/run", flush=True)
train(alpha=0.0)   # control == plain MSE
train(alpha=10.0)
