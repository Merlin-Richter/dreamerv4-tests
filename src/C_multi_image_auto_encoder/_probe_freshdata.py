"""Decisive test: with FRESH batches each step (mimicking real training, low per-example
repetition), does the latent collapse eventually escape given enough steps, or never?
Pre-loads a large clip pool on GPU (uint8), samples a fresh batch/step, logs latent cos on a
FIXED probe set of revealed frames periodically. Wall-clock capped so it terminates.
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
POOL = 10000
BATCH = 32
MAX_SECONDS = 720
LOG_EVERY = 200

rng = np.random.default_rng(0)
clips = np.empty((POOL, L, 64, 64, 3), np.uint8)
acts = np.empty((POOL, L), np.int64)
for i in range(POOL):
    ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - L)
    clips[i] = raw[ep, s:s+L]; acts[i] = act[ep, s:s+L]
clips_t = torch.from_numpy(clips).to(device)          # uint8 (POOL,L,64,64,3)
acts_t = torch.from_numpy(acts).to(device)

# fixed probe: 64 revealed frames from the pool's last 500 clips (as single-frame clips)
pr_x, seen = [], 0
for i in range(POOL - 1, POOL - 600, -1):
    idx = torch.where(acts_t[i] == 0)[0]
    if len(idx):
        pr_x.append(clips_t[i, idx[0]]); seen += 1
    if seen >= 64:
        break
probe = torch.stack(pr_x).float().unsqueeze(1) / 255.   # (64,1,64,64,3)

torch.manual_seed(0)
cfg = AutoEncoderConfig(dtype=torch.float32, img_input_H=64, img_input_W=64,
                        max_temporal_length=L, mae_min_mask=0.0, mae_max_mask=0.0,
                        drop_rate=0.1, att_drop_rate=0.1)
m = AutoEncoder(cfg).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)


def probe_cos():
    m.eval()
    with torch.no_grad():
        z = m.encoder(probe).float().reshape(64, -1)
        pred = m(probe).float()
    zc = z / (z.norm(dim=1, keepdim=True) + 1e-6)
    cos = (zc @ zc.T)[~torch.eye(64, dtype=bool, device=device)].mean().item()
    rps = pred.reshape(64, -1).std(1).mean().item()
    m.train()
    return cos, rps


print(f"FRESH-batch training: pool={POOL} clips, batch={BATCH}, lr3e-4, log4, <= {MAX_SECONDS}s\n", flush=True)
print(f"{'step':>6} | {'train_mse':>9} | {'probe latent cos':>16} | {'probe pred std':>14}", flush=True)
t0 = time.time(); step = 0
while time.time() - t0 < MAX_SECONDS:
    bi = torch.randint(0, POOL, (BATCH,), device=device)
    x = clips_t[bi].float() / 255.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = ((m(x) - x) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    step += 1
    if step % LOG_EVERY == 0:
        cos, rps = probe_cos()
        print(f"{step:6d} | {loss.item():9.5f} | {cos:16.3f} | {rps:14.4f}", flush=True)
cos, rps = probe_cos()
print(f"\nFINAL step {step} | probe latent cos {cos:.3f} | probe pred std {rps:.4f}", flush=True)
print("escaped" if cos < 0.9 else "COLLAPSED", flush=True)
