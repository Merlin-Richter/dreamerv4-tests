"""Why does C_ collapse to constant-gray on the dense occluded data?

Train the C_ AutoEncoder on a dense subset under different regimes and look at whether the
revealed-frame reconstruction develops spatial structure (output std) or stays uniform.
Saves a montage per config (rows: input / reconstruction).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

DEVICE = "cuda"
DATA = _SRC.parent.parent / "occluded.npy"
ACTS = _SRC.parent.parent / "occluded_actions.npy"
N_EPISODES = 400


class FrameDS(Dataset):
    """Single revealed frames as L=1 clips (B_-style independent images)."""
    def __init__(self, frames):  # frames: (M,64,64,3) float
        self.f = frames
    def __len__(self): return len(self.f)
    def __getitem__(self, i): return self.f[i].unsqueeze(0)  # (1,64,64,3)


class ClipDS(Dataset):
    def __init__(self, eps, L):
        self.eps = eps; self.L = L
        self.pairs = [(e, j*L) for e in range(eps.shape[0]) for j in range(eps.shape[1]//L)]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        e, s = self.pairs[i]; return self.eps[e, s:s+self.L]


def montage(model, val_x, name, use_amp):
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(val_x).float()
    std = pred.std().item()
    mse = ((pred - val_x) ** 2).mean().item()
    def bgr(img):
        u8 = (img.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    cols = [np.vstack([bgr(val_x[b,0]), bgr(pred[b,0])]) for b in range(min(8, val_x.shape[0]))]
    m = cv2.resize(np.hstack(cols), (8*64*3, 2*64*3), interpolation=cv2.INTER_NEAREST)
    out = _SRC / f"_dense_{name}.png"; cv2.imwrite(str(out), m)
    print(f"{name:24s} | recon MSE {mse:.5f} | pred std {std:.4f} | -> {out.name}")


def run(name, dtype, L, bs, use_amp, epochs, frames_eps, val_x):
    torch.manual_seed(0)
    cfg = AutoEncoderConfig(dtype=dtype, img_input_H=64, img_input_W=64, max_temporal_length=max(L,1),
                            mae_min_mask=0.0, mae_max_mask=0.0)
    model = AutoEncoder(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if L == 1:
        flat = frames_eps.reshape(-1, 64, 64, 3)
        ds = FrameDS(flat)
    else:
        ds = ClipDS(frames_eps, L)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(epochs):
        model.train()
        for b in loader:
            b = b.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = ((model(b) - b) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    montage(model, val_x, name, use_amp)


if __name__ == "__main__":
    raw = np.load(DATA, mmap_mode="r"); act = np.load(ACTS, mmap_mode="r")
    fr = np.asarray(raw[:N_EPISODES]).astype(np.float32)/255.0
    ac = np.asarray(act[:N_EPISODES])
    x = torch.from_numpy(fr)
    # train on ALL frames (incl curtain) like the real run; val = revealed frames only
    train_eps = x[:N_EPISODES-20]
    val_pool = x[N_EPISODES-20:]
    val_ac = ac[N_EPISODES-20:]
    # build val batch of revealed single frames
    rev = np.argwhere(val_ac == 0)
    sel = rev[np.random.default_rng(0).choice(len(rev), 8, replace=False)]
    val_x = torch.stack([val_pool[e, t] for e, t in sel]).unsqueeze(1).to(DEVICE)  # (8,1,64,64,3)

    print(f"{N_EPISODES} eps subset\n")
    run("A_clips_bf16", torch.bfloat16, L=16, bs=64,  use_amp=True, epochs=10, frames_eps=train_eps, val_x=val_x)
    run("B_frames_bf16", torch.bfloat16, L=1, bs=256, use_amp=True, epochs=6, frames_eps=train_eps, val_x=val_x)
    run("C_frames_fp32", torch.float32, L=1, bs=256, use_amp=True, epochs=6, frames_eps=train_eps, val_x=val_x)
