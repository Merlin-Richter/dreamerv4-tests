"""The temp-fix escapes collapse in a clean overfit but the FULL training run still collapses.
Isolate which real-regime ingredient (bf16 / dropout / low-lr) re-triggers the collapse, by
adding each to the known-good 8-image overfit.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

device = "cuda"
raw = np.load(_SRC.parent.parent / "occluded.npy", mmap_mode="r")
act = np.load(_SRC.parent.parent / "occluded_actions.npy", mmap_mode="r")


def grab(n, seed=2):
    rng = np.random.default_rng(seed); fr = []
    while len(fr) < n:
        ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - 16)
        a = np.asarray(act[ep, s:s+16]); idx = np.where(a == 0)[0]
        if len(idx): fr.append(np.asarray(raw[ep, s+idx[0]]))
    return torch.from_numpy(np.stack(fr).astype(np.float32) / 255.).to(device)


N = 8
x = grab(N).unsqueeze(0)  # (1,N,64,64,3)


def run(name, lr=1e-3, steps=1500, drop=0.0, bf16=False):
    torch.manual_seed(0)
    cfg = AutoEncoderConfig(dtype=torch.float32, img_input_H=64, img_input_W=64,
                            max_temporal_length=N, mae_min_mask=0.0, mae_max_mask=0.0,
                            drop_rate=drop, att_drop_rate=drop)
    m = AutoEncoder(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    for _ in range(steps):
        if bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ((m(x) - x) ** 2).mean()
        else:
            loss = ((m(x) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(x).float()
    z = m.encoder(x).float().reshape(N, -1); zc = z / (z.norm(dim=1, keepdim=True) + 1e-6)
    cos = (zc @ zc.T)[~torch.eye(N, dtype=bool, device=device)].mean().item()
    print(f"{name:34s} | eval MSE {((pred-x)**2).mean().item():.5f} | pred std/img {pred.reshape(N,-1).std(1).mean():.4f} | latent cos {cos:.3f}")


print(f"8-image overfit; isolate real-regime ingredient that re-collapses\n")
run("A good (fp32,no-drop,lr1e-3)")
run("B +lr3e-4 (4000 steps)", lr=3e-4, steps=4000)
run("C +dropout0.1", drop=0.1)
run("D +bf16 autocast", bf16=True)
run("E real regime (bf16,drop,lr3e-4,4000)", lr=3e-4, steps=4000, drop=0.1, bf16=True)
