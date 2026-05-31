"""Can C_ reconstruct dense frames at all, and is the Tanh bottleneck the collapse cause?

Overfit 4 dense revealed frames hard under variants and report per-image output std (uniform
== still collapsed) + save montages.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import video_auto_encoder as vae
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

device = "cuda"
raw = np.load(_SRC.parent.parent / "occluded.npy", mmap_mode="r")
act = np.load(_SRC.parent.parent / "occluded_actions.npy", mmap_mode="r")

rng = np.random.default_rng(2)
frames = []
while len(frames) < 4:
    ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1]-16)
    a = np.asarray(act[ep, s:s+16]); idx = np.where(a == 0)[0]
    if len(idx): frames.append(np.asarray(raw[ep, s+idx[0]]))
x = torch.from_numpy(np.stack(frames).astype(np.float32)/255.0).unsqueeze(1).to(device)  # (4,1,64,64,3)


def bgr(img):
    u8 = (img.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


def run(name, bottleneck):
    torch.manual_seed(0)
    cfg = AutoEncoderConfig(dtype=torch.float32, img_input_H=64, img_input_W=64,
                            max_temporal_length=1, mae_min_mask=0.0, mae_max_mask=0.0,
                            drop_rate=0.0, att_drop_rate=0.0)
    model = AutoEncoder(cfg).to(device)
    if bottleneck == "linear":
        model.encoder.act = nn.Identity()
    elif bottleneck == "tanh_smallinit":
        nn.init.normal_(model.encoder.bottleneck_proj.weight, std=0.02)
        nn.init.zeros_(model.encoder.bottleneck_proj.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1000):
        loss = ((model(x) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x).float()
    z = model.encoder(x).float().reshape(4, -1)
    zc = z / (z.norm(dim=1, keepdim=True)+1e-6)
    cos = (zc@zc.T)[~torch.eye(4, dtype=bool, device=device)].mean().item()
    print(f"{name:18s} | final MSE {loss.item():.5f} | pred std/img {pred.reshape(4,-1).std(1).mean():.4f} "
          f"| latent cos across imgs {cos:.3f} | z range [{z.min():.2f},{z.max():.2f}]")
    m = np.hstack([np.vstack([bgr(x[b,0]), bgr(pred[b,0])]) for b in range(4)])
    cv2.imwrite(str(_SRC / f"_ovf_{name}.png"), cv2.resize(m,(4*64*3,2*64*3),interpolation=cv2.INTER_NEAREST))


print("Overfit 4 dense frames, 1000 steps, lr 1e-3\n")
run("asis_tanh", "tanh")
run("linear", "linear")
run("tanh_smallinit", "tanh_smallinit")
