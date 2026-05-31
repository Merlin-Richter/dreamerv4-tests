"""Quantify whether a tokenizer checkpoint escaped the all-black collapse.

Loads a checkpoint + the dataset, reconstructs a batch of clips, and reports reconstruction
MSE vs the all-black baseline plus the recovery on bright (ball) pixels.

    ../../venv/Scripts/python.exe _probe_recon.py --frames ../../occluded.npy --checkpoint autoencoder_occluded.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from video_auto_encoder import AutoEncoder, AutoEncoderConfig
from dataclasses import fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=_SRC.parent.parent / "occluded.npy")
    ap.add_argument("--checkpoint", type=Path, default=_SRC / "autoencoder_occluded.pt")
    ap.add_argument("--n-clips", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    allowed = {f.name for f in fields(AutoEncoderConfig)}
    cfg = AutoEncoderConfig(**{k: v for k, v in payload["config"].items() if k in allowed})
    model = AutoEncoder(cfg).to(device).eval()
    model.load_state_dict(payload["model_state_dict"])

    L = cfg.max_temporal_length
    raw = np.load(args.frames, mmap_mode="r")
    rng = np.random.default_rng(0)
    clips = []
    for _ in range(args.n_clips):
        ep = rng.integers(raw.shape[0]); s = rng.integers(0, raw.shape[1] - L)
        clips.append(np.asarray(raw[ep, s:s + L]))
    x = torch.from_numpy(np.stack(clips).astype(np.float32) / 255.0).to(device)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred = model(x).float()

    mse = ((pred - x) ** 2).mean().item()
    black = (x ** 2).mean().item()
    bright = (x.max(-1).values > 0.3)
    bm = bright.unsqueeze(-1).expand_as(x)
    cm = ((pred - x)[bm] ** 2).mean().item()
    cb = (x[bm] ** 2).mean().item()
    print(f"recon MSE {mse:.5f} | black baseline {black:.5f} | "
          f"bright-pixel recovery {100 * (1 - cm / cb):5.1f}% | pred[min {pred.min():.2f} max {pred.max():.2f} mean {pred.mean():.3f}]")
    print("=> escaped black collapse" if pred.max() > 0.3 and mse < black * 0.9
          else "=> still near black")


if __name__ == "__main__":
    main()
