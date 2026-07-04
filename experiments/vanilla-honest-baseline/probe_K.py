"""K-sweep of the inference sampler (Merlin's ask): teacher-forced next-frame + short free-run
position accuracy as a function of shortcut steps K per generated frame.

K=1  -> single x-pred step from pure noise (d=1)
K=4  -> the standard inference schedule (4 steps of d=1/4)
K=128-> K_max: every step at d_min=1/128 — the exact conditioning Arm D's anchor trained.

Reuses the probe machinery (same seeds -> comparable to results_probe.json).
Usage:  venv/Scripts/python.exe -u experiments/vanilla-honest-baseline/probe_K.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "vanilla-inwindow-diagnosis"))
from probe_next_pos import (B, decode_last, encode, make_context, score,  # noqa: E402
                            _load_checkpoint, _tokenizer_window)
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [1, 2, 4, 128]
CKPTS = {
    "armD_tau0": "checkpoints/gridworld/dynamics_vanilla_tau0.pt",
    "vanilla_ref": "checkpoints/gridworld/dynamics_vanilla.pt",
}


@torch.no_grad()
def run(name, path, tok, tok_w):
    model, cfg = _load_checkpoint(ROOT / path, DynamicsModel, DynamicsModelConfig, DEV)
    print(f"\n== {name} (K_max={cfg.max_sampling_steps})")
    seeds = list(range(B))
    a0 = torch.zeros((B,), dtype=torch.long, device=DEV)
    out = {}
    for K in KS:
        res = {"tf": {}, "fr": {}}
        # teacher-forced 1 frame ahead, all-real revealed context
        for t in (4, 8):
            torch.manual_seed(0)
            envs, frames, last, colors = make_context(seeds, t)
            lat = encode(tok, frames)
            state = model.rollout_init(lat, torch.zeros((B, t), dtype=torch.long, device=DEV), K)
            true_cells = []
            for env in envs:
                _, s = env.step(0)
                true_cells.append((int(s[0]), int(s[1])))
            z = model.rollout_step(state, a0, commit=False)
            pred = decode_last(tok, lat[:, -(tok_w - 1):], z, tok_w)
            res["tf"][t] = score(pred, true_cells, last, colors)
        # short free-run (4 frames) from 4 ctx
        torch.manual_seed(0)
        envs, frames, last, colors = make_context(seeds, 4)
        lat = encode(tok, frames)
        state = model.rollout_init(lat, torch.zeros((B, 4), dtype=torch.long, device=DEV), K)
        lat_buf = lat[:, -(tok_w - 1):]
        for j in range(1, 5):
            true_cells = []
            for env in envs:
                _, s = env.step(0)
                true_cells.append((int(s[0]), int(s[1])))
            z = model.rollout_step(state, a0, commit=True)
            pred = decode_last(tok, lat_buf, z, tok_w)
            lat_buf = torch.cat((lat_buf, z), dim=1)[:, -(tok_w - 1):]
            res["fr"][j] = score(pred, true_cells, last, colors)
        fr = [res["fr"][j]["pos_acc"] for j in range(1, 5)]
        print(f"  K={K:>3}: tf t=4 {res['tf'][4]['pos_acc']:.3f}  t=8 {res['tf'][8]['pos_acc']:.3f}"
              f"  | free-run j1..4: " + " ".join(f"{v:.3f}" for v in fr)
              + f"  | d_true tf8 {res['tf'][8]['d_true']:.2f}")
        out[K] = res
    return out


def main():
    torch.manual_seed(0)
    tok, _ = _load_checkpoint(ROOT / "checkpoints/gridworld/tokenizer.pt",
                              AutoEncoder, AutoEncoderConfig, DEV)
    for p in tok.parameters():
        p.requires_grad_(False)
    tok_w = _tokenizer_window(tok)
    results = {}
    for name, path in CKPTS.items():
        results[name] = run(name, path, tok, tok_w)
    out = Path(__file__).parent / "results_K_sweep.json"
    out.write_text(json.dumps({m: {str(k): v for k, v in r.items()} for m, r in results.items()},
                              indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
