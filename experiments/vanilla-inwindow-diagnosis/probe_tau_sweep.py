"""Mechanism probe: does vanilla denoise (read position from its OWN noisy latent) instead of
predict (derive position from context)?

Setup mirrors the training loss's finest-step flow term exactly: real episode of T=8 revealed
frames; context frames 0..T-2 held near-clean (context_signal on the tau grid); the LAST frame's
own latent noised to a swept tau (z~ = (1-tau) z0 + tau z1); ONE forward at finest d; decode x^1 of
the last frame; read the square.

  pos_acc(tau) ~ 1 at moderate tau but ~ chance at tau ~ 0  =>  the model learned the denoise
  shortcut and never learned dynamics-from-context. Also reported: per-tau flow MSE on the last
  frame, the ramp weight w(tau), and copy_prev (predicting the last VISIBLE frame's cell).

Runs vanilla + ff9 (ff9 forward uses learned-init memory tokens here, NO carried memory — so a high
ff9 score at tau=0 proves position-from-context is learnable through the latent/temporal pathway
in-window, i.e. vanilla's failure is the objective, not the architecture).

Usage:  venv/Scripts/python.exe -u experiments/vanilla-inwindow-diagnosis/probe_tau_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from envs.gridworld import PALETTE, GridWorldEnv  # noqa: E402
from evals.gridworld.readout import read_square  # noqa: E402
from evals.gridworld.recall import _load_checkpoint, _tokenizer_window  # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402

COLOR_NAMES = list(PALETTE.keys())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, T = 64, 8

CKPTS = {
    "vanilla": "checkpoints/gridworld/dynamics_vanilla.pt",
    "ff9": "checkpoints/gridworld/dynamics_ff9.pt",
}


def make_episode(seeds, t):
    """t revealed frames per seed + the per-frame true cells."""
    envs = [GridWorldEnv().reset(s) for s in seeds]
    frames, cells = [], []
    for env in envs:
        fs, cs = [], []
        for _ in range(t):
            f, s = env.step(0)
            fs.append(f)
            cs.append((int(s[0]), int(s[1])))
        frames.append(np.stack(fs))
        cells.append(cs)
    return np.stack(frames), cells


@torch.no_grad()
def main():
    torch.manual_seed(0)
    tok, _ = _load_checkpoint(ROOT / "checkpoints/gridworld/tokenizer.pt",
                              AutoEncoder, AutoEncoderConfig, DEV)
    tok_w = _tokenizer_window(tok)
    frames, cells = make_episode(list(range(B)), T)
    x = torch.from_numpy(frames.astype(np.float32) / 255.0).to(DEV)
    z1 = tok.encoder(x)  # (B, T, L, D) clean latents
    true_last = [c[-1] for c in cells]   # cell in the swept frame (the x-pred target)
    prev_cell = [c[-2] for c in cells]   # cell in the last VISIBLE context frame

    results = {}
    for name, path in CKPTS.items():
        model, cfg = _load_checkpoint(ROOT / path, DynamicsModel, DynamicsModelConfig, DEV)
        K_max, n_d = model.K_max, model.n_d
        ramp_min = cfg.ramp_min
        ctx_idx = min(int(round(cfg.context_signal * K_max)), K_max - 1)
        print(f"\n== {name} (K_max={K_max} n_d={n_d} ramp_min={ramp_min} ctx_tau_idx={ctx_idx})")

        actions = model.action_features(torch.zeros((B, T), dtype=torch.long, device=DEV))
        d_idx = torch.full((B, T), n_d - 1, dtype=torch.long, device=DEV)  # finest step
        sweep = sorted({0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, K_max - 1})
        sweep = [s for s in sweep if s < K_max]

        res = {}
        g = torch.Generator(device=DEV).manual_seed(1234)
        z0_ctx = torch.randn(z1.shape, generator=g, device=DEV)
        for ti in sweep:
            tau_idx = torch.full((B, T), ctx_idx, dtype=torch.long, device=DEV)
            tau_idx[:, -1] = ti
            tau = (tau_idx.float() / K_max)[..., None, None]
            z_tilde = (1 - tau) * z0_ctx + tau * z1
            z_hat1 = model(z_tilde, tau_idx, d_idx, actions)
            mse = float(((z_hat1[:, -1] - z1[:, -1]) ** 2).mean())
            win = torch.cat((z1[:, :-1], z_hat1[:, -1:]), dim=1)[:, -tok_w:]
            dec = tok.decoder(win)[:, -1].clamp(0, 1).cpu().float().numpy()
            pred = (dec * 255.0).round().astype(np.uint8)
            acc, cp, dts = [], [], []
            for b in range(B):
                rd = read_square(pred[b])
                d_t = max(abs(rd["col"] - true_last[b][0]), abs(rd["row"] - true_last[b][1]))
                d_p = max(abs(rd["col"] - prev_cell[b][0]), abs(rd["row"] - prev_cell[b][1]))
                acc.append(int(d_t == 0))
                cp.append(int(d_p == 0))
                dts.append(d_t)
            t_val = ti / K_max
            w = (1 - ramp_min) * t_val + ramp_min
            res[ti] = dict(tau=t_val, pos_acc=float(np.mean(acc)), copy_prev=float(np.mean(cp)),
                           d_true=float(np.mean(dts)), flow_mse=mse, ramp_w=w)
            print(f"  tau={t_val:5.3f} (idx {ti:>2}): pos_acc {np.mean(acc):.3f}  "
                  f"copy_prev {np.mean(cp):.3f}  d_true {np.mean(dts):.2f}  "
                  f"flow_mse {mse:.5f}  w(tau) {w:.2f}")
        results[name] = res

    out = Path(__file__).parent / "results_tau_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
