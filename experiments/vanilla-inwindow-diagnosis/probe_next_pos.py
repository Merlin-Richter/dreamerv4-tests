"""Probe: is vanilla's GridWorld position failure an IN-WINDOW LEARNING failure or a rollout artifact?

Everything here is fully REVEALED (action=0 everywhere): no occlusion => no memory demand at all.
A model that learned the (deterministic) transition map must score ~1.0 on next-cell position.

Conditions:
  A. teacher-forced 1-step: t TRUE frames as context (t in {2,4,8,15}) -> rollout_init -> ONE
     read-only rollout_step -> decode -> read the square -> score vs the env truth advanced 1 tick.
     Zero compounding; purest test of "did it learn the transition map".
  B. free-run: 4 true ctx frames -> 12 committed rollout_steps (exactly the sheet_normal setting),
     score every step vs env truth. Difference vs A isolates compounding.

Metrics per condition:
  pos_acc   exact 6x6 cell match vs true next cell
  d_true    mean Chebyshev cell-distance to the true cell
  copy_rate fraction predicting the LAST OBSERVED cell (behavioral mode: copy vs extrapolate vs random)
  col_acc   4-way square-colour accuracy (sanity: colour pathway works)

Usage:  venv/Scripts/python.exe -u experiments/vanilla-inwindow-diagnosis/probe_next_pos.py
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
B = 64
K = 4

CKPTS = {
    "vanilla": "checkpoints/gridworld/dynamics_vanilla.pt",
    "ff9": "checkpoints/gridworld/dynamics_ff9.pt",
    "mem2mem_5050": "checkpoints/gridworld/dynamics_mem2mem.pt",
    "m2m_rollout_noff9": "checkpoints/gridworld/dynamics_mem2mem_rollout_noff9_clean.pt",
}


def make_context(seeds, t):
    """t revealed true frames per seed. Returns envs, frames (B,t,H,W,3), last-observed cells, colors."""
    envs = [GridWorldEnv().reset(s) for s in seeds]
    frames, last = [], []
    for env in envs:
        fs, s = [], None
        for _ in range(t):
            f, s = env.step(0)
            fs.append(f)
        frames.append(np.stack(fs))
        last.append((int(s[0]), int(s[1])))
    colors = [COLOR_NAMES.index(e.color_name) for e in envs]
    return envs, np.stack(frames), last, colors


@torch.no_grad()
def encode(tok, frames):
    x = torch.from_numpy(frames.astype(np.float32) / 255.0).to(DEV)
    return tok.encoder(x)


@torch.no_grad()
def decode_last(tok, lat_buf, z, tok_w):
    win = torch.cat((lat_buf, z), dim=1)[:, -tok_w:]
    dec = tok.decoder(win)[:, -1].clamp(0, 1).cpu().float().numpy()
    return (dec * 255.0).round().astype(np.uint8)


def score(pred, true_cells, last, colors):
    accs, dts, copies, cols = [], [], [], []
    for b in range(pred.shape[0]):
        rd = read_square(pred[b])
        d_true = max(abs(rd["col"] - true_cells[b][0]), abs(rd["row"] - true_cells[b][1]))
        d_last = max(abs(rd["col"] - last[b][0]), abs(rd["row"] - last[b][1]))
        accs.append(int(d_true == 0))
        dts.append(d_true)
        copies.append(int(d_last == 0))
        cols.append(int(rd["color_idx"] == colors[b]))
    return dict(pos_acc=float(np.mean(accs)), d_true=float(np.mean(dts)),
                copy_rate=float(np.mean(copies)), col_acc=float(np.mean(cols)))


@torch.no_grad()
def run_model(name, path, tok, tok_w):
    model, cfg = _load_checkpoint(ROOT / path, DynamicsModel, DynamicsModelConfig, DEV)
    print(f"\n== {name}  (n_memory={cfg.n_memory} ff9_k={cfg.ff9_k} "
          f"W={cfg.max_temporal_length} n_actions={cfg.n_actions})")
    seeds = list(range(B))
    a0 = torch.zeros((B,), dtype=torch.long, device=DEV)
    out = {"teacher_forced": {}, "free_run": {}}

    # A: teacher-forced 1-step at several context lengths
    for t in (2, 4, 8, 15):
        envs, frames, last, colors = make_context(seeds, t)
        lat = encode(tok, frames)
        state = model.rollout_init(lat, torch.zeros((B, t), dtype=torch.long, device=DEV), K)
        true_cells = []
        for env in envs:
            _, s = env.step(0)  # advance truth ONE tick
            true_cells.append((int(s[0]), int(s[1])))
        z = model.rollout_step(state, a0, commit=False)
        pred = decode_last(tok, lat[:, -(tok_w - 1):], z, tok_w)
        r = score(pred, true_cells, last, colors)
        out["teacher_forced"][t] = r
        print(f"  A t={t:>2} 1-step : pos_acc {r['pos_acc']:.3f}  d_true {r['d_true']:.2f}  "
              f"copy_rate {r['copy_rate']:.3f}  col_acc {r['col_acc']:.3f}")

    # B: free-run from 4 ctx frames (the sheet_normal setting)
    envs, frames, last, colors = make_context(seeds, 4)
    lat = encode(tok, frames)
    state = model.rollout_init(lat, torch.zeros((B, 4), dtype=torch.long, device=DEV), K)
    lat_buf = lat[:, -(tok_w - 1):]
    for j in range(1, 13):
        true_cells = []
        for env in envs:
            _, s = env.step(0)
            true_cells.append((int(s[0]), int(s[1])))
        z = model.rollout_step(state, a0, commit=True)
        pred = decode_last(tok, lat_buf, z, tok_w)
        lat_buf = torch.cat((lat_buf, z), dim=1)[:, -(tok_w - 1):]
        r = score(pred, true_cells, last, colors)
        out["free_run"][j] = r
        if j in (1, 2, 3, 4, 8, 12):
            print(f"  B free-run j={j:>2}: pos_acc {r['pos_acc']:.3f}  d_true {r['d_true']:.2f}  "
                  f"copy_rate {r['copy_rate']:.3f}  col_acc {r['col_acc']:.3f}")
    return out


def main():
    torch.manual_seed(0)
    tok, _ = _load_checkpoint(ROOT / "checkpoints/gridworld/tokenizer.pt",
                              AutoEncoder, AutoEncoderConfig, DEV)
    for p in tok.parameters():
        p.requires_grad_(False)
    tok_w = _tokenizer_window(tok)
    print(f"device={DEV} B={B} K={K} tok_window={tok_w}")
    results = {}
    for name, path in CKPTS.items():
        if not (ROOT / path).exists():
            print(f"\n== {name}: MISSING {path} — skipped")
            continue
        results[name] = run_model(name, path, tok, tok_w)
    out = Path(__file__).parent / "results_next_pos.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
