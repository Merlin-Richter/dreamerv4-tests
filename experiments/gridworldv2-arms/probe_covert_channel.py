"""Covert-channel probe: does a 'no-memory' model carry belief through its OWN committed
occluded latents?

Standard recall commits the MODEL'S generated latent for each occluded tick (rollout_step) —
nothing forces a generated 'gray' frame's latents to be information-free, so a vanilla model
could relay its position belief through them (covert channel). SANITIZED recall instead commits
the encoded TRUE curtain frames (teacher-forced occlusion via model._commit_context_frame):
committed latents are ground-truth gray = position-free by construction. Only memory tokens (if
any) can carry state.

Prediction if covert: vanilla's past-window residual (w16 high-k ~0.2-0.3) collapses to
~copy_last under sanitization. For the sparse arm, sanitized-vs-standard decomposes its carrier
(memory tokens vs committed latents).

Usage:  venv/Scripts/python.exe -u experiments/gridworldv2-arms/probe_covert_channel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "sparse-write-slots"))
from model import DynamicsModelSparseWS  # noqa: E402
from envs.gridworld import PALETTE  # noqa: E402
from envs.gridworldv2 import A_HIDE, A_REVEAL, GridWorldV2Env, sample_moves  # noqa: E402
from evals.gridworld.recall import (_check_ks, _load_checkpoint, _tokenizer_window,  # noqa: E402
                                    position_credit, score_reveal)
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402

COLOR_NAMES = list(PALETTE.keys())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = {
    "A_vanilla_tau0": ("checkpoints/gridworldv2/dynamics_vanilla_tau0.pt", DynamicsModel),
    "D_sparse_n8": ("checkpoints/gridworldv2/dynamics_sparse_n8.pt", DynamicsModelSparseWS),
}


@torch.no_grad()
def encode_seq(tok, frames, tok_w):
    """Encode (B,T,H,W,3) uint8 in non-overlapping tok_w windows (latent-cache convention)."""
    x = torch.from_numpy(frames.astype(np.float32) / 255.0).to(DEV)
    zs = [tok.encoder(x[:, w0:w0 + tok_w]) for w0 in range(0, x.shape[1], tok_w)]
    return torch.cat(zs, dim=1)


@torch.no_grad()
def sanitized_roll(model, tok, seeds, n_ctx, max_k, K, window):
    """Recallv2 protocol but occluded ticks are TEACHER-FORCED with true curtain latents."""
    tok_w = _tokenizer_window(tok)
    max_ctx = max(1, window - 1)
    B = len(seeds)
    envs = [GridWorldV2Env().reset(s) for s in seeds]
    streams = [sample_moves(env.rng, n_ctx + max_k) for env in envs]

    # true frame sequence: n_ctx revealed moves + hide + max_k occluded moves
    T_total = n_ctx + 1 + max_k
    frames = np.empty((B, T_total, 64, 64, 3), dtype=np.uint8)
    acts = np.empty((B, T_total), dtype=np.int64)
    cells = [[] for _ in range(B)]
    for b, env in enumerate(envs):
        t = 0
        for i in range(n_ctx):
            a = streams[b][i]
            frames[b, t], s = env.step(a)
            acts[b, t] = a
            cells[b].append((int(s[0]), int(s[1])))
            t += 1
        frames[b, t], s = env.step(A_HIDE)
        acts[b, t] = A_HIDE
        cells[b].append((int(s[0]), int(s[1])))
        t += 1
        for i in range(max_k):
            a = streams[b][n_ctx + i]
            frames[b, t], s = env.step(a)
            acts[b, t] = a
            cells[b].append((int(s[0]), int(s[1])))
            t += 1
    lat = encode_seq(tok, frames, tok_w)                      # (B, T_total, L, D) TRUE latents
    colors = [(COLOR_NAMES.index(e.bg_name), COLOR_NAMES.index(e.color_name)) for e in envs]
    last = [c[n_ctx - 1] for c in cells]                      # last OBSERVED cell

    act_t = torch.from_numpy(acts).to(DEV)
    state = model.rollout_init(lat[:, :n_ctx], act_t[:, :n_ctx], K, max_ctx=max_ctx)
    lat_buf = lat[:, :n_ctx][:, -(tok_w - 1):]
    # teacher-forced hide tick + occluded ticks: commit TRUE latents
    model._commit_context_frame(state, lat[:, n_ctx:n_ctx + 1], act_t[:, n_ctx:n_ctx + 1])
    lat_buf = torch.cat((lat_buf, lat[:, n_ctx:n_ctx + 1]), dim=1)[:, -(tok_w - 1):]

    a_rev = torch.full((B,), A_REVEAL, dtype=torch.long, device=DEV)
    check = set(_check_ks(max_k))
    recs = []
    for k in range(1, max_k + 1):
        t = n_ctx + k
        model._commit_context_frame(state, lat[:, t:t + 1], act_t[:, t:t + 1])
        lat_buf = torch.cat((lat_buf, lat[:, t:t + 1]), dim=1)[:, -(tok_w - 1):]
        if k in check:
            z_rev = model.rollout_step(state, a_rev, commit=False)
            win = torch.cat((lat_buf, z_rev), dim=1)[:, -tok_w:]
            dec = tok.decoder(win)[:, -1].clamp(0, 1).cpu().float().numpy()
            pred = (dec * 255.0).round().astype(np.uint8)
            for b in range(B):
                tcell = cells[b][t]
                recs.append((k, score_reveal(pred[b], tcell, colors[b])["pos_correct"],
                             int(max(abs(last[b][0] - tcell[0]), abs(last[b][1] - tcell[1])) == 0)))
    out = {}
    for k in sorted(check):
        vals = [r[1] for r in recs if r[0] == k]
        cls = [r[2] for r in recs if r[0] == k]
        out[k] = (float(np.mean(vals)), float(np.mean(cls)))
    return out


def main():
    torch.manual_seed(0)
    tok, _ = _load_checkpoint(ROOT / "checkpoints/gridworld/tokenizer.pt",
                              AutoEncoder, AutoEncoderConfig, DEV)
    for p in tok.parameters():
        p.requires_grad_(False)
    results = {}
    for name, (path, cls) in ARMS.items():
        model, cfg = _load_checkpoint(ROOT / path, cls, DynamicsModelConfig, DEV)
        for window in (16, 8):
            r = sanitized_roll(model, tok, list(range(64)), 4, 64, 4, window)
            results[f"{name}_w{window}"] = {k: v[0] for k, v in r.items()}
            print(f"== {name} w{window} SANITIZED: " +
                  " ".join(f"k{k}={v[0]:.2f}" for k, v in sorted(r.items())))
            if name.startswith("A") and window == 16:
                print("   copy_last:            " +
                      " ".join(f"k{k}={v[1]:.2f}" for k, v in sorted(r.items())))
        del model
        torch.cuda.empty_cache() if DEV == "cuda" else None
    Path(__file__).with_name("results_covert_channel.json").write_text(json.dumps(results, indent=2))
    print("wrote results_covert_channel.json")


if __name__ == "__main__":
    main()
