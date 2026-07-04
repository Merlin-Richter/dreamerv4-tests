"""Analytic Bayes baselines for GridWorldV2 recall — what is inferable WITHOUT memory?

B1 windowed-actions-only: exact posterior over the 6x6 cells starting UNIFORM, propagated through
   only the last (window-1) in-window actions (deterministic clamped transitions). This is the
   ceiling for a memoryless model that reads the visible action tokens: wall-clamping concentrates
   the posterior (e.g. left x5 => col 0), so this is NOT chance.
B2 full-info filter: last-observed cell + ALL occluded actions => exact position (deterministic
   env) = 1.0 by construction — the ceiling for a perfect-memory model (sanity only).

Compares B1 to the SANITIZED vanilla numbers (same seeds/action streams as the recall protocol).

Usage:  venv/Scripts/python.exe -u experiments/gridworldv2-arms/bayes_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from envs.gridworld import GRID_N  # noqa: E402
from envs.gridworldv2 import MOVES, A_HIDE, GridWorldV2Env, sample_moves  # noqa: E402
from evals.gridworld.recall import _check_ks  # noqa: E402

N_CTX, MAX_K, B = 4, 64, 64


def propagate(post: np.ndarray, action: int) -> np.ndarray:
    """Exact one-step posterior propagation under a deterministic clamped move (col,row grid)."""
    if action not in MOVES:
        return post  # toggle ticks don't move
    dc, dr = MOVES[action]
    out = np.zeros_like(post)
    for c in range(GRID_N):
        for r in range(GRID_N):
            nc = min(max(c + dc, 0), GRID_N - 1)
            nr = min(max(r + dr, 0), GRID_N - 1)
            out[nc, nr] += post[c, r]
    return out


def main():
    check = _check_ks(MAX_K)
    acc = {w: {k: [] for k in check} for w in (16, 8)}
    for seed in range(B):
        env = GridWorldV2Env().reset(seed)
        stream = sample_moves(env.rng, N_CTX + MAX_K)
        # replay the protocol to get true cells at every occluded tick
        for i in range(N_CTX):
            env.step(stream[i])
        env.step(A_HIDE)
        actions_occ = stream[N_CTX:]
        cells = []
        for a in actions_occ:
            _, s = env.step(a)
            cells.append((int(s[0]), int(s[1])))
        # B1: at each checked k, uniform posterior propagated through the last (w-1) actions
        # (the actions whose frames are in the sliding window at the branch position).
        for w in (16, 8):
            for k in check:
                hist = actions_occ[max(0, k - (w - 1)):k]
                post = np.full((GRID_N, GRID_N), 1.0 / GRID_N ** 2)
                for a in hist:
                    post = propagate(post, a)
                pred = np.unravel_index(np.argmax(post), post.shape)
                acc[w][k].append(int(pred == cells[k - 1]))
    out = {f"w{w}": {k: float(np.mean(v)) for k, v in acc[w].items()} for w in (16, 8)}
    for w in (16, 8):
        print(f"B1 windowed-actions-only Bayes (w={w}): " +
              " ".join(f"k{k}={out[f'w{w}'][k]:.2f}" for k in check))
    Path(__file__).with_name("results_bayes_baselines.json").write_text(json.dumps(out, indent=2))
    print("wrote results_bayes_baselines.json")


if __name__ == "__main__":
    main()
