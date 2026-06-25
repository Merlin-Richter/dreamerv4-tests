"""Deterministic, closed-form readout of GridWorld frames.

The whole reason for the discrete env (D-032): reading the hidden state out of a (predicted or
true) frame is EXACT and self-contained, not a fuzzy blob detection. A frame has 36 cells; 35
are background and 1 is the square. So:

    inferred background = the MEDIAN cell color (robust to the single outlier square),
    square cell         = the cell whose interior color is FARTHEST from that background,
    square / bg color   = nearest of the 4 palette colors to those interior colors.

This needs NO ground-truth input (it infers bg from the frame itself), so it works identically on
model-predicted frames and on true frames. On a true frame it recovers the exact (col,row,color)
by construction — see test_gridworld_eval.py. `margin` (top1−top2 distance-from-bg) is a confidence
/ smear flag: a crisp prediction has a large margin; a hallucinated/blurred square a small one.

All colors BGR (env channel-order contract).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from envs.gridworld import CELL, GRID_N, PALETTE, VIS, cell_origin  # noqa: E402

_PALETTE = np.array(list(PALETTE.values()), dtype=np.float32)  # (4,3) BGR


def cell_mean_colors(frame: np.ndarray) -> np.ndarray:
    """Mean interior color of every cell. frame (H,W,3) -> (GRID_N, GRID_N, 3) float32 (row,col)."""
    f = frame.astype(np.float32)
    out = np.empty((GRID_N, GRID_N, 3), dtype=np.float32)
    for r in range(GRID_N):
        y0 = cell_origin(r)
        for c in range(GRID_N):
            x0 = cell_origin(c)
            out[r, c] = f[y0:y0 + VIS, x0:x0 + VIS].reshape(-1, 3).mean(0)
    return out


def nearest_palette_idx(color: np.ndarray) -> int:
    """Index (PALETTE order) of the closest of the 4 palette colors to `color` (BGR)."""
    return int(np.argmin(((_PALETTE - np.asarray(color, np.float32)) ** 2).sum(-1)))


def read_square(frame: np.ndarray) -> dict:
    """Read hidden state out of one frame. Returns dict:
        col, row          : square cell (ints, 0..GRID_N-1)
        color_idx         : nearest-palette index of the square color
        bg_idx            : nearest-palette index of the inferred background
        margin            : top1-top2 distance-from-bg over cells (confidence)
        is_occluded       : True if the frame has no black grid-line pixels (a flat curtain frame)
    """
    means = cell_mean_colors(frame)                       # (GRID_N,GRID_N,3)
    flat = means.reshape(-1, 3)                            # (GRID_N**2,3)
    bg = np.median(flat, axis=0)                           # inferred background color
    dist = np.sqrt(((flat - bg) ** 2).sum(-1))            # (64,) distance from bg
    order = np.argsort(dist)[::-1]
    top, second = dist[order[0]], dist[order[1]]
    idx = int(order[0])
    row, col = divmod(idx, GRID_N)
    # Occlusion is read off the raw pixels, not the cell means: a revealed frame has black grid
    # lines (some pixel dark in all channels), a curtain frame is flat gray with none. The all-
    # channels test is symmetric, so BGR vs RGB is irrelevant here.
    has_black = bool((frame < 25).all(axis=-1).any())
    return {
        "col": col, "row": row,
        "color_idx": nearest_palette_idx(means[row, col]),
        "bg_idx": nearest_palette_idx(bg),
        "margin": float(top - second),
        "is_occluded": not has_black,  # no black grid-line pixels -> flat curtain frame
    }
