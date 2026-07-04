"""GridWorldV2 — action-driven GridWorld (spec: specs/envs/gridworldv2.md, DRAFT).

v1's square moves autonomously (velocity + reflection); v2's square moves ONLY when commanded.
7 actions: 0=reveal, 1=hide (curtain LATCH; the square does not move on a toggle tick),
2=up, 3=down, 4=left, 5=right (clamped at walls — blocked move = stay), 6=stay.
Deterministic given (reset seed, action sequence). Under occlusion the hidden position is a
nonlinear function of the whole action stream (clamping), so belief-tracking must integrate
actions, not extrapolate ballistics — the point of this env.

Geometry / palette / rendering are IMPORTED from v1, so the v1 closed-form readout
(`evals/gridworld/readout.read_square`) is exact on v2 frames unchanged.

CHANNEL ORDER: BGR end-to-end (envs/base.py contract).
"""
from __future__ import annotations

import numpy as np

from .base import BaseEnv
from .gridworld import (COLOR_NAMES, CURTAIN_COLOR, GRID_N, IMG_SIZE, PALETTE,
                        make_grid_background, stamp_square)

# Action ids (0/1 keep v1's reveal/occlude semantics so eval conventions transfer).
A_REVEAL, A_HIDE, A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY = range(7)
MOVE_ACTIONS = (A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY)
# action -> (dcol, drow)
MOVES = {A_UP: (0, -1), A_DOWN: (0, 1), A_LEFT: (-1, 0), A_RIGHT: (1, 0), A_STAY: (0, 0)}


def sample_moves(rng: np.random.Generator, n: int, run_max: int = 4) -> list[int]:
    """Movement-action stream in {2..6}: direction RUNS (uniform action, length ~ U{1..run_max}).

    Shared by datagen and the recall eval so train/eval action statistics match; runs give real
    displacement (a pure uniform random walk barely moves).
    """
    out: list[int] = []
    while len(out) < n:
        a = int(rng.choice(MOVE_ACTIONS))
        out.extend([a] * int(rng.integers(1, run_max + 1)))
    return out[:n]


class GridWorldV2Env(BaseEnv):
    """Action-driven grid env: curtain latch (actions 0/1) + clamped square movement (2..6).

    Per-episode constants (random, seeded): background color, square color (!= background),
    start cell. Curtain starts UP. State: ``[col, row, curtain]`` (float32). Categorical colors
    exposed measurement-only exactly like v1 (``.color``/``.color_name``/``.bg_color``/``.bg_name``).
    """

    n_actions = 7

    def __init__(self, img_size: int = IMG_SIZE):
        if img_size != IMG_SIZE:
            raise ValueError(f"GridWorldV2Env geometry is fixed at {IMG_SIZE}px (got {img_size}).")
        self.img_size = img_size
        self.rng = np.random.default_rng()
        self.bg_name = self.color_name = None
        self.bg_color = self.color = None
        self._grid_template = None
        self.col = self.row = 0
        self.curtain = 0
        self.t = 0

    def reset(self, seed: int | None = None) -> "GridWorldV2Env":
        self.rng = np.random.default_rng(seed)
        bg_idx, sq_idx = self.rng.choice(len(COLOR_NAMES), size=2, replace=False)
        self.bg_name = COLOR_NAMES[bg_idx]
        self.color_name = COLOR_NAMES[sq_idx]
        self.bg_color = PALETTE[self.bg_name]
        self.color = PALETTE[self.color_name]
        self._grid_template = make_grid_background(self.bg_color, self.img_size)
        self.col = int(self.rng.integers(0, GRID_N))
        self.row = int(self.rng.integers(0, GRID_N))
        self.curtain = 0
        self.t = 0
        return self

    def _render(self) -> np.ndarray:
        if self.curtain:
            return np.full((self.img_size, self.img_size, 3), CURTAIN_COLOR, dtype=np.uint8)
        return self.render_revealed()

    def render_revealed(self) -> np.ndarray:
        """Measurement-only: the revealed render at the CURRENT position, regardless of the
        curtain latch. The oracle-frame source for the recall eval. NEVER a model input."""
        frame = self._grid_template.copy()
        stamp_square(frame, self.col, self.row, self.color)
        return frame

    def step(self, action: int = A_STAY) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame. Returns (frame uint8 HWC BGR, state float32[3] = [col,row,curtain]).

        Toggle ticks (0/1) set the curtain latch and do NOT move the square; movement ticks
        (2..6) move the square (clamped) and do NOT change the curtain.
        """
        action = int(action)
        if action in (A_REVEAL, A_HIDE):
            self.curtain = action  # A_REVEAL=0 -> up, A_HIDE=1 -> down
        elif action in MOVES:
            dc, dr = MOVES[action]
            self.col = min(max(self.col + dc, 0), GRID_N - 1)
            self.row = min(max(self.row + dr, 0), GRID_N - 1)
        else:
            raise ValueError(f"invalid action {action} (0..6)")
        frame = self._render()
        state = np.array((self.col, self.row, float(self.curtain)), dtype=np.float32)
        self.t += 1
        return frame, state

    def hidden_state(self) -> np.ndarray:
        """Measurement-only: current [col, row, curtain] (float32). NEVER a model input."""
        return np.array((self.col, self.row, float(self.curtain)), dtype=np.float32)
