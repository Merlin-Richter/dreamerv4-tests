"""
GridWorld occluded-memory environment — a clean, discrete, low-entropy memory env.

Why this env (vs. OccludedBouncingEnv):
  * **Discrete & solid, not fluid.** The world is a 6x6 grid drawn with black lines. A single
    square occupies exactly one cell and moves exactly one cell per tick. No sub-pixel
    positions, no continuous velocity, no textured gradient backgrounds. This makes recall a
    crisp classification problem rather than a fuzzy regression:
        - position recall -> which of the 6x6=36 cells (exact-match accuracy), and
        - color recall    -> which of 4 colors (4-way accuracy),
    both measurable without the ΔRGB / sub-pixel noise the fluid env suffered from.
  * **Low, controlled entropy.** Background is ONE solid color from {red, green, blue, pink};
    the square is a DIFFERENT one of those four. Hidden state to retain across an occlusion is
    therefore exactly: square color (4-way), square cell (6x6), and direction (1 of 8).
  * **Same occlusion logic as OccludedBouncingEnv.** Two *absolute* actions set the curtain for
    the current frame:
        action[t] = 0 -> curtain UP   (revealed: grid + background + square visible)
        action[t] = 1 -> curtain DOWN (occluded: a flat opaque gray fills the whole frame)
    The square keeps moving (with wall reflections) behind the curtain, so predicting the
    reveal frame requires integrating hidden position+direction across the occluded stretch.

Geometry (64x64, chosen with Merlin — D-038):
    3px border + 6 cells * 8px interior + 5 internal lines * 2px + 3px border = 64.
    Cell `i`'s interior starts at pixel ``3 + 10*i`` and spans 8px; everything else is a black
    line/border pixel. Every cell interior is a uniform 8x8 block. Cells are deliberately NOT
    8px-stride-aligned to the tokenizer's 8x8 patch grid (8px interior + 2px line = 10px stride),
    so the tokenizer cannot trivially overfit one cell per patch (the change's whole purpose).

Convention: ``action[t]`` describes frame ``t`` (the curtain you observe at t), matching the
per-frame action token to the frame it explains — identical to OccludedBouncingEnv.

This module is the ENV ONLY (the steppable simulator). Dataset writing and playback live in
`src/datagen/generate_gridworld.py`.

CHANNEL ORDER: BGR end-to-end (see envs/base.py contract). All colors below are BGR.
"""
from __future__ import annotations

import numpy as np

from .base import BaseEnv

# ---------------------------------------------------------------------------
# Geometry + palette
# ---------------------------------------------------------------------------

IMG_SIZE = 64
GRID_N = 6        # cells per axis
CELL = 10         # stride per cell (px): 8 interior + 2 line
VIS = 8           # visible interior of each cell (px)
BORDER = 3        # black border on each side (px)
BLACK = (0, 0, 0)
CURTAIN_COLOR = (128, 128, 128)  # neutral gray: distinct from black lines AND all 4 palette colors

# Four allowed colors, BGR (native cv2 / dataset order). Ordered -> stable class indices.
PALETTE: dict[str, tuple[int, int, int]] = {
    "red":   (0,   0,   255),
    "green": (0,   200, 0),
    "blue":  (255, 0,   0),
    "pink":  (180, 105, 255),
}
COLOR_NAMES = tuple(PALETTE.keys())  # index <-> name mapping for 4-way recall


def cell_origin(idx: int) -> int:
    """Top-left interior pixel of cell `idx` (0..GRID_N-1) along one axis."""
    return BORDER + CELL * idx


def interior_axis_mask(img_size: int = IMG_SIZE) -> np.ndarray:
    """Boolean mask over one axis: True = cell interior, False = black line/border.

    Layout: 3px border, then 8px interior / 2px line alternating (6 interiors, 5 internal
    lines), ending in a 3px border: 3 + 6*8 + 5*2 + 3 = 64. Built explicitly per cell so
    there are no modular edge cases.
    """
    mask = np.zeros(img_size, dtype=bool)
    for idx in range(GRID_N):
        o = cell_origin(idx)
        mask[o:o + VIS] = True
    return mask


def make_grid_background(bg_color: tuple[int, int, int], img_size: int = IMG_SIZE) -> np.ndarray:
    """Solid-color frame with the black grid drawn on top (no square). Returns HWC uint8 BGR."""
    frame = np.empty((img_size, img_size, 3), dtype=np.uint8)
    frame[:] = bg_color
    black_axis = ~interior_axis_mask(img_size)
    frame[black_axis, :, :] = BLACK
    frame[:, black_axis, :] = BLACK
    return frame


def stamp_square(frame: np.ndarray, col: int, row: int, color: tuple[int, int, int]) -> None:
    """In-place: fill cell (col,row)'s 8x8 interior with `color`."""
    y0, x0 = cell_origin(row), cell_origin(col)
    frame[y0:y0 + VIS, x0:x0 + VIS] = color


# All 8 directions: (dcol, drow) in {-1,0,1}^2 minus (0,0). Orthogonal + diagonal.
DIRECTIONS: tuple[tuple[int, int], ...] = tuple(
    (dc, dr) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dc == 0 and dr == 0)
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GridWorldEnv(BaseEnv):
    """Single-episode discrete grid env: a square steps one cell/tick and reflects off walls.

    Per-episode constants (random, seeded): background color, square color (!= background),
    start cell, start direction. The curtain action chooses revealed vs. occluded rendering;
    physics runs regardless.

    State layout (per-step `state` and `hidden_state()`): ``[col, row, dcol, drow, curtain]``
    (float32). Categorical color is exposed separately for measurement: ``.color`` (square BGR,
    BaseEnv convention), ``.color_name``, ``.bg_color`` (BGR), ``.bg_name`` — all measurement-only.
    """

    n_actions = 2  # 0 = curtain up (revealed), 1 = curtain down (occluded)

    def __init__(self, img_size: int = IMG_SIZE):
        if img_size != IMG_SIZE:
            # Geometry constants assume the 64px / 6-cell layout.
            raise ValueError(f"GridWorldEnv geometry is fixed at {IMG_SIZE}px (got {img_size}).")
        self.img_size = img_size
        self.rng = np.random.default_rng()
        self.bg_name = self.color_name = None
        self.bg_color = self.color = None
        self._grid_template = None  # cached background+grid (square stamped per-frame)
        self.col = self.row = 0
        self.dcol = self.drow = 0
        self.curtain = 0
        self.t = 0

    def reset(self, seed: int | None = None) -> "GridWorldEnv":
        self.rng = np.random.default_rng(seed)
        # Distinct background and square colors from the 4-color palette.
        bg_idx, sq_idx = self.rng.choice(len(COLOR_NAMES), size=2, replace=False)
        self.bg_name = COLOR_NAMES[bg_idx]
        self.color_name = COLOR_NAMES[sq_idx]
        self.bg_color = PALETTE[self.bg_name]
        self.color = PALETTE[self.color_name]
        self._grid_template = make_grid_background(self.bg_color, self.img_size)
        # Random start cell + direction (1 of 8).
        self.col = int(self.rng.integers(0, GRID_N))
        self.row = int(self.rng.integers(0, GRID_N))
        self.dcol, self.drow = DIRECTIONS[int(self.rng.integers(0, len(DIRECTIONS)))]
        self.curtain = 0
        self.t = 0
        return self

    @staticmethod
    def _reflect(pos: int, vel: int) -> tuple[int, int]:
        """One discrete step with wall reflection on a 0..GRID_N-1 axis. Returns (new_pos, new_vel)."""
        nxt = pos + vel
        if nxt < 0 or nxt > GRID_N - 1:
            vel = -vel
            nxt = pos + vel
        return nxt, vel

    def _advance_physics(self) -> None:
        self.col, self.dcol = self._reflect(self.col, self.dcol)
        self.row, self.drow = self._reflect(self.row, self.drow)

    def _render(self, action: int) -> np.ndarray:
        if action:
            return np.full((self.img_size, self.img_size, 3), CURTAIN_COLOR, dtype=np.uint8)
        frame = self._grid_template.copy()
        stamp_square(frame, self.col, self.row, self.color)
        return frame

    def step(self, action: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame. Returns (frame uint8 HWC BGR, state float32[5])."""
        self._advance_physics()
        self.curtain = int(action)
        frame = self._render(self.curtain)
        state = np.array(
            (self.col, self.row, self.dcol, self.drow, float(self.curtain)), dtype=np.float32
        )
        self.t += 1
        return frame, state

    def hidden_state(self) -> np.ndarray:
        """Measurement-only: current [col, row, dcol, drow, curtain] (float32). NEVER a model input."""
        return np.array(
            (self.col, self.row, self.dcol, self.drow, float(self.curtain)), dtype=np.float32
        )
