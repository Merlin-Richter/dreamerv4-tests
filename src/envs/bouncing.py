"""
Bouncing-shape environment — DVD-screensaver-style sim (the unconditioned baseline env).

Shapes: circle, rect, donut, diamond, triangle, star, cross.

This module is the ENV ONLY (shape drawing + a steppable `BouncingEnv`). Dataset writing and
playback live in `src/datagen/generate_bouncing.py`. `BouncingEnv` has `n_actions == 0` (no
actions): `step()` ignores its argument and just advances physics — it is the simplest
`BaseEnv`, kept conforming so evals/data-gen treat all envs uniformly.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import BaseEnv


# ---------------------------------------------------------------------------
# Shape drawing
# ---------------------------------------------------------------------------

def _polygon(frame, cx, cy, n_sides, radius, angle_offset, color):
    """Draw a filled regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle_offset
    pts = np.array(
        [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], color)


def draw_shape(frame, shape, cx, cy, radius, color):
    """Draw `shape` centred at (cx, cy) with given radius and color."""

    if shape == "circle":
        cv2.circle(frame, (cx, cy), radius, color, -1)

    elif shape == "rect":
        cv2.rectangle(
            frame,
            (cx - radius, cy - radius),
            (cx + radius, cy + radius),
            color, -1,
        )

    elif shape == "donut":
        cv2.circle(frame, (cx, cy), radius, color, -1)
        # Punch a hole — use black, ~40 % of outer radius
        hole_r = max(1, int(radius * 0.42))
        cv2.circle(frame, (cx, cy), hole_r, (0, 0, 0), -1)

    elif shape == "diamond":
        # 4-sided polygon rotated 45°
        _polygon(frame, cx, cy, 4, radius, np.pi / 4, color)

    elif shape == "triangle":
        # Equilateral, pointing up
        _polygon(frame, cx, cy, 3, radius, -np.pi / 2, color)

    elif shape == "star":
        # 5-point star: alternate outer/inner vertices
        outer_r = radius
        inner_r = max(1, int(radius * 0.42))
        pts = []
        for i in range(10):
            r = outer_r if i % 2 == 0 else inner_r
            a = np.pi * i / 5 - np.pi / 2
            pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
        cv2.fillPoly(frame, [np.array(pts, dtype=np.int32)], color)

    elif shape == "cross":
        arm = radius
        thickness = max(2, radius // 3)
        cv2.rectangle(frame, (cx - arm, cy - thickness),
                              (cx + arm, cy + thickness), color, -1)
        cv2.rectangle(frame, (cx - thickness, cy - arm),
                              (cx + thickness, cy + arm), color, -1)

    else:
        raise ValueError(f"Unknown shape: {shape!r}")


SHAPES = ["circle", "rect", "donut", "diamond", "triangle", "star", "cross"]


# ---------------------------------------------------------------------------
# Step-by-step environment
# ---------------------------------------------------------------------------

class BouncingEnv(BaseEnv):
    """Single shape bouncing in a box. Unconditioned (`n_actions == 0`).

    State layout (per-step, returned by `step` and `hidden_state`): ``[x, y, vx, vy]``
    (float32). ``.color`` is the current shape colour (BGR); it changes on bounce only when
    `change_color_on_bounce=True`.
    """

    n_actions = 0

    def __init__(
        self,
        img_size: int = 64,
        shape: str = "circle",
        radius: int = 8,
        color=None,
        change_color_on_bounce: bool = False,
    ):
        self.img_size = img_size
        self.shape = shape
        self.radius = radius
        self._init_color = color
        self.change_color_on_bounce = change_color_on_bounce
        self.rng = np.random.default_rng()
        self.color = None
        self.x = self.y = self.vx = self.vy = 0.0
        self.t = 0

    def _random_color(self):
        return tuple(self.rng.integers(80, 255, size=3).tolist())

    def reset(self, seed: int = 42) -> "BouncingEnv":
        self.rng = np.random.default_rng(seed)
        self.color = self._random_color() if self._init_color is None else self._init_color
        margin = self.radius + 1
        self.x = float(self.rng.uniform(margin, self.img_size - margin))
        self.y = float(self.rng.uniform(margin, self.img_size - margin))
        self.vx = float(self.rng.choice([-1, 1]) * self.rng.uniform(1.5, 3.0))
        self.vy = float(self.rng.choice([-1, 1]) * self.rng.uniform(1.5, 3.0))
        self.t = 0
        return self

    def step(self, action: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame (action ignored). Returns (frame uint8 HWC, state float32[4])."""
        img_size, radius = self.img_size, self.radius
        self.x += self.vx
        self.y += self.vy

        bounced = False
        if self.x - radius < 0:
            self.vx = abs(self.vx)
            self.x = float(radius)
            bounced = True
        elif self.x + radius > img_size - 1:
            self.vx = -abs(self.vx)
            self.x = float(img_size - 1 - radius)
            bounced = True
        if self.y - radius < 0:
            self.vy = abs(self.vy)
            self.y = float(radius)
            bounced = True
        elif self.y + radius > img_size - 1:
            self.vy = -abs(self.vy)
            self.y = float(img_size - 1 - radius)
            bounced = True

        if bounced and self.change_color_on_bounce:
            self.color = self._random_color()

        frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        draw_shape(frame, self.shape, int(round(self.x)), int(round(self.y)), radius, self.color)
        state = np.array((self.x, self.y, self.vx, self.vy), dtype=np.float32)
        self.t += 1
        return frame, state

    def hidden_state(self) -> np.ndarray:
        """Measurement-only: current [x, y, vx, vy] (float32). NEVER a model input."""
        return np.array((self.x, self.y, self.vx, self.vy), dtype=np.float32)
