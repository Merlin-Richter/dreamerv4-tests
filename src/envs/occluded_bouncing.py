"""
Occluded bouncing-ball environment — a dense, action-conditioned memory env.

Why this env:
  * **Dense frames.** Every frame has a smooth per-episode textured background, so the
    reconstruction target is never the degenerate all-black image. (A ~5%-area object on a
    black background is a pathological case for MSE: predicting all-black is a deep local
    optimum. Dense backgrounds remove that trap.)
  * **Actions.** Two *absolute* actions set the curtain state for the current frame:
        action[t] = 0  -> curtain UP   (revealed: background + ball visible)
        action[t] = 1  -> curtain DOWN (occluded: an opaque curtain hides the whole frame)
    Absolute (not toggle) actions are Markov in the action, so the curtain itself needs no
    memory — only the *hidden ball* does.
  * **Memory.** The ball keeps bouncing (with wall reflections) behind the curtain. To predict
    the frame where the curtain lifts, a model must integrate the ball's hidden position and
    velocity across the occluded stretch — information absent from every occluded frame.

Convention: ``action[t]`` describes frame ``t`` (the curtain state you observe at t), so a
per-frame action token lines up with the frame it explains.

This module is the ENV ONLY (the steppable simulator). Dataset writing and playback live in
`src/data/generate_occluded.py`; see that module for the `.npy` writer and viewers.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import BaseEnv


# ---------------------------------------------------------------------------
# Background + curtain
# ---------------------------------------------------------------------------

# Curtain is a fixed colour across the whole dataset so "curtain down" is a single, easily
# learned appearance fully determined by the action. Mid-tone (not near-black) on purpose.
CURTAIN_COLOR = (52, 48, 60)  # BGR


def _dark_color(rng: np.random.Generator, val_lo: int = 50, val_hi: int = 95) -> np.ndarray:
    """One dark but hueful BGR colour — random hue, moderate saturation, low value."""
    hue = int(rng.integers(0, 180))
    sat = int(rng.integers(140, 220))
    val = int(rng.integers(val_lo, val_hi))
    hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


def make_background(rng: np.random.Generator, img_size: int) -> np.ndarray:
    """Smooth dark gradient between 2-3 random hue anchors (e.g. dark blue → dark orange).

    Low frequency keeps it easy for a patch (ViT) tokenizer while still filling every pixel.
    """
    c_a = _dark_color(rng)
    c_b = _dark_color(rng)
    if rng.random() < 0.5:
        # Two-colour gradient (horizontal or vertical).
        if rng.random() < 0.5:
            corners = np.array([[c_a, c_b], [c_a, c_b]], dtype=np.uint8)
        else:
            corners = np.array([[c_a, c_a], [c_b, c_b]], dtype=np.uint8)
    else:
        # Three-colour gradient: three distinct corners, fourth repeats one anchor.
        c_c = _dark_color(rng)
        repeat = c_a if rng.random() < 0.5 else c_b
        corners = np.array([[c_a, c_b], [c_c, repeat]], dtype=np.uint8)
    return cv2.resize(corners, (img_size, img_size), interpolation=cv2.INTER_LINEAR)


def _bright_ball_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """High-value, saturated BGR colour that pops against the dark background."""
    hue = int(rng.integers(0, 180))
    sat = int(rng.integers(180, 256))
    val = int(rng.integers(225, 256))
    hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


# ---------------------------------------------------------------------------
# Step-by-step environment
# ---------------------------------------------------------------------------

class OccludedBouncingEnv(BaseEnv):
    """Single-episode env: physics always runs; the action picks revealed vs occluded render.

    State layout (per-step, returned by `step` and `hidden_state`): ``[x, y, vx, vy, curtain]``
    (float32). ``.color`` is the per-episode ball colour in native (BGR) order.
    """

    n_actions = 2  # 0 = curtain up (revealed), 1 = curtain down (occluded)

    def __init__(self, img_size: int = 64, radius: int = 10):
        self.img_size = img_size
        self.radius = radius
        self.rng = np.random.default_rng()
        self.bg = None
        self.color = None
        self.x = self.y = self.vx = self.vy = 0.0
        self.curtain = 0
        self.t = 0

    def reset(self, seed: int = 42) -> "OccludedBouncingEnv":
        self.rng = np.random.default_rng(seed)
        self.bg = make_background(self.rng, self.img_size)
        self.color = _bright_ball_color(self.rng)
        margin = self.radius + 1
        self.x = float(self.rng.uniform(margin, self.img_size - margin))
        self.y = float(self.rng.uniform(margin, self.img_size - margin))
        self.vx = float(self.rng.choice([-1, 1]) * self.rng.uniform(1.5, 3.0))
        self.vy = float(self.rng.choice([-1, 1]) * self.rng.uniform(1.5, 3.0))
        self.curtain = 0
        self.t = 0
        return self

    def _advance_physics(self) -> None:
        img_size, radius = self.img_size, self.radius
        self.x += self.vx
        self.y += self.vy
        if self.x - radius < 0:
            self.vx = abs(self.vx)
            self.x = float(radius)
        elif self.x + radius > img_size - 1:
            self.vx = -abs(self.vx)
            self.x = float(img_size - 1 - radius)
        if self.y - radius < 0:
            self.vy = abs(self.vy)
            self.y = float(radius)
        elif self.y + radius > img_size - 1:
            self.vy = -abs(self.vy)
            self.y = float(img_size - 1 - radius)

    def _render(self, action: int) -> np.ndarray:
        if action:
            return np.full((self.img_size, self.img_size, 3), CURTAIN_COLOR, dtype=np.uint8)
        frame = self.bg.copy()
        cv2.circle(
            frame,
            (int(round(self.x)), int(round(self.y))),
            self.radius,
            self.color,
            -1,
        )
        return frame

    def step(self, action: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame. Returns (frame uint8 HWC, state float32[5])."""
        self._advance_physics()
        self.curtain = int(action)
        frame = self._render(action)
        state = np.array(
            (self.x, self.y, self.vx, self.vy, float(action)), dtype=np.float32
        )
        self.t += 1
        return frame, state

    def hidden_state(self) -> np.ndarray:
        """Measurement-only: current [x, y, vx, vy, curtain] (float32). NEVER a model input."""
        return np.array((self.x, self.y, self.vx, self.vy, float(self.curtain)), dtype=np.float32)
