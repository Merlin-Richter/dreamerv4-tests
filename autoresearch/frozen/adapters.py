"""Reference world-model adapters for the comeback eval. FROZEN LAYER.

The eval drives any world model through this interface:

    class Adapter:
        def begin(self, prefix_frames, prefix_actions) -> None
            # (P,64,64,3) uint8 RGB real frames, (P,) int actions, actions[0]=STAY.
            # Teacher-forced observation phase: commit these as real context.
        def step(self, action: int) -> np.ndarray
            # One imagination step; returns the imagined next frame (64,64,3) uint8.

adapter_factory(env) is called AFTER env.reset() for each episode; real models
must IGNORE env (no peeking) — it exists so the privileged baselines (oracle) can
read the ground truth. The frozen baselines here define the reference points:

  oracle             -> composite must be exactly 1.0 (gate test)
  perfect_imaginary  -> the canonical 'consistent liar': ignores the prefix,
                        renders its OWN persistent random world perfectly.
                        consistency = 1.0, real-anchored ~ chance. This is WHY the
                        composite is anchored 0.7 to ground truth.
  noise_cells        -> fresh random cell colors every frame (no memory at all)
  constant_color     -> the Goodhart cheat the gates must kill (all-one-color)
  copy_last          -> frozen frame; fails action fidelity, produces no comebacks
"""

import numpy as np

from .env import (CELL_EDGE_PX, CELL_PX, GRID_COLOR, PALETTE, STAY,
                  apply_action, build_world, render)
from .readout import cells_in_view, view_tl, VIEW_PX


class OracleAdapter:
    """Renders the TRUE world at the path-integral position. Privileged."""

    def __init__(self, env):
        self.world = build_world(env.map)
        self.pos = env.pos  # position at factory time == episode start

    def begin(self, prefix_frames, prefix_actions):
        for a in prefix_actions[1:]:
            self.pos = apply_action(self.pos, int(a), check=True)

    def step(self, action):
        # Bands on oracle frames are the true bands, so the closed-loop policy
        # never pushes the true border: check=True must hold.
        self.pos = apply_action(self.pos, int(action), check=True)
        return render(self.world, self.pos)


class _CellPainter:
    """Paint a view frame at an arbitrary (possibly off-lattice) position from a
    color_fn(cell)->palette_idx. Same geometry as the real renderer."""

    @staticmethod
    def paint(pos, color_fn):
        frame = np.empty((VIEW_PX, VIEW_PX, 3), dtype=np.uint8)
        for ci, cj, y0, x0, ov_y, ov_x in cells_in_view(pos):
            frame[y0:y0 + ov_y, x0:x0 + ov_x] = PALETTE[color_fn((ci, cj))]
        tly, tlx = view_tl(pos)
        ys = (tly + np.arange(VIEW_PX)) % CELL_PX
        xs = (tlx + np.arange(VIEW_PX)) % CELL_PX
        y_edge = (ys < CELL_EDGE_PX) | (ys >= CELL_PX - CELL_EDGE_PX)
        x_edge = (xs < CELL_EDGE_PX) | (xs >= CELL_PX - CELL_EDGE_PX)
        frame[y_edge, :, :] = GRID_COLOR
        frame[:, x_edge, :] = GRID_COLOR
        return frame


class PerfectImaginaryAdapter:
    """Consistent liar: a persistent, borderless, iid random world of its own —
    ignores the real prefix's CONTENT entirely. Perfect memory of the WRONG world.

    It starts from env.pos (privileged) so its 24px grid is phase-aligned with the
    tracker's registration grid. A real model gets this alignment for free from
    the prefix frames it conditions on; this baseline ignores them, hence the peek."""

    def __init__(self, env, seed=1234):
        self.rng = np.random.default_rng(seed)
        self.cells = {}
        self.pos = env.pos

    def _color(self, cell):
        if cell not in self.cells:
            self.cells[cell] = int(self.rng.integers(0, 5))
        return self.cells[cell]

    def begin(self, prefix_frames, prefix_actions):
        for a in prefix_actions[1:]:
            self.pos = apply_action(self.pos, int(a), check=False)

    def step(self, action):
        self.pos = apply_action(self.pos, int(action), check=False)
        return _CellPainter.paint(self.pos, self._color)


class NoiseCellsAdapter:
    """No memory at all: fresh iid cell colors every frame."""

    def __init__(self, env, seed=1234):
        self.rng = np.random.default_rng(seed)
        self.pos = (45, 45)

    def begin(self, prefix_frames, prefix_actions):
        pass

    def step(self, action):
        self.pos = apply_action(self.pos, int(action), check=False)
        return _CellPainter.paint(self.pos, lambda cell: int(self.rng.integers(0, 5)))


class ConstantColorAdapter:
    """The Goodhart cheat: a single flat color forever. Perfectly self-consistent
    — the entropy + fidelity gates must send it to 0.0."""

    def __init__(self, env, color=1):
        self.frame = np.empty((VIEW_PX, VIEW_PX, 3), dtype=np.uint8)
        self.frame[:] = PALETTE[color]

    def begin(self, prefix_frames, prefix_actions):
        pass

    def step(self, action):
        return self.frame.copy()


class CopyLastAdapter:
    """Returns the last real frame forever (the classic no-op baseline)."""

    def __init__(self, env):
        self.frame = None

    def begin(self, prefix_frames, prefix_actions):
        self.frame = prefix_frames[-1].copy()

    def step(self, action):
        return self.frame.copy()


BASELINES = {
    "oracle": OracleAdapter,
    "perfect_imaginary": PerfectImaginaryAdapter,
    "noise_cells": NoiseCellsAdapter,
    "constant_color": ConstantColorAdapter,
    "copy_last": CopyLastAdapter,
}


def make_adapter(name):
    """adapter_factory for a named frozen baseline."""
    cls = BASELINES[name]
    return lambda env: cls(env)
