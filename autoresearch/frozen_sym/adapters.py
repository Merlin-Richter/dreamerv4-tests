"""Reference world-model adapters for the ColorField-SYM comeback eval.
FROZEN-LAYER-sym (see env.py header; spec:
tasks/in-progress/colorfield-sym-frozen-layer.md).

The eval drives any world model through this interface:

    class Adapter:
        def begin(self, prefix_grids, prefix_actions) -> None
            # (P,5,5) uint8 symbolic grids, (P,) int per-TICK actions,
            # actions[0]=STAY, off-phase ticks STAY. Teacher-forced
            # observation phase: commit these as real context.
        def step(self, action: int) -> np.ndarray
            # One imagination TICK; returns the imagined next grid (5,5) uint8.
            # The eval loop enforces the phase rule (STAY off-phase).

adapter_factory(env) is called AFTER env.reset() for each episode; real models
must IGNORE env (no peeking) — it exists so the privileged baselines (oracle)
can read the ground truth. The frozen baselines here define the reference
points:

  oracle             -> composite must be exactly 1.0 (gate test)
  perfect_imaginary  -> the canonical 'consistent liar': ignores the prefix
                        CONTENT, renders its OWN persistent random borderless
                        cell world perfectly. consistency = 1.0, real-anchored
                        ~ chance. This is WHY the composite is anchored 0.7 to
                        ground truth. Unlike the pixel tier it needs NO
                        privileged position peek: symbols have no sub-cell
                        grid phase — it integrates the same action stream as
                        the tracker, so its world is a constant translation of
                        the tracker's registration from ANY internal origin.
  noise_cells        -> fresh random cell colors every TICK (no memory at
                        all); fails the off-phase-unchanged fidelity gate
  constant_color     -> the Goodhart cheat: all-one-color forever. In the sym
                        tier a uniform grid is shift-invariant so it PASSES
                        fidelity — the entropy gate is what sends it to 0.0
  copy_last          -> frozen grid; fails fidelity on phase-0 moves (~80%
                        of ticks are off-phase and pass free, < the 0.90 bar)
"""

import numpy as np

from .env import (N_COLORS, VIEW_CELLS, VIEW_HALF, apply_action, render_grid)


def paint_grid(pos, color_fn):
    """Paint a (5,5) viewport at an arbitrary (possibly off-board) center from
    a color_fn(cell)->palette_id. Same geometry as the real renderer."""
    g = np.empty((VIEW_CELLS, VIEW_CELLS), dtype=np.uint8)
    for i in range(VIEW_CELLS):
        for j in range(VIEW_CELLS):
            g[i, j] = color_fn((pos[0] - VIEW_HALF + i, pos[1] - VIEW_HALF + j))
    return g


class OracleAdapter:
    """Renders the TRUE board at the path-integral center. Privileged."""

    def __init__(self, env):
        self.map = env.map.copy()
        self.pos = env.pos  # position at factory time == episode start

    def begin(self, prefix_grids, prefix_actions):
        for a in prefix_actions[1:]:
            # off-phase actions are STAY (no-op delta); moves are phase-0 only,
            # so plain per-tick integration is exact.
            self.pos = apply_action(self.pos, int(a), check=True)

    def step(self, action):
        # Bands on oracle grids are the true bands, so the closed-loop policy
        # never pushes the true border: check=True must hold.
        self.pos = apply_action(self.pos, int(action), check=True)
        return render_grid(self.map, self.pos)


class PerfectImaginaryAdapter:
    """Consistent liar: a persistent, BORDERLESS, iid random cell world of its
    own — ignores the real prefix's CONTENT entirely. Perfect memory of the
    WRONG world. Needs no env peek (see module header); env is accepted and
    ignored for factory-signature parity."""

    def __init__(self, env, seed=1234):
        self.rng = np.random.default_rng(seed)
        self.cells = {}
        self.pos = (0, 0)   # arbitrary origin: consistency is translation-invariant

    def _color(self, cell):
        if cell not in self.cells:
            self.cells[cell] = int(self.rng.integers(0, N_COLORS))
        return self.cells[cell]

    def begin(self, prefix_grids, prefix_actions):
        for a in prefix_actions[1:]:
            self.pos = apply_action(self.pos, int(a), check=False)

    def step(self, action):
        self.pos = apply_action(self.pos, int(action), check=False)
        return paint_grid(self.pos, self._color)


class NoiseCellsAdapter:
    """No memory at all: a fresh iid random grid every tick. Violates the
    off-phase-unchanged fidelity gate almost surely -> gated to 0."""

    def __init__(self, env, seed=1234):
        self.rng = np.random.default_rng(seed)

    def begin(self, prefix_grids, prefix_actions):
        pass

    def step(self, action):
        return self.rng.integers(0, N_COLORS, size=(VIEW_CELLS, VIEW_CELLS),
                                 dtype=np.uint8)


class ConstantColorAdapter:
    """The Goodhart cheat: a single flat color forever. Perfectly
    self-consistent AND (sym-tier) perfectly shift-fidelitous — the entropy
    gate must send it to 0.0."""

    def __init__(self, env, color=1):
        self.grid = np.full((VIEW_CELLS, VIEW_CELLS), color, dtype=np.uint8)

    def begin(self, prefix_grids, prefix_actions):
        pass

    def step(self, action):
        return self.grid.copy()


class CopyLastAdapter:
    """Returns the last real grid forever (the classic no-op baseline)."""

    def __init__(self, env):
        self.grid = None

    def begin(self, prefix_grids, prefix_actions):
        self.grid = prefix_grids[-1].copy()

    def step(self, action):
        return self.grid.copy()


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
