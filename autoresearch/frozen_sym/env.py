"""ColorField-SYM environment — the symbolic (tokenizer-free) frozen memory env
of the autoresearch harness. FROZEN-LAYER-sym: once MANIFEST-sym records this
file's hash, do not edit — the driver scores any run against a tampered frozen
layer as chance.

Design (Merlin, 2026-07-07; spec: tasks/in-progress/colorfield-sym-frozen-layer.md):
the pixel tier's 20-min budgets died learning the APPEARANCE prior, so this tier
removes pixels entirely. Same 15x15 board of iid cells over the same 5-color
palette (ids shared with the pixel tier; OUT_IDX=5 outside the board), but the
observation is SYMBOLIC: a (5,5) uint8 grid of palette ids — the viewport c+/-2
around a center c in [0,14]^2, out-of-board cells reading OUT — plus a phase
int. Time is DILATED 5x (phase-5, Merlin): the move in actions[t] applies only
at ticks with t % 5 == 0; on off-phase ticks the env FORCES STAY —
valid_actions() == [STAY], anything else raises — the same uniform
invalid-action semantics as outward-at-board-edge at phase 0 (which also
raises: it cannot even be tried, so it never occurs in any dataset). W=16
ticks then spans only 3.2 effective moves while 25/225 = 11% of the board is
visible. Fully deterministic given (map, start, action stream); hidden_state()
is measurement-only — never a model input.

Action/tick convention (mirrors the pixel dataset convention 1:1): actions[t]
produces tick t, actions[0] == STAY; obs[t] = (grid at pos[t], phase = t % 5).
The spec's fidelity gate anchors this reading: AT PHASE-0 ticks the grid equals
the previous grid shifted by the action; off-phase it is unchanged.

Shared constants (palette ids, OUT_IDX, action ids) are IMPORTED from the
sealed pixel tier (autoresearch.frozen.env — read-only), never duplicated.
"""

import numpy as np

from autoresearch.frozen.env import (  # sealed pixel-tier constants — read-only reuse
    ACTION_NAMES, DELTAS, DOWN, LEFT, N_ACTIONS, N_CELLS, N_COLORS, OPPOSITE,
    OUT_IDX, PALETTE, RIGHT, STAY, UP, sample_map)

__all__ = [
    "ACTION_NAMES", "BOARD", "DELTAS", "DOWN", "LEFT", "N_ACTIONS", "N_CELLS",
    "N_COLORS", "OPPOSITE", "OUT_IDX", "PALETTE", "PHASE_PERIOD", "RIGHT",
    "STAY", "UP", "VIEW_CELLS", "VIEW_HALF", "ColorFieldSymEnv", "apply_action",
    "out_bands", "positions_from", "render_episode", "render_grid",
    "sample_map", "spatial_valid_actions", "valid_actions"]

# --- Geometry / time ----------------------------------------------------------
BOARD = N_CELLS          # 15 — the center lattice IS the cell grid, c in [0,14]^2
VIEW_CELLS = 5           # viewport side in cells
VIEW_HALF = 2            # viewport = center +/- VIEW_HALF
PHASE_PERIOD = 5         # moves apply only at ticks with t % PHASE_PERIOD == 0


def render_grid(map_arr, pos) -> np.ndarray:
    """Symbolic viewport at center pos=(r,c): (5,5) uint8 of palette ids, cells
    outside the board reading OUT_IDX. Accepts ANY integer center (extended
    coords) — in imagination the path-integral center may leave the board."""
    r, c = int(pos[0]), int(pos[1])
    g = np.full((VIEW_CELLS, VIEW_CELLS), OUT_IDX, dtype=np.uint8)
    r0, c0 = r - VIEW_HALF, c - VIEW_HALF
    lo_r, hi_r = max(r0, 0), min(r0 + VIEW_CELLS, BOARD)
    lo_c, hi_c = max(c0, 0), min(c0 + VIEW_CELLS, BOARD)
    if lo_r < hi_r and lo_c < hi_c:
        g[lo_r - r0:hi_r - r0, lo_c - c0:hi_c - c0] = map_arr[lo_r:hi_r, lo_c:hi_c]
    return g


def out_bands(grid) -> dict:
    """Width (in cells) of the contiguous fully-OUT band at each viewport edge —
    the closed-loop eval policies' only input besides their own action history.
    A line belongs to the band iff ALL its 5 cells are OUT (the exact analogue
    of the pixel tier's >=90%-OUT rows). On real grids bands are in {0,1,2} and
    band == 2 <=> center at the board edge <=> outward invalid; imagined grids
    may paint wider bands (up to 5) — the policies block on band >= 2."""
    is_out = (np.asarray(grid) == OUT_IDX)
    rows = is_out.all(axis=1)
    cols = is_out.all(axis=0)

    def run_len(v):
        w = 0
        for x in v:
            if x:
                w += 1
            else:
                break
        return w

    return {"up": run_len(rows), "down": run_len(rows[::-1]),
            "left": run_len(cols), "right": run_len(cols[::-1])}


def spatial_valid_actions(pos):
    """The PHASE-0 valid action set at pos: STAY always; outward-at-edge never."""
    r, c = pos
    acts = [STAY]
    if r > 0:
        acts.append(UP)
    if r < BOARD - 1:
        acts.append(DOWN)
    if c > 0:
        acts.append(LEFT)
    if c < BOARD - 1:
        acts.append(RIGHT)
    return acts


def valid_actions(pos, phase):
    """Valid action set for an action landing on a tick with the given phase:
    off-phase (phase % 5 != 0) it is exactly [STAY] (forced-STAY semantics);
    at phase 0 it is the spatial set (outward-at-edge is not a valid action)."""
    if phase % PHASE_PERIOD != 0:
        return [STAY]
    return spatial_valid_actions(pos)


def apply_action(pos, action, check: bool = True):
    """pos after a SPATIAL action (phase handling is the caller's: env.step /
    positions_from force STAY off-phase). check=True enforces board bounds (the
    env contract — outward at the edge raises, it cannot be tried). check=False
    is for eval-time path-integral registration in IMAGINATION, where the
    center may legitimately leave the true board."""
    dr, dc = DELTAS[action]
    npos = (pos[0] + dr, pos[1] + dc)
    if check and not (0 <= npos[0] < BOARD and 0 <= npos[1] < BOARD):
        raise ValueError(
            f"invalid action {ACTION_NAMES[action]} at {pos}: outward moves at "
            "the board edge are not valid actions (cannot be tried)")
    return npos


def positions_from(start, actions, check: bool = True):
    """Path-integral center positions, one per TICK: positions[0] = start
    (actions[0] must be STAY by dataset convention), positions[t] follows
    actions[t] under the phase-5 rule — the move applies only when t % 5 == 0;
    off-phase ticks NEVER move (env physics), and with check=True an off-phase
    non-STAY action raises (uniform invalid-action semantics). check=False
    additionally allows the center to leave the board (imagination)."""
    pos = tuple(int(v) for v in start)
    out = [pos]
    for t in range(1, len(actions)):
        a = int(actions[t])
        if t % PHASE_PERIOD != 0:
            if check and a != STAY:
                raise ValueError(
                    f"invalid action {ACTION_NAMES[a]} at off-phase tick {t}: "
                    "off-phase valid actions are exactly [STAY]")
        else:
            pos = apply_action(pos, a, check=check)
        out.append(pos)
    return np.array(out, dtype=np.int64)


def render_episode(map_arr, start, actions, check: bool = True) -> np.ndarray:
    """Procedural episode renderer: grids (T, 5, 5) uint8, T = len(actions),
    grids[0] = render_grid(start), grids[t] = after actions[t]. Phases are
    implicit (phase[t] = t % 5). Dataset convention: actions[0] == STAY."""
    pos_list = positions_from(start, actions, check=check)
    grids = np.empty((len(actions), VIEW_CELLS, VIEW_CELLS), dtype=np.uint8)
    for t, pos in enumerate(pos_list):
        grids[t] = render_grid(map_arr, pos)
    return grids


class ColorFieldSymEnv:
    """Stateful wrapper. Deterministic given (seed | map_arr, start, actions).
    reset()/step() return (grid, phase) — the per-tick symbolic observation."""

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)
        self.map = None
        self.pos = None
        self.t = 0

    @property
    def phase(self):
        return self.t % PHASE_PERIOD

    def reset(self, seed=None, map_arr=None, start=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.map = sample_map(self._rng) if map_arr is None else np.asarray(map_arr, dtype=np.uint8)
        assert self.map.shape == (BOARD, BOARD)
        if start is None:
            start = (int(self._rng.integers(0, BOARD)), int(self._rng.integers(0, BOARD)))
        self.pos = tuple(int(v) for v in start)
        if not (0 <= self.pos[0] < BOARD and 0 <= self.pos[1] < BOARD):
            raise ValueError(f"start {start} off the {BOARD}x{BOARD} board")
        self.t = 0
        return render_grid(self.map, self.pos), 0

    def valid_actions(self):
        """Valid set for the NEXT step() call — the action producing tick t+1.
        Off-phase ((t+1) % 5 != 0) this is exactly [STAY]."""
        return valid_actions(self.pos, self.t + 1)

    def step(self, action):
        """Apply a VALID action producing tick t+1; raises ValueError on an
        invalid one (non-STAY at an off-phase tick, or outward at the board
        edge at a phase-0 tick — such actions cannot be tried, they are not
        no-ops). A failed step mutates nothing."""
        a = int(action)
        new_t = self.t + 1
        if new_t % PHASE_PERIOD != 0:
            if a != STAY:
                raise ValueError(
                    f"invalid action {ACTION_NAMES[a]} at off-phase tick {new_t}: "
                    "off-phase valid actions are exactly [STAY]")
        else:
            self.pos = apply_action(self.pos, a, check=True)
        self.t = new_t
        return render_grid(self.map, self.pos), new_t % PHASE_PERIOD

    def hidden_state(self):
        """Measurement-only (eval/oracle use). NEVER a model input."""
        return {"map": self.map.copy(), "pos": tuple(self.pos),
                "t": self.t, "phase": self.phase}
