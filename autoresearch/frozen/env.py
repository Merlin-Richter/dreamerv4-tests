"""ColorField environment — the frozen memory env of the autoresearch harness.

FROZEN LAYER: once MANIFEST.json records this file's hash, do not edit — the driver
scores any run against a tampered frozen layer as chance.

Design (Merlin, 2026-07-06; spec: tasks/*/colorfield-env-and-eval.md):
- 15x15 cell map, cell = 12px -> 180x180 px world. Cell colors iid uniform over a
  5-color palette; everything outside the map is a fixed 6th color (OUT).
- Egocentric 64x64 RGB uint8 view. The view position p = (pr, pc) lives on a 90x90
  sub-cell lattice (2px pitch = 1/6 cell): pr, pc in [0, 89].
  View top-left in world coords = 2*p - 31, so an OUT band of width 31 - 2*p px is
  visible on a side whenever p <= 15 on that axis — the only absolute-position
  landmark. Band widths on real frames are always odd: {1, 3, ..., 31}.
- Actions: 0=up 1=down 2=left 3=right (one lattice step = 2px), 4=stay.
  INVALID-ACTION SEMANTICS (Merlin): an outward move at the lattice edge is NOT a
  valid action — it cannot even be tried; step() raises. It therefore NEVER occurs
  in any dataset (policies must sample from valid_actions()).
- Fully deterministic given (map, start, action stream).
- hidden_state() is measurement-only — never a model input.

RGB end-to-end inside autoresearch/ (self-contained; convert only at cv2 display).
"""

import numpy as np

# --- Geometry ---------------------------------------------------------------
N_CELLS = 15                       # map is N_CELLS x N_CELLS cells
CELL_PX = 12                       # square cell side in px
WORLD_PX = N_CELLS * CELL_PX       # 180
VIEW_PX = 64                       # egocentric view side in px
PITCH_PX = 2                       # lattice pitch = 1/6 cell
LATTICE = N_CELLS * CELL_PX // PITCH_PX   # 90 positions per axis, p in [0, LATTICE-1]
TL_OFFSET = -31                    # view top-left (world px) = PITCH_PX * p + TL_OFFSET
PAD_PX = -TL_OFFSET                # 31px OUT padding on every side of the world image
MAX_BAND_PX = PAD_PX               # widest possible OUT band on a real frame (at p=0)

# --- Colors (RGB uint8) ------------------------------------------------------
N_COLORS = 5                       # in-map palette size
OUT_IDX = 5                        # the 6th color: outside the map
PALETTE = np.array([
    [230,  40,  40],   # 0 red
    [ 60, 200,  60],   # 1 green
    [ 60,  90, 235],   # 2 blue
    [245, 160,  30],   # 3 orange
    [160,  60, 220],   # 4 purple
    [ 45,  45,  45],   # 5 OUT (outside the map)
], dtype=np.uint8)

# --- Actions ------------------------------------------------------------------
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
N_ACTIONS = 5
ACTION_NAMES = ("up", "down", "left", "right", "stay")
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), STAY: (0, 0)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT, STAY: STAY}


def sample_map(rng: np.random.Generator) -> np.ndarray:
    """iid uniform cell colors, shape (N_CELLS, N_CELLS) uint8 in [0, N_COLORS)."""
    return rng.integers(0, N_COLORS, size=(N_CELLS, N_CELLS), dtype=np.uint8)


def build_world(map_arr: np.ndarray) -> np.ndarray:
    """Padded world image, shape (242, 242, 3) uint8 RGB.

    The map occupies [PAD_PX : PAD_PX + WORLD_PX) on both axes; everything else is
    the OUT color. The view at position p is the pure slice
    world[2*pr : 2*pr + 64, 2*pc : 2*pc + 64].
    """
    assert map_arr.shape == (N_CELLS, N_CELLS)
    size = WORLD_PX + 2 * PAD_PX  # 242
    world = np.empty((size, size, 3), dtype=np.uint8)
    world[:] = PALETTE[OUT_IDX]
    tiles = PALETTE[map_arr]                                  # (15, 15, 3)
    tiles = np.repeat(np.repeat(tiles, CELL_PX, axis=0), CELL_PX, axis=1)
    world[PAD_PX:PAD_PX + WORLD_PX, PAD_PX:PAD_PX + WORLD_PX] = tiles
    return world


def render(world: np.ndarray, pos) -> np.ndarray:
    """View at lattice position pos=(pr,pc) from a build_world() image. Copy."""
    pr, pc = pos
    if not (0 <= pr < LATTICE and 0 <= pc < LATTICE):
        raise ValueError(f"position {pos} off the {LATTICE}x{LATTICE} lattice")
    y, x = PITCH_PX * pr, PITCH_PX * pc
    return world[y:y + VIEW_PX, x:x + VIEW_PX].copy()


def valid_actions(pos):
    """The valid action set at pos. STAY is always valid; outward-at-edge is not."""
    pr, pc = pos
    acts = [STAY]
    if pr > 0:
        acts.append(UP)
    if pr < LATTICE - 1:
        acts.append(DOWN)
    if pc > 0:
        acts.append(LEFT)
    if pc < LATTICE - 1:
        acts.append(RIGHT)
    return acts


def apply_action(pos, action, check: bool = True):
    """pos after action. check=True enforces lattice bounds (the env contract).
    check=False is for eval-time path-integral registration in IMAGINATION, where
    the position may legitimately leave the true lattice."""
    dr, dc = DELTAS[action]
    npos = (pos[0] + dr, pos[1] + dc)
    if check and not (0 <= npos[0] < LATTICE and 0 <= npos[1] < LATTICE):
        raise ValueError(
            f"invalid action {ACTION_NAMES[action]} at {pos}: outward moves at the "
            "lattice edge are not valid actions (cannot be tried)")
    return npos


def positions_from(start, actions, check: bool = True):
    """Path-integral positions, shape (len(actions)+... ) — one per FRAME:
    positions[0] = start (actions[0] must be STAY by dataset convention),
    positions[t] = positions[t-1] + delta(actions[t])."""
    pos = tuple(start)
    out = [pos]
    for a in actions[1:]:
        pos = apply_action(pos, int(a), check=check)
        out.append(pos)
    return np.array(out, dtype=np.int64)


def render_episode(map_arr, start, actions, check: bool = True) -> np.ndarray:
    """Procedural episode renderer: frames (T, 64, 64, 3) uint8 RGB, where
    T = len(actions), frames[0] = render(start), frames[t] = after actions[t].
    Dataset convention: actions[0] == STAY (the action 'producing' frame 0)."""
    world = build_world(map_arr)
    pos_list = positions_from(start, actions, check=check)
    frames = np.empty((len(actions), VIEW_PX, VIEW_PX, 3), dtype=np.uint8)
    for t, pos in enumerate(pos_list):
        frames[t] = render(world, pos)
    return frames


class ColorFieldEnv:
    """Stateful wrapper. Deterministic given (seed | map_arr, start, actions)."""

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)
        self.map = None
        self.world = None
        self.pos = None
        self.t = 0

    def reset(self, seed=None, map_arr=None, start=None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.map = sample_map(self._rng) if map_arr is None else np.asarray(map_arr, dtype=np.uint8)
        self.world = build_world(self.map)
        if start is None:
            start = (int(self._rng.integers(0, LATTICE)), int(self._rng.integers(0, LATTICE)))
        self.pos = tuple(int(v) for v in start)
        if not (0 <= self.pos[0] < LATTICE and 0 <= self.pos[1] < LATTICE):
            raise ValueError(f"start {start} off the lattice")
        self.t = 0
        return render(self.world, self.pos)

    def valid_actions(self):
        return valid_actions(self.pos)

    def step(self, action: int) -> np.ndarray:
        """Apply a VALID action; raises ValueError on an invalid one (outward at
        the lattice edge — such an action cannot be tried, it is not a no-op)."""
        self.pos = apply_action(self.pos, int(action), check=True)
        self.t += 1
        return render(self.world, self.pos)

    def hidden_state(self):
        """Measurement-only (eval/oracle use). NEVER a model input."""
        return {"map": self.map.copy(), "pos": tuple(self.pos), "t": self.t}
