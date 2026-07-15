"""Closed-form frame readout for ColorField — pure numpy, identical on true and
predicted frames. The pixel layer is currently unsealed (see env.py header).

Provides:
- read_cells(frame, pos): per-cell color read for every cell overlapping the view
  at lattice position pos (extended cell indices — out-of-map tiles included).
- on-screen definition (Merlin): overlap >= 12px in x AND y <=> cell center inside
  the view (exact equivalence on this geometry: view top-left is odd, cell centers
  even, so center-offsets are odd and the boundary cases 0/64 never occur).
- border_bands(frame): width of the contiguous OUT-color band at each view edge —
  the closed-loop eval policies' only input besides their own action history.
- estimate_shift(prev, cur): integer content shift between consecutive frames, for
  the action-fidelity gate. cur[y, x] == prev[y + dy, x + dx] for a view that moved
  (dy, dx) world-px; a commanded action a implies (dy, dx) = 4 * DELTAS[a].
"""

from dataclasses import dataclass

import numpy as np

from .env import (CELL_EDGE_PX, CELL_PX, GRID_COLOR, OUT_IDX, PALETTE,
                  PITCH_PX, TL_OFFSET, VIEW_PX)

ON_SCREEN_MIN_OVERLAP = CELL_PX // 2
GRID_IDX = len(PALETTE)
LABEL_COLORS = np.concatenate([PALETTE, GRID_COLOR[None]], axis=0)


def view_tl(pos):
    """View top-left in world px for lattice position pos=(pr,pc)."""
    return (PITCH_PX * pos[0] + TL_OFFSET, PITCH_PX * pos[1] + TL_OFFSET)


def nearest_palette(rgb) -> int:
    """Index of the nearest of the 6 palette colors (Euclidean in RGB)."""
    d = ((PALETTE.astype(np.int64) - np.asarray(rgb, dtype=np.int64)) ** 2).sum(axis=1)
    return int(np.argmin(d))


def label_pixels(frame) -> np.ndarray:
    """(64, 64) nearest label: five cell colors, OUT, or black gridline."""
    f = frame.astype(np.int64)
    d = ((f[:, :, None, :] - LABEL_COLORS[None, None, :, :].astype(np.int64)) ** 2).sum(axis=-1)
    return np.argmin(d, axis=-1)


@dataclass
class CellRead:
    color: int          # nearest-palette index of the cell's visible-mean color
    ov_y: int           # visible extent in px along y (rows)
    ov_x: int           # visible extent in px along x (cols)
    on_screen: bool     # overlap >= half a cell on both axes (<=> center in view)
    mean_rgb: tuple     # mean RGB over the visible rectangle


def cells_in_view(pos):
    """Yield (ci, cj, y0, x0, ov_y, ov_x) for every cell with >= 1px overlap.
    ci/cj are EXTENDED cell indices (out-of-map tiles have indices outside
    [0, N_CELLS)); y0/x0 are the visible rectangle's top-left in view coords."""
    tly, tlx = view_tl(pos)
    for ci in range(tly // CELL_PX, (tly + VIEW_PX - 1) // CELL_PX + 1):
        y = ci * CELL_PX - tly
        y0, y1 = max(0, y), min(VIEW_PX, y + CELL_PX)
        ov_y = y1 - y0
        if ov_y <= 0:
            continue
        for cj in range(tlx // CELL_PX, (tlx + VIEW_PX - 1) // CELL_PX + 1):
            x = cj * CELL_PX - tlx
            x0, x1 = max(0, x), min(VIEW_PX, x + CELL_PX)
            ov_x = x1 - x0
            if ov_x <= 0:
                continue
            yield ci, cj, y0, x0, ov_y, ov_x


def read_cells(frame, pos):
    """{(ci, cj): CellRead} for every cell with >= 1px overlap at pos."""
    out = {}
    tly, tlx = view_tl(pos)
    for ci, cj, y0, x0, ov_y, ov_x in cells_in_view(pos):
        # Exclude the known black cell edges whenever interior pixels are visible.
        # A sliver consisting only of a gridline is never an on-screen read.
        cell_y = ci * CELL_PX - tly
        cell_x = cj * CELL_PX - tlx
        iy0 = max(y0, cell_y + CELL_EDGE_PX)
        iy1 = min(y0 + ov_y, cell_y + CELL_PX - CELL_EDGE_PX)
        ix0 = max(x0, cell_x + CELL_EDGE_PX)
        ix1 = min(x0 + ov_x, cell_x + CELL_PX - CELL_EDGE_PX)
        if iy1 > iy0 and ix1 > ix0:
            region = frame[iy0:iy1, ix0:ix1].reshape(-1, 3)
        else:
            region = frame[y0:y0 + ov_y, x0:x0 + ov_x].reshape(-1, 3)
        mean = region.mean(axis=0)
        out[(ci, cj)] = CellRead(
            color=nearest_palette(mean),
            ov_y=ov_y, ov_x=ov_x,
            on_screen=(ov_y >= ON_SCREEN_MIN_OVERLAP and ov_x >= ON_SCREEN_MIN_OVERLAP),
            mean_rgb=tuple(float(v) for v in mean),
        )
    return out


def border_bands(frame, out_frac: float = 0.9) -> dict:
    """Width in px of the contiguous near-OUT band at each view edge.

    A row/column belongs to the band if >= out_frac of its pixels label as OUT.
    Black gridlines have a distinct label. On real frames the OUT band width on
    a side is exactly max(0, 31 - 4p) for that
    axis's lattice coordinate p. On imagined frames it is whatever the model
    painted — which is exactly what the closed-loop policies must respect.
    Returns {"up": w, "down": w, "left": w, "right": w} (up = top edge)."""
    labels = label_pixels(frame)
    is_out = (labels == OUT_IDX)
    row_frac = is_out.mean(axis=1)   # per row
    col_frac = is_out.mean(axis=0)   # per column

    def run_len(fracs):
        w = 0
        for f in fracs:
            if f >= out_frac:
                w += 1
            else:
                break
        return w

    return {
        "up": run_len(row_frac),
        "down": run_len(row_frac[::-1]),
        "left": run_len(col_frac),
        "right": run_len(col_frac[::-1]),
    }


def estimate_shift(prev, cur, max_shift: int = PITCH_PX + 1):
    """Best integer (dy, dx) with cur[y, x] ~= prev[y + dy, x + dx], by min MSE
    over the overlap. Returns (dy, dx, mse). Ties resolve to the scan-order first
    candidate — near-uniform frames therefore read as an arbitrary shift and count
    as a fidelity mismatch, which is intended (a texture-free imagination cannot
    demonstrate action fidelity)."""
    p = prev.astype(np.int64)
    c = cur.astype(np.int64)
    best = None
    for dy in range(-max_shift, max_shift + 1):
        ys_c = slice(max(0, -dy), VIEW_PX - max(0, dy))
        ys_p = slice(max(0, dy), VIEW_PX - max(0, -dy))
        for dx in range(-max_shift, max_shift + 1):
            xs_c = slice(max(0, -dx), VIEW_PX - max(0, dx))
            xs_p = slice(max(0, dx), VIEW_PX - max(0, -dx))
            mse = float(((c[ys_c, xs_c] - p[ys_p, xs_p]) ** 2).mean())
            if best is None or mse < best[2]:
                best = (dy, dx, mse)
    return best
