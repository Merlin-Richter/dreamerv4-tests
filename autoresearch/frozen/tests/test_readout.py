"""Gate tests: readout exactness + shift estimation on real frames."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen.env import (  # noqa: E402
    CELL_PX, DELTAS, LATTICE, N_CELLS, OUT_IDX, PALETTE, PITCH_PX, STAY,
    apply_action, build_world, render, sample_map, valid_actions)
from autoresearch.frozen.readout import (  # noqa: E402
    LABEL_COLORS, estimate_shift, label_pixels, nearest_palette, read_cells)


def gt_color(m, ci, cj):
    return int(m[ci, cj]) if 0 <= ci < N_CELLS and 0 <= cj < N_CELLS else OUT_IDX


def test_nearest_palette_identity():
    for i, c in enumerate(PALETTE):
        assert nearest_palette(c) == i


def test_read_cells_exact_on_real_frames():
    rng = np.random.default_rng(0)
    m = sample_map(rng)
    world = build_world(m)
    positions = [(0, 0), (89, 89), (0, 45), (45, 89), (44, 44)]
    positions += [tuple(int(v) for v in rng.integers(0, LATTICE, 2)) for _ in range(15)]
    n_out_seen = 0
    for pos in positions:
        frame = render(world, pos)
        for (ci, cj), r in read_cells(frame, pos).items():
            if r.on_screen:
                assert r.color == gt_color(m, ci, cj), (pos, ci, cj, r)
                n_out_seen += r.color == OUT_IDX
    assert n_out_seen > 0  # corner positions must include OUT tiles


def test_label_pixels_exact():
    rng = np.random.default_rng(1)
    m = sample_map(rng)
    world = build_world(m)
    frame = render(world, (3, 3))  # corner: includes OUT band
    labels = label_pixels(frame)
    for y in range(0, 64, 7):
        for x in range(0, 64, 7):
            assert np.array_equal(LABEL_COLORS[labels[y, x]], frame[y, x])


def test_estimate_shift_matches_actions():
    rng = np.random.default_rng(2)
    m = sample_map(rng)
    world = build_world(m)
    pos = (40, 40)
    for _ in range(40):
        a = int(rng.choice(valid_actions(pos)))
        npos = apply_action(pos, a)
        dy, dx, mse = estimate_shift(render(world, pos), render(world, npos))
        assert (dy, dx) == (PITCH_PX * DELTAS[a][0],
                            PITCH_PX * DELTAS[a][1]), (pos, a, dy, dx)
        assert mse == 0.0
        pos = npos


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("test_readout: ALL PASS")
