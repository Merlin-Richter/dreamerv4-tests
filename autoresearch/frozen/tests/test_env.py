"""Gate tests: ColorField env geometry, rendering, invalid-action semantics."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen.env import (  # noqa: E402
    CELL_EDGE_PX, CELL_PX, GRID_COLOR, ColorFieldEnv, DELTAS, DOWN, LATTICE,
    LEFT, N_CELLS, OUT_IDX, PALETTE, PITCH_PX, RIGHT, STAY, TL_OFFSET, UP,
    VIEW_PX, WORLD_PX,
    apply_action, build_world, render, render_episode, sample_map,
    valid_actions)
from autoresearch.frozen.readout import border_bands, read_cells, view_tl  # noqa: E402


def reference_render(map_arr, pos):
    """Independent pixel-by-pixel renderer (no padding trick)."""
    tly, tlx = PITCH_PX * pos[0] + TL_OFFSET, PITCH_PX * pos[1] + TL_OFFSET
    frame = np.empty((VIEW_PX, VIEW_PX, 3), dtype=np.uint8)
    for y in range(VIEW_PX):
        for x in range(VIEW_PX):
            wy, wx = tly + y, tlx + x
            if 0 <= wy < WORLD_PX and 0 <= wx < WORLD_PX:
                edge_y = wy % CELL_PX
                edge_x = wx % CELL_PX
                if (edge_y < CELL_EDGE_PX or edge_y >= CELL_PX - CELL_EDGE_PX or
                        edge_x < CELL_EDGE_PX or edge_x >= CELL_PX - CELL_EDGE_PX):
                    frame[y, x] = GRID_COLOR
                else:
                    frame[y, x] = PALETTE[map_arr[wy // CELL_PX, wx // CELL_PX]]
            else:
                frame[y, x] = PALETTE[OUT_IDX]
    return frame


def test_geometry_constants():
    assert WORLD_PX == 360 and LATTICE == 90 and VIEW_PX == 64
    assert N_CELLS == 15 and CELL_PX == 24 and CELL_EDGE_PX == 1
    assert PITCH_PX == 4 and TL_OFFSET == -31


def test_render_matches_reference():
    rng = np.random.default_rng(0)
    m = sample_map(rng)
    world = build_world(m)
    positions = [(0, 0), (89, 89), (0, 89), (89, 0), (45, 45), (15, 16), (16, 15)]
    positions += [tuple(rng.integers(0, LATTICE, 2)) for _ in range(12)]
    for pos in positions:
        assert np.array_equal(render(world, pos), reference_render(m, pos)), pos


def test_band_widths_exact_on_real_frames():
    rng = np.random.default_rng(1)
    m = sample_map(rng)
    world = build_world(m)
    for _ in range(25):
        pos = tuple(int(v) for v in rng.integers(0, LATTICE, 2))
        b = border_bands(render(world, pos))
        tly, tlx = view_tl(pos)
        assert b["up"] == max(0, -tly), (pos, b)
        assert b["left"] == max(0, -tlx), (pos, b)
        assert b["down"] == max(0, tly + VIEW_PX - WORLD_PX), (pos, b)
        assert b["right"] == max(0, tlx + VIEW_PX - WORLD_PX), (pos, b)


def test_invalid_action_semantics():
    env = ColorFieldEnv()
    env.reset(seed=3, start=(0, 0))
    assert set(env.valid_actions()) == {STAY, DOWN, RIGHT}
    for bad in (UP, LEFT):
        try:
            env.step(bad)
            assert False, "invalid action did not raise"
        except ValueError:
            pass
    assert env.pos == (0, 0)  # a failed try must not move anything
    env.reset(seed=3, start=(89, 89))
    assert set(env.valid_actions()) == {STAY, UP, LEFT}
    env.reset(seed=3, start=(45, 0))
    assert set(env.valid_actions()) == {STAY, UP, DOWN, RIGHT}
    env.reset(seed=3, start=(45, 45))
    assert set(env.valid_actions()) == {STAY, UP, DOWN, LEFT, RIGHT}


def test_on_screen_iff_center_in_view():
    rng = np.random.default_rng(2)
    m = sample_map(rng)
    world = build_world(m)
    for _ in range(15):
        pos = tuple(int(v) for v in rng.integers(0, LATTICE, 2))
        tly, tlx = view_tl(pos)
        reads = read_cells(render(world, pos), pos)
        for (ci, cj), r in reads.items():
            center_in = (0 < ci * CELL_PX + CELL_PX // 2 - tly < VIEW_PX) and \
                        (0 < cj * CELL_PX + CELL_PX // 2 - tlx < VIEW_PX)
            assert r.on_screen == center_in, (pos, ci, cj, r)


def test_render_episode_matches_env_steps():
    env = ColorFieldEnv()
    f0 = env.reset(seed=7, start=(40, 40))
    rng = np.random.default_rng(7)
    actions = [STAY]
    frames = [f0]
    for _ in range(60):
        a = int(rng.choice(env.valid_actions()))
        frames.append(env.step(a))
        actions.append(a)
    proc = render_episode(env.map, (40, 40), np.array(actions))
    assert np.array_equal(proc, np.stack(frames))


def test_determinism():
    e1, e2 = ColorFieldEnv(), ColorFieldEnv()
    f1, f2 = e1.reset(seed=11), e2.reset(seed=11)
    assert np.array_equal(f1, f2) and np.array_equal(e1.map, e2.map) and e1.pos == e2.pos


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("test_env: ALL PASS")
