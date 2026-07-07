"""Gate tests: ColorField-SYM env — viewport geometry vs an independent
brute-force extraction (incl. corners and off-board centers), phase-5
forced-STAY semantics, uniform invalid-action semantics (raises, no mutation),
determinism, procedural render == stepping. FROZEN-LAYER-sym test; spec:
tasks/in-progress/colorfield-sym-frozen-layer.md."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen import env as pixel_env  # noqa: E402
from autoresearch.frozen_sym.env import (  # noqa: E402
    BOARD, ColorFieldSymEnv, DOWN, LEFT, N_COLORS, OUT_IDX, PHASE_PERIOD,
    RIGHT, STAY, UP, VIEW_CELLS, VIEW_HALF, out_bands, positions_from,
    render_episode, render_grid, sample_map, spatial_valid_actions)


def reference_grid(map_arr, pos):
    """Independent cell-by-cell viewport extraction (no slicing trick)."""
    g = np.empty((VIEW_CELLS, VIEW_CELLS), dtype=np.uint8)
    for i in range(VIEW_CELLS):
        for j in range(VIEW_CELLS):
            r, c = pos[0] - VIEW_HALF + i, pos[1] - VIEW_HALF + j
            g[i, j] = map_arr[r, c] if (0 <= r < BOARD and 0 <= c < BOARD) else OUT_IDX
    return g


def test_geometry_constants_shared_with_pixel_tier():
    assert BOARD == 15 and VIEW_CELLS == 5 and VIEW_HALF == 2 and PHASE_PERIOD == 5
    assert N_COLORS == 5 and OUT_IDX == 5
    # shared ids/constants must literally BE the pixel tier's (imported, not copied)
    assert BOARD == pixel_env.N_CELLS and OUT_IDX == pixel_env.OUT_IDX
    assert (UP, DOWN, LEFT, RIGHT, STAY) == (pixel_env.UP, pixel_env.DOWN,
                                             pixel_env.LEFT, pixel_env.RIGHT,
                                             pixel_env.STAY)


def test_render_matches_reference():
    rng = np.random.default_rng(0)
    m = sample_map(rng)
    positions = [(0, 0), (14, 14), (0, 14), (14, 0), (7, 7), (2, 2), (1, 13), (12, 1)]
    positions += [tuple(rng.integers(0, BOARD, 2)) for _ in range(12)]
    # extended (off-board) centers — legal in imagination registration
    positions += [(-1, 7), (7, -3), (16, 7), (7, 20), (-2, -2), (20, 20), (-5, 7)]
    for pos in positions:
        assert np.array_equal(render_grid(m, pos), reference_grid(m, pos)), pos


def test_out_bands_exact_on_real_grids():
    rng = np.random.default_rng(1)
    m = sample_map(rng)
    for _ in range(40):
        pos = tuple(int(v) for v in rng.integers(0, BOARD, 2))
        b = out_bands(render_grid(m, pos))
        assert b["up"] == max(0, VIEW_HALF - pos[0]), (pos, b)
        assert b["down"] == max(0, pos[0] - (BOARD - 1 - VIEW_HALF)), (pos, b)
        assert b["left"] == max(0, VIEW_HALF - pos[1]), (pos, b)
        assert b["right"] == max(0, pos[1] - (BOARD - 1 - VIEW_HALF)), (pos, b)
        assert all(0 <= w <= 2 for w in b.values())  # real-grid bands are {0,1,2}
    # fully-OUT grid (possible only off-board in imagination): bands maxed
    g = np.full((VIEW_CELLS, VIEW_CELLS), OUT_IDX, dtype=np.uint8)
    assert all(w == VIEW_CELLS for w in out_bands(g).values())


def test_phase_forced_stay_semantics():
    env = ColorFieldSymEnv()
    grid, phase = env.reset(seed=3, start=(7, 7))
    assert phase == 0 and env.t == 0
    # ticks 1..4 are off-phase: valid_actions == [STAY], non-STAY raises, no mutation
    for t in range(1, PHASE_PERIOD):
        assert env.valid_actions() == [STAY], t
        for bad in (UP, DOWN, LEFT, RIGHT):
            try:
                env.step(bad)
                assert False, "off-phase non-STAY did not raise"
            except ValueError:
                pass
        assert env.pos == (7, 7) and env.t == t - 1  # failed step mutates nothing
        g, p = env.step(STAY)
        assert p == t % PHASE_PERIOD and env.pos == (7, 7)
    # tick 5 is phase-0: full spatial set, a move applies
    assert set(env.valid_actions()) == {STAY, UP, DOWN, LEFT, RIGHT}
    g, p = env.step(RIGHT)
    assert p == 0 and env.pos == (7, 8)
    # and the next 4 ticks are forced STAY again
    assert env.valid_actions() == [STAY]


def test_invalid_action_semantics_at_phase0():
    env = ColorFieldSymEnv()
    for start, valid, invalid in [
        ((0, 0), {STAY, DOWN, RIGHT}, (UP, LEFT)),
        ((14, 14), {STAY, UP, LEFT}, (DOWN, RIGHT)),
        ((7, 0), {STAY, UP, DOWN, RIGHT}, (LEFT,)),
        ((7, 7), {STAY, UP, DOWN, LEFT, RIGHT}, ()),
    ]:
        env.reset(seed=3, start=start)
        for _ in range(PHASE_PERIOD - 1):
            env.step(STAY)                 # advance so the next tick is phase-0
        assert set(env.valid_actions()) == valid, start
        assert set(spatial_valid_actions(start)) == valid, start
        for bad in invalid:
            try:
                env.step(bad)
                assert False, "invalid action did not raise"
            except ValueError:
                pass
            assert env.pos == start and env.t == PHASE_PERIOD - 1  # no mutation


def test_positions_from_phase_discipline():
    # off-phase non-STAY: raises with check=True ...
    actions = np.array([STAY, STAY, RIGHT, STAY, STAY, RIGHT], dtype=np.uint8)
    try:
        positions_from((7, 7), actions, check=True)
        assert False, "off-phase non-STAY did not raise"
    except ValueError:
        pass
    # ... and NEVER moves, even with check=False (env physics)
    pos = positions_from((7, 7), actions, check=False)
    assert [tuple(p) for p in pos] == [(7, 7)] * 5 + [(7, 8)]
    # check=False also allows leaving the board at phase-0
    acts = np.zeros(11, dtype=np.uint8)
    acts[:] = STAY
    acts[5] = acts[10] = UP
    pos = positions_from((1, 7), acts, check=False)
    assert tuple(pos[-1]) == (-1, 7)


def test_render_episode_matches_env_steps():
    env = ColorFieldSymEnv()
    g0, _ = env.reset(seed=7, start=(6, 9))
    rng = np.random.default_rng(7)
    actions = [STAY]
    grids = [g0]
    for t in range(1, 61):
        a = int(rng.choice(env.valid_actions()))
        g, _ = env.step(a)
        grids.append(g)
        actions.append(a)
    proc = render_episode(env.map, (6, 9), np.array(actions))
    assert np.array_equal(proc, np.stack(grids))


def test_determinism():
    e1, e2 = ColorFieldSymEnv(), ColorFieldSymEnv()
    (g1, p1), (g2, p2) = e1.reset(seed=11), e2.reset(seed=11)
    assert np.array_equal(g1, g2) and p1 == p2 == 0
    assert np.array_equal(e1.map, e2.map) and e1.pos == e2.pos


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("test_env: ALL PASS")
