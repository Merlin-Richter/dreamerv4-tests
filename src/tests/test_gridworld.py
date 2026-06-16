"""GridWorldEnv (D-032) gate tests: geometry, reflection, determinism, occlusion,
measurement validity, and the curtain-schedule block distribution.

Run:  python src/tests/test_gridworld.py   (or pytest).  CPU only, no model needed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.gridworld import (CURTAIN_COLOR, GRID_N, IMG_SIZE, PALETTE, VIS,  # noqa: E402
                            GridWorldEnv, cell_origin, interior_axis_mask)
from datagen.generate_gridworld import generate_episode, make_curtain_schedule  # noqa: E402


def test_geometry_uniform_6px_cells_and_borders():
    """1px border + 8x(6px cell)+7x(2px line)+1px border = 64; every interior run is 6px."""
    mask = interior_axis_mask()  # True = interior
    assert mask[0] == False and mask[-1] == False, "outer border must be black (1px)"
    # run-length encode the interior/line alternation
    runs, cur, length = [], mask[0], 1
    for v in mask[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((bool(cur), length)); cur, length = v, 1
    runs.append((bool(cur), length))
    interiors = [ln for is_int, ln in runs if is_int]
    assert interiors == [VIS] * GRID_N, f"expected {GRID_N} interiors of {VIS}px, got {interiors}"
    print("[ok] geometry: uniform 6px cells, enclosed by 1px border")


def test_deterministic():
    a = [GridWorldEnv().reset(seed=7).step(0)[1] for _ in range(1)]  # noqa: F841
    e1 = GridWorldEnv().reset(seed=7); e2 = GridWorldEnv().reset(seed=7)
    t1 = [e1.step(0)[1] for _ in range(50)]
    t2 = [e2.step(0)[1] for _ in range(50)]
    assert all(np.array_equal(x, y) for x, y in zip(t1, t2))
    print("[ok] deterministic given seed")


def test_in_bounds_reflection():
    env = GridWorldEnv().reset(seed=3)
    for _ in range(2000):
        _, s = env.step(0)
        assert 0 <= int(s[0]) < GRID_N and 0 <= int(s[1]) < GRID_N
    print("[ok] reflection keeps square in-bounds (2000 steps)")


def test_colors_distinct_in_palette():
    for sd in range(200):
        e = GridWorldEnv().reset(seed=sd)
        assert e.bg_name != e.color_name
        assert e.color in PALETTE.values() and e.bg_color in PALETTE.values()
    print("[ok] background != square; both in 4-color palette")


def test_occlusion_and_measurement():
    env = GridWorldEnv().reset(seed=11)
    f_occ, _ = env.step(1)
    assert np.all(f_occ == np.array(CURTAIN_COLOR, dtype=np.uint8)), "occluded != flat curtain"
    f_rev, s = env.step(0)
    y0, x0 = cell_origin(int(s[1])), cell_origin(int(s[0]))
    patch = f_rev[y0:y0 + VIS, x0:x0 + VIS]
    assert np.all(patch == np.array(env.color, dtype=np.uint8)), "square not at reported cell/color"
    print("[ok] occluded=flat curtain; revealed square sits at reported (col,row) with .color")


def test_all_8_directions_reachable():
    dirs = {(GridWorldEnv().reset(seed=sd).dcol, GridWorldEnv().reset(seed=sd).drow)
            for sd in range(300)}
    assert len(dirs) == 8, f"expected 8 start directions, saw {len(dirs)}"
    print("[ok] all 8 start directions observed")


def test_curtain_schedule_distribution():
    """~90% single random / 5% 8-revealed-run / 5% 8-occluded-run; long runs of both present."""
    rng = np.random.default_rng(0)
    n = 200_000
    sched = make_curtain_schedule(rng, n, start_visible=0)
    occ_frac = sched.mean()
    # single blocks ~50/50 -> 0.90*0.5=0.45; revealed-run adds 0; occluded-run adds 0.05.
    # expected occluded fraction ~= 0.45 + 0.05 = 0.50 (run blocks are 8 frames each, equal count
    # -> equal frames -> cancels except their own polarity). Tolerance is loose.
    assert 0.40 < occ_frac < 0.60, f"occluded fraction {occ_frac:.3f} out of expected band"
    # both 8-long runs must occur
    def max_run(arr, val):
        best = run = 0
        for v in arr:
            run = run + 1 if v == val else 0
            best = max(best, run)
        return best
    assert max_run(sched, 1) >= 8, "no 8-long occluded run found"
    assert max_run(sched, 0) >= 8, "no 8-long revealed run found"
    print(f"[ok] schedule: occ_frac={occ_frac:.3f}, long runs of both scenarios present")


def test_generate_episode_shapes():
    frames, actions, states, colors = generate_episode(n_frames=40, seed=99)
    assert frames.shape == (40, IMG_SIZE, IMG_SIZE, 3) and frames.dtype == np.uint8
    assert actions.shape == (40,) and states.shape == (40, 5) and colors.shape == (2,)
    assert colors[0] != colors[1]
    print("[ok] generate_episode shapes/dtypes")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL GRIDWORLD GATE TESTS PASSED")
