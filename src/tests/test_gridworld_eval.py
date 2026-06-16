"""Validation that the GridWorld recall metric is a SOUND instrument (D-032).

The discrete env's payoff is that the readout is closed-form and provable:
  * on TRUE frames the readout recovers the exact (col,row,color,bg)  -> instrument is exact;
  * the ORACLE frame source scores position_acc == 1.0 at every k       -> ceiling sanity;
  * the COPY-LAST (no-memory) source decays with k and never beats oracle -> baseline is meaningful;
  * a RANDOM-cell source sits at ~chance (1/64)                          -> floor sanity.
If these hold, a real model's curve between copy-last and oracle is interpretable as "memory."

Run:  python src/tests/test_gridworld_eval.py   (CPU only).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datagen.generate_gridworld import generate_episode  # noqa: E402
from envs.gridworld import GRID_N, PALETTE, GridWorldEnv, make_grid_background, stamp_square  # noqa: E402
from evals.gridworld.readout import read_square  # noqa: E402
from evals.gridworld.recall import (aggregate, copylast_frames, find_reveal_events,  # noqa: E402
                                     oracle_frames, score_episode)

_PAL = list(PALETTE.values())


def test_readout_exact_on_true_frames():
    for sd in range(50):
        fr, ac, st, col = generate_episode(n_frames=30, seed=sd)
        for t in range(len(fr)):
            if ac[t] == 1:
                continue  # occluded frame: nothing to localize
            rd = read_square(fr[t])
            assert (rd["col"], rd["row"]) == (int(st[t, 0]), int(st[t, 1])), f"pos sd{sd} t{t}"
            assert rd["color_idx"] == int(col[1]), f"square color sd{sd} t{t}"
            assert rd["bg_idx"] == int(col[0]), f"bg sd{sd} t{t}"
    print("[ok] readout exact on true frames (pos+color+bg), all revealed frames, 50 episodes")


def test_oracle_is_ceiling():
    recs = []
    for sd in range(100):
        fr, ac, st, col = generate_episode(n_frames=120, seed=sd)
        recs += score_episode(oracle_frames(st, col, ac), st, col, ac)
    agg = aggregate(recs)
    assert agg["n_events"] > 200, f"too few reveal events: {agg['n_events']}"
    assert all(v == 1.0 for v in agg["position_acc"].values()), agg["position_acc"]
    assert all(v == 1.0 for v in agg["color_acc"].values()), agg["color_acc"]
    print(f"[ok] oracle ceiling: position_acc==1.0 at all k ({agg['n_events']} events, "
          f"k={sorted(agg['position_acc'])})")


def test_copylast_decays_and_below_oracle():
    recs = []
    for sd in range(300):
        fr, ac, st, col = generate_episode(n_frames=160, seed=sd)
        recs += score_episode(copylast_frames(st, col, ac), st, col, ac)
    agg = aggregate(recs)
    pa = agg["position_acc"]
    # copy-last is exact at k=1 only if the square didn't move (it always moves 1 cell), so even
    # k=1 is < 1.0; and accuracy must not INCREASE with k (square drifts away monotonically-ish).
    ks = sorted(pa)
    assert pa[ks[0]] < 1.0, f"copy-last should be imperfect even at k=1: {pa}"
    # color is static -> copy-last keeps it perfectly (sanity: memory of identity is trivial here)
    assert all(v == 1.0 for v in agg["color_acc"].values()), agg["color_acc"]
    # large-k position accuracy should be low (square has wandered far)
    assert pa[ks[-1]] < 0.5, f"copy-last position should be poor at large k: {pa}"
    print(f"[ok] copy-last: pos_acc decays {pa[ks[0]]:.2f}@k{ks[0]} -> {pa[ks[-1]]:.2f}@k{ks[-1]}; "
          f"color stays 1.0 (static identity)")


def test_random_is_chance():
    rng = np.random.default_rng(0)

    def random_frames(states, colors, curtain):
        bg = _PAL[int(colors[0])]; sq = _PAL[int(colors[1])]
        base = make_grid_background(bg)
        T = len(states)
        out = np.empty((T, GridWorldEnv.img_size, GridWorldEnv.img_size, 3), dtype=np.uint8)
        for t in range(T):
            f = base.copy()
            stamp_square(f, int(rng.integers(GRID_N)), int(rng.integers(GRID_N)), sq)
            out[t] = f
        return out

    recs = []
    for sd in range(400):
        fr, ac, st, col = generate_episode(n_frames=120, seed=sd)
        recs += score_episode(random_frames(st, col, ac), st, col, ac)
    agg = aggregate(recs)
    overall = np.mean([r["pos_correct"] for r in recs])
    assert overall < 0.05, f"random position acc should be ~1/64: {overall:.4f}"
    print(f"[ok] random-cell source ~chance: overall pos_acc={overall:.4f} (1/64={1/64:.4f})")


def test_reflection_split_present():
    recs = []
    for sd in range(400):
        fr, ac, st, col = generate_episode(n_frames=160, seed=sd)
        recs += score_episode(oracle_frames(st, col, ac), st, col, ac)
    agg = aggregate(recs)
    n_bounced = sum(r["bounced"] for r in recs)
    assert 0 < n_bounced < len(recs), "need both bounced and straight events for the split"
    print(f"[ok] reflection split available: {n_bounced}/{len(recs)} events bounced during occlusion")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL GRIDWORLD EVAL VALIDATION TESTS PASSED")
