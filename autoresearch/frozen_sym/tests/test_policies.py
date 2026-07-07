"""Gate tests: ColorField-SYM datagen policy zoo validity/coverage under the
phase-5 loop discipline + closed-loop eval-policy safety (never pushes a
blocked band, real or adversarially imagined). FROZEN-LAYER-sym test; spec:
tasks/in-progress/colorfield-sym-frozen-layer.md."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen_sym.env import (  # noqa: E402
    ColorFieldSymEnv, DOWN, LEFT, PHASE_PERIOD, STAY, out_bands)
from autoresearch.frozen_sym.eval_policies import (  # noqa: E402
    BAND_BLOCK, EVAL_SUITE, _BAND_KEY, allowed_moves)
from autoresearch.frozen_sym.policies import (  # noqa: E402
    POLICY_REGISTRY, rollout_policy)


def test_datagen_policies_valid_and_covering():
    for name, cls in POLICY_REGISTRY:
        for seed in (0, 1, 2):
            env = ColorFieldSymEnv()
            env.reset(seed=seed * 7919 + 13)
            rng = np.random.default_rng(seed)
            start = env.pos
            positions = {start}
            policy = cls()
            policy.reset(rng, start)
            # replicate rollout_policy's phase loop but record coverage
            for t in range(1, 2001):
                if t % PHASE_PERIOD == 0:
                    a = int(policy.act(env.pos, rng))
                else:
                    a = STAY
                assert a in env.valid_actions(), (name, seed, t, env.pos, a)
                env.step(a)
                positions.add(env.pos)
            assert len(positions) >= 15, (name, seed, len(positions))


def test_datagen_rollout_convention():
    env = ColorFieldSymEnv()
    env.reset(seed=5)
    rng = np.random.default_rng(5)
    policy = POLICY_REGISTRY[0][1]()
    actions = rollout_policy(policy, env, 300, rng)
    assert actions[0] == STAY and actions.shape == (300,)
    # off-phase ticks are forced STAY — the phase discipline is in the data
    off = np.arange(300) % PHASE_PERIOD != 0
    assert (actions[off] == STAY).all()
    # phase-0 ticks actually move (a frozen policy would defeat the point)
    assert (actions[~off] != STAY).any()


def test_eval_policies_never_push_banded_sides():
    """Fuzz with ADVERSARIAL synthetic bands (imagined-early borders up to the
    full 5-cell viewport, all-blocked, flickering) — an eval policy must never
    return a move into a band >= BAND_BLOCK."""
    rng = np.random.default_rng(0)
    for name, factory in EVAL_SUITE:
        pol = factory()
        pol.reset(np.random.default_rng(1))
        prng = np.random.default_rng(2)
        for t in range(2500):
            bands = {k: 0 for k in ("up", "down", "left", "right")}
            n_block = int(rng.integers(0, 5))
            for side in rng.choice(["up", "down", "left", "right"], size=n_block, replace=False):
                bands[side] = int(rng.integers(BAND_BLOCK, 6))  # 2..5: imagined-wide too
            for k in bands:
                if bands[k] == 0 and rng.random() < 0.5:
                    bands[k] = 1                                # sub-threshold noise
            a = pol.act(bands, prng)
            if a != STAY:
                assert bands[_BAND_KEY[a]] < BAND_BLOCK, (name, t, bands, a)


def test_eval_policies_valid_on_real_env():
    """On REAL grids, band >= 2 <=> at the true board edge, so a closed-loop
    policy driven by real bands (consulted at phase-0 only, STAY forced
    off-phase) must never emit an invalid action."""
    for name, factory in EVAL_SUITE:
        env = ColorFieldSymEnv()
        rng = np.random.default_rng(7)
        pol = factory()
        pol.reset(rng)
        grid, _ = env.reset(seed=42, start=(1, 1))  # near a corner, borders in view
        for t in range(1, 2001):
            if t % PHASE_PERIOD == 0:
                a = pol.act(out_bands(grid), rng)
            else:
                a = STAY
            assert a in env.valid_actions(), (name, t, env.pos, a)
            grid, _ = env.step(a)


def test_allowed_moves_helper():
    bands = {"up": 2, "down": 0, "left": 1, "right": 5}
    assert set(allowed_moves(bands)) == {DOWN, LEFT}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("test_policies: ALL PASS")
