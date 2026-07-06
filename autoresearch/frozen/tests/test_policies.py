"""Gate tests: datagen policy zoo validity/coverage + closed-loop eval-policy
safety (never pushes a border, real or imagined)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen.env import (  # noqa: E402
    ColorFieldEnv, STAY, valid_actions)
from autoresearch.frozen.eval_policies import (  # noqa: E402
    BAND_BLOCK, EVAL_SUITE, _BAND_KEY, allowed_moves)
from autoresearch.frozen.policies import (  # noqa: E402
    POLICY_REGISTRY, rollout_policy)
from autoresearch.frozen.readout import border_bands  # noqa: E402


def test_datagen_policies_valid_and_covering():
    for name, cls in POLICY_REGISTRY:
        for seed in (0, 1, 2):
            env = ColorFieldEnv()
            env.reset(seed=seed * 7919 + 13)
            rng = np.random.default_rng(seed)
            start = env.pos
            positions = {start}
            policy = cls()
            policy.reset(rng, start)
            # replicate rollout_policy but record coverage
            for _ in range(1500):
                a = int(policy.act(env.pos, rng))
                assert a in env.valid_actions(), (name, seed, env.pos, a)
                env.step(a)
                positions.add(env.pos)
            assert len(positions) >= 25, (name, seed, len(positions))


def test_datagen_rollout_convention():
    env = ColorFieldEnv()
    env.reset(seed=5)
    rng = np.random.default_rng(5)
    policy = POLICY_REGISTRY[0][1]()
    actions = rollout_policy(policy, env, 300, rng)
    assert actions[0] == STAY and actions.shape == (300,)


def test_eval_policies_never_push_banded_sides():
    """Fuzz with ADVERSARIAL synthetic bands (imagined-early borders, all-blocked,
    flickering) — an eval policy must never return a move into a band >= BAND_BLOCK."""
    rng = np.random.default_rng(0)
    for name, factory in EVAL_SUITE:
        pol = factory()
        pol.reset(np.random.default_rng(1))
        prng = np.random.default_rng(2)
        for t in range(2500):
            bands = {k: 0 for k in ("up", "down", "left", "right")}
            n_block = int(rng.integers(0, 5))
            for side in rng.choice(["up", "down", "left", "right"], size=n_block, replace=False):
                bands[side] = int(rng.integers(BAND_BLOCK, 65))
            for k in bands:
                if bands[k] == 0 and rng.random() < 0.5:
                    bands[k] = int(rng.integers(0, BAND_BLOCK))  # sub-threshold noise
            a = pol.act(bands, prng)
            if a != STAY:
                assert bands[_BAND_KEY[a]] < BAND_BLOCK, (name, t, bands, a)


def test_eval_policies_valid_on_real_env():
    """On REAL frames, band >= 30 <=> at the true lattice edge, so a closed-loop
    policy driven by real bands must never emit an invalid action."""
    for name, factory in EVAL_SUITE:
        env = ColorFieldEnv()
        env.reset(seed=42, start=(4, 4))  # near a corner, borders in view
        rng = np.random.default_rng(7)
        pol = factory()
        pol.reset(rng)
        frame = env.reset(seed=42, start=(4, 4))
        for t in range(1200):
            a = pol.act(border_bands(frame), rng)
            assert a in env.valid_actions(), (name, t, env.pos, a)
            frame = env.step(a)


def test_allowed_moves_helper():
    bands = {"up": 31, "down": 0, "left": 29, "right": 30}
    from autoresearch.frozen.env import DOWN, LEFT
    assert set(allowed_moves(bands)) == {DOWN, LEFT}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("test_policies: ALL PASS")
