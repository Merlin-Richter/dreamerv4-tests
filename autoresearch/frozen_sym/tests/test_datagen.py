"""Gate tests: ColorField-SYM procedural dataset — generation, determinism,
replay-exactness under the phase-5 rule. FROZEN-LAYER-sym test; spec:
tasks/in-progress/colorfield-sym-frozen-layer.md."""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen_sym.datagen import ColorFieldSymDataset, generate  # noqa: E402
from autoresearch.frozen_sym.env import (  # noqa: E402
    ColorFieldSymEnv, PHASE_PERIOD, STAY, VIEW_CELLS)


def test_generate_load_replay_determinism():
    d1 = tempfile.mkdtemp()
    d2 = tempfile.mkdtemp()
    try:
        generate(d1, n_episodes=3, T=130, seed=99, verbose=False)
        generate(d2, n_episodes=3, T=130, seed=99, verbose=False)
        ds = ColorFieldSymDataset(d1)
        ds2 = ColorFieldSymDataset(d2)
        assert len(ds) == 3
        for name in ("maps", "starts", "actions", "policy_ids"):
            assert np.array_equal(getattr(ds, name), getattr(ds2, name)), name
        assert (ds.actions[:, 0] == STAY).all()
        # phase discipline is IN the data: off-phase ticks are STAY
        off = np.arange(ds.actions.shape[1]) % PHASE_PERIOD != 0
        assert (ds.actions[:, off] == STAY).all()
        assert (ds.actions[:, ~off] != STAY).any()
        assert ds.policy_ids.max() < len(ds.meta["policies"])
        # procedural render == stepping the env with the stored actions
        for i in range(3):
            grids, actions = ds.episode(i)
            assert grids.shape == (130, VIEW_CELLS, VIEW_CELLS) and grids.dtype == np.uint8
            env = ColorFieldSymEnv()
            g, _ = env.reset(map_arr=ds.maps[i], start=tuple(ds.starts[i]))
            assert np.array_equal(grids[0], g)
            for t in range(1, len(actions)):
                g, _ = env.step(int(actions[t]))
                assert np.array_equal(grids[t], g), (i, t)
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


if __name__ == "__main__":
    test_generate_load_replay_determinism()
    print("PASS test_generate_load_replay_determinism")
    print("test_datagen: ALL PASS")
