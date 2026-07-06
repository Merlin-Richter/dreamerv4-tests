"""Gate tests: procedural dataset — generation, determinism, replay-exactness."""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen.datagen import ColorFieldDataset, generate  # noqa: E402
from autoresearch.frozen.env import ColorFieldEnv, STAY  # noqa: E402


def test_generate_load_replay_determinism():
    d1 = tempfile.mkdtemp()
    d2 = tempfile.mkdtemp()
    try:
        generate(d1, n_episodes=3, T=128, seed=99, verbose=False)
        generate(d2, n_episodes=3, T=128, seed=99, verbose=False)
        ds = ColorFieldDataset(d1)
        ds2 = ColorFieldDataset(d2)
        assert len(ds) == 3
        for name in ("maps", "starts", "actions", "policy_ids"):
            assert np.array_equal(getattr(ds, name), getattr(ds2, name)), name
        assert (ds.actions[:, 0] == STAY).all()
        assert ds.policy_ids.max() < len(ds.meta["policies"])
        # procedural render == stepping the env with the stored actions
        for i in range(3):
            frames, actions = ds.episode(i)
            env = ColorFieldEnv()
            f = env.reset(map_arr=ds.maps[i], start=tuple(ds.starts[i]))
            assert np.array_equal(frames[0], f)
            for t in range(1, len(actions)):
                f = env.step(int(actions[t]))
                assert np.array_equal(frames[t], f), (i, t)
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


if __name__ == "__main__":
    test_generate_load_replay_determinism()
    print("PASS test_generate_load_replay_determinism")
    print("test_datagen: ALL PASS")
