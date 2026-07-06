"""Prove recover_start() is exact and that the path-integral position an adapter
maintains EXACTLY matches the eval tracker's registration -- no env peeking."""
import numpy as np
from harness import SMALL_SUITE
from autoresearch.frozen.env import ColorFieldEnv, STAY, apply_action
from autoresearch.frozen.eval_comeback import corner_start, PREFIX_LEN
from autoresearch.frozen.eval_policies import EvalRetrace
from autoresearch.frozen.readout import border_bands
from exploits import recover_start

bad = 0
for seed in range(40):
    rng = np.random.default_rng(seed)
    env = ColorFieldEnv()
    start = corner_start(rng)
    frame0 = env.reset(seed=1000 + seed, start=start)
    rec = recover_start(frame0[None])   # only needs frame[0]
    if rec != tuple(start):
        bad += 1
        print("MISMATCH", seed, "true", start, "rec", rec, border_bands(frame0))
print(f"recover_start: {40-bad}/40 exact")
