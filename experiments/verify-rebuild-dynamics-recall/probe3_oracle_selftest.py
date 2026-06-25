"""Probe 3: oracle self-test (recall spec 2c). Reading the square out of the TRUE revealed frame
must be exact at every scored k: pos_correct==1 and color_correct==1. Reproduces recall's exact
oracle path (env.step(0) advances physics; score_reveal on the true frame). 64 seeds, max_k=32.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from envs.gridworld import GridWorldEnv, PALETTE
from evals.gridworld.recall import score_reveal, _check_ks

COLOR_NAMES = list(PALETTE.keys())
n_ctx, max_k, n_seeds = 4, 32, 64
checks = set(_check_ks(max_k))
bad = []
for seed in range(n_seeds):
    env = GridWorldEnv().reset(seed)
    s = None
    for _ in range(n_ctx):
        _, s = env.step(0)
    colors = (COLOR_NAMES.index(env.bg_name), COLOR_NAMES.index(env.color_name))
    for k in range(1, max_k + 1):
        f_true, s_true = env.step(0)
        if k in checks:
            rec = score_reveal(f_true, (int(s_true[0]), int(s_true[1])), colors)
            if rec["pos_correct"] != 1 or rec["color_correct"] != 1 or rec["pos_score"] != 1.0:
                bad.append((seed, k, rec))
print("checked k's:", sorted(checks))
print("seeds:", n_seeds, " total oracle scorings:", n_seeds * len(checks))
print("oracle failures (pos!=1 or color!=1):", len(bad))
if bad: print("first few:", bad[:5])
print("ORACLE SELF-TEST PASS:", len(bad) == 0)
