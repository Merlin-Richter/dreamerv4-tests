"""Establish reference numbers for the frozen baselines at the reduced config."""
from harness import SMALL, summarize
from autoresearch.frozen.eval_comeback import run_eval
from autoresearch.frozen.adapters import make_adapter

for name in ("oracle", "perfect_imaginary", "noise_cells", "constant_color", "copy_last"):
    res = run_eval(make_adapter(name), **SMALL)
    summarize(res, name)
    print()
