"""Memory-Maze evals.

`sheets.py`: action-conditioned ground-truth-vs-rollout filmstrip PNGs on held-out episodes (the
QUALITATIVE memmaze eval, counterpart of `evals.gridworld.sheets`, whose drawing layer it reuses).

`rollout_error.py` + `plot_rollout_error.py`: the QUANTITATIVE short-horizon instrument — decoded
pixel MSE vs ground truth over a 32-frame autoregressive rollout after a 128-frame streamed prefill,
measured identically for every model, saved to reusable JSON and overlaid into one comparison figure.
A short-horizon reconstruction comparison (butterfly-divergence caveat), not a full memory-retention
measure.
"""
