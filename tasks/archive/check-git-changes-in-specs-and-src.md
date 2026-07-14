# Check the git changes that happened in specs/ and src/ and verify that specs and src still matches. Then commit
## Result (2026-06-26)
Verified the uncommitted specs/+src/ diff is coherent and spec↔code consistent:
- dynamics_model.py: `max_ctx`/`window` knob threaded through `rollout_init`/`generate` — matches dynamics_model.md.
- recall.py: `window=` param + `__main__` CLI writing curves + `meta` block (n_memory/window/native_window) — matches recall.md.
- NEW plot_recall.py + sheets.py: signatures, CLIs, helpers all match new specs plot_recall.md / sheets.md.
- Minor additive drift (non-blocking): sheets.py CLI has extra `--actions` and `--occ-seed0` flags not listed in sheets.md.
Gate tests all green: test_gridworld, test_gridworld_eval, test_dynamics, test_dynamics_cache.
Committed as rollback point.
