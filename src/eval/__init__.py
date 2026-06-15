"""src/eval — working, reusable evaluation & rollout utilities (NON-frozen).

Distinct from `src/probe/`, which is the FROZEN revisit/position-consistency probe SPINE
(frozen @ 5503e75; any change there is a logged decision — see GOAL.md §8). Code here is the
mutable toolbox that *uses* the frozen probe: motion curves, A/B drivers, rollout diagnostics
that recur across experiments. Extracted out of experiments/EXP-NNN/ so it's findable and shared.
"""
