# evals/gridworldv2/recall.py — action-conditioned memory recall (GridWorldV2)

> **STATUS: DRAFT** (agent-written 2026-07-04; Merlin has not signed off). Agent-chosen decision
> point flagged for review: the ALIGNMENT convention below (branch-after-commit; k = number of
> occluded MOVEMENT actions integrated) — v2's toggle-tick semantics make this the natural
> reading, but it differs in bookkeeping from v1's branch-before-commit convention.

The v2 memory question: how well can the model track the hidden square when, behind the curtain,
its position is driven by the ACTION stream (clamped at walls — a nonlinear function of the whole
sequence)? Memory must carry the last observed state AND integrate every subsequent action, not
extrapolate ballistics.

## Protocol (mirrors v1's batched driver; per env seed)

1. **Context**: `n_ctx` REVEALED frames driven by the shared movement policy
   (`envs.gridworldv2.sample_moves`, seeded from the env's rng — same statistics as datagen).
   Context actions are passed to `rollout_init` (the model is action-conditioned).
2. **Hide tick**: one committed `hide` (action 1) — curtain latches down, square does not move.
   Not scored.
3. **Occluded rollout**, k = 1..max_k: per tick, one movement action `m_k` from the same policy
   stream; the env advances (truth `p_k`) and the model COMMITS the tick with action `m_k`.
4. **Scored reveal (read-only)**: at each checked k (v1's `_check_ks` grid), branch
   `rollout_step(action=0, commit=False)` — semantically "reveal now, no move" — decode, read the
   square (v1 readout, unchanged), score against `p_k`. The branch never mutates the carried
   state; the occluded rollout continues.

**ALIGNMENT**: the branch is taken AFTER committing the k-th movement tick, so the belief at k
reflects the context + the hide tick + k committed occluded movement ticks, and is scored against
the position after exactly those k movements. k therefore counts occluded MOVEMENT actions
integrated. (No off-by-one: the reveal action itself does not move the square.)

## Baselines / ceiling (same readout, same alignment)

- `oracle` — read the TRUE revealed render (`env.render_revealed()`, measurement-only) at each
  checked k. Must be 1.0 (instrument self-test).
- `copy_last` — freeze the square at the last OBSERVED cell (end of context). The no-memory,
  no-action-integration reference.
- `chance` — v1's analytic floors (identical grid/palette; imported).

## Interface

`recall(model, tokenizer, *, n_ctx=4, max_k, n_rollouts=64, K=4, device, window=None,
batch_size=64) -> dict` with the same result schema as v1 (`model/copy_last/oracle/chance`,
metrics `position_acc/position_score/color_acc` per k). CLI mirrors v1's
(`--checkpoint --tokenizer --max-k [--window --n-ctx --n-rollouts --K --batch-size --out]`),
default output `outputs/recall/recallv2_<checkpoint-stem>.json`, meta records `env=gridworldv2`.

Reuses from v1 (`evals/gridworld/recall.py`): `score_reveal`, `chance_levels`, `_check_ks`,
`_tokenizer_window`, `_load_checkpoint` — the scorer stays a single implementation; only the env
driving differs.
