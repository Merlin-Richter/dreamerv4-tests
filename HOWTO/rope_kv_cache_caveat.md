# RoPE + sliding window + KV cache — the rotation-continuity trap

**Status:** flagged 2026-06-12 (Merlin, milestone / D-011). **ADDRESSED for D** on
2026-06-13 (T-008 / D-017): the dynamics model now has an absolute-position RoPE path and a
KV cache (`DynamicsModel.generate_cached`, `new_kv_cache`; temporal `Attention.forward`
`positions=`/`layer_cache=`/`commit=` args). The trap below is handled exactly as prescribed
(on-the-fly rotation at a never-reset absolute position; cached K/V stored already-rotated and
never re-indexed) and validated bit-for-bit in `src/D_dynamics_model/test_kv_cache.py`
(incremental==full forward at T beyond the table; seeded generate_cached==generate). Still
UNIMPLEMENTED for the tokenizer C — read this before caching C for streaming long sequences.

## The facts (verified against code)

- **RoPE is relative.** Attention dot-products depend only on the *difference* of
  token positions, not their absolute values. Consequence: a model trained at context
  length N can be run at any inference window **M < N with no retraining**, and a
  rollout 1000 steps deep is identical in kind to one at step N.
- **The dynamics `generate()` already slides the window** (`dynamics_model.py` ~L391:
  `window = seq[:, -max_ctx:]`, `max_ctx = max_temporal_length - 1`). It recomputes
  the full window each step, assigning RoPE positions within the freshly-fed window.
  This is correct **only because there is no cache** — every K/V is recomputed each
  step, so re-indexing positions per window is harmless.
- **A sliding-window transformer has no persistent state.** Info older than N−1 frames
  is absent from the model — not in the input, not stored. Register tokens are
  per-frame scratch, not a cross-window carrier. (This is the H2 baseline, by
  construction — see GOAL.md H2.)

## The trap (when you add KV cache)

A KV cache freezes each token's K/V at the rotation it was given on entry. You then
**cannot re-rotate a cached token** when the window slides. So you must NOT use the
current scheme (re-index positions 0..w within each window). Instead:

- Use a **continuously-advancing absolute position counter** (a running clock) that is
  **never reset** across the rollout. Token X keeps the absolute rotation it got on
  entry; relative distances between any two cached tokens stay correct because RoPE
  rotation is a function of absolute position and the dot-product depends on the
  difference.
- The current fixed `cos/sin` lookup table (size `max_temporal_length`, ~16) is
  therefore **cache-incompatible for long rollouts** — positions exceed the table.
  Compute the rotation **on the fly** for arbitrary positions (or use an unbounded /
  sufficiently large generator), and let the sliding window drop old *entries* while
  the position clock keeps progressing.

Symptom if you get this wrong: cached long rollouts silently corrupt (positions
collide / reset), degrading recall in a way that looks like a memory failure but is a
positional-encoding bug. Validate any KV-cache implementation by checking that a
cached rollout is bit-for-bit (or within fp tolerance) equal to the uncached
recompute-each-step rollout over the same horizon.

Applies to both D (dynamics) and C (tokenizer, if used for streaming long observed
sequences).
