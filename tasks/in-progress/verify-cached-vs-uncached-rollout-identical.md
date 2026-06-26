# Verify the KV-cached rollout is numerically equivalent to an uncached forward

The carrying rollout (`generate` / `rollout_init` / `rollout_step`) is the only inference path, and it
always uses the across-time KV cache (§4 of `specs/models/dynamics_model.md`). The cache is purely an
optimization: a frame's K diffusion steps only *read* committed past K/V, and the 5th pass commits the
frame's K/V at near-clean + written memory. With RoPE rotated at the absolute rollout index, the cached
path should be **bit-for-bit (up to fp tolerance) identical** to recomputing the whole sliding window
through `forward` with `cache=None` every step.

Goal: write and run a test that **proves** cached and uncached produce the same rollout, for both the
vanilla (`n_memory=0`) and memory (`n_memory>0`) models, including through window eviction.

## What to build

Add a gate test (e.g. `src/tests/test_dynamics_cache.py`, or extend `src/tests/test_dynamics.py`) that:

1. **Writes an uncached reference rollout.** There is no no-cache mode on the rollout API today, so the
   test must implement an uncached reference that calls `forward(..., cache=None)` over the **full
   sliding window** on each step, faithfully replicating the cached protocol:
   - same per-frame K shortcut denoising steps (read-only over the in-window context),
   - the same near-clean context held at `context_signal` with each committed frame's **written** memory
     token (memory relay must match — read-old/write-new, not a frozen activation),
   - present old generated frames excatly like the 5th-pass commit semantics commit them to cache for a correct test (the frame re-presented near-clean enters the window),
   - RoPE not at absolute rollout indices like chached does, but simply as index inside the current context window, and sliding-window eviction once the window fills.
2. **Controls the noise.** `rollout_step` samples `torch.randn` internally per frame/step. The cached and
   uncached runs MUST consume the identical noise sequence (seed/`Generator`, or refactor so both draw
   the same noise) — otherwise the comparison is meaningless. Make this explicit in the test. Both need to work exactly the same appart from the cache, so noise should be based on a (seed + absolute rollout position).
3. **Compares outputs.** Run both paths from the same context + actions and assert the generated latents
   match within a tight fp tolerance (`eval()` mode so dropout is off; mind bf16 default dtype — run in
   fp32 or use a tolerance that reflects bf16). Cover:
   - vanilla (`n_memory=0`),
   - memory (`n_memory>0, ff9_k>0`),
   - `n_generate > max_temporal_length` so eviction is exercised,
   - the read-only branch (`rollout_step(commit=False)`) agrees with an uncached read-only prediction.

## Notes / gotchas
- If an exact match is impossible because of an ordering/rotation/eviction detail, that is a **finding**:
  the cache would not be a pure optimization. Localize it (which layer/step diverges) and report it
  rather than loosening the tolerance to hide it. See `HOWTO/rope_kv_cache_caveat.md`.
- Keep it CPU-only and fast, matching the other gate tests' style (small config like `BASE` in
  `test_dynamics.py`). Run with `python -u`.
- This test should work with any model, like a random model or a trained model.

## Done when
- The test runs green on CPU and is wired into the gate-test set (runnable like the other
  `src/tests/test_*.py`).
- Both vanilla and memory models pass the cached==uncached check through eviction, with the noise
  control made explicit.
- If any genuine divergence is found, it is documented (here + `agent/EXPERIMENTS.md`) with the root
  cause, not masked by tolerance.
