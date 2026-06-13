# T-012 — Cross-frame sliding-window KV eviction cache (D-020)

**Goal:** a persistent KV cache for autoregressive sliding-window rollouts of the dynamics model D,
as the efficient + correct substrate the eventual register-relay rollout-training method will unroll on.
Easily verifiable (Merlin-directed). Infra, not an experiment — the correctness tests are the artifact.

## Context (code-grounded, dynamics_model.py)
- `generate()` (:518): uncached sliding-window rollout. Per generated frame: window = last
  `max_ctx = N-1 = 7` frames, re-noise the whole window to `context_signal` with FRESH randn, run K
  shortcut substeps (`_denoise_next`). Re-encodes the full window every frame.
- `generate_cached()` (:603) / `_denoise_next_cached()` (:555): caches the window's K/V across the K
  substeps WITHIN one frame, then DISCARDS the cache and rebuilds it next frame. Bit-identical to
  `generate()`; ~Kx fewer temporal FLOPs; **no cross-frame reuse.**
- Temporal `Attention.forward` (:132): K/V are stored **already-RoPE-rotated at absolute positions**
  (:156-168). `cache`/`commit`/`positions` plumbing + `new_kv_cache()` (:550) already exist (T-008).
- FF7 register-memory path (`use_register_memory=True`) is window-1 already → no cache benefit → leave
  dispatched to `generate_memory` unchanged.

## Design
Persistent per-block cache across rollout steps with sliding-window eviction:
1. **Prefill** (`stream_rollout_init`): noise the `T_ctx` context frames to `context_signal` ONCE,
   commit their K/V into a fresh `new_kv_cache()` at absolute positions `0..T_ctx-1`. If `T_ctx > max_ctx`,
   keep only the last `max_ctx`. Return state: `{cache, next_pos, max_ctx, K, tau/d idx consts, act state}`.
2. **Step** (`stream_rollout_step`): denoise the new frame at absolute `next_pos` via K substeps,
   attending to the cache with `commit=False` (substeps must NOT mutate the cache). After finalizing the
   clean latent `z`, noise it to `context_signal` ONCE and commit it at `next_pos` (`commit=True`), then
   **evict** the oldest time-column if the cache length now exceeds `max_ctx` (`cache[b]['k']=...[:,:,1:,:]`
   on the time axis; same for v). `next_pos += 1`. Return `(z, state)`.
3. **Wrapper** (`generate_streaming`): thin loop over init+step, mirroring `generate_memory`.

**Eviction = pure slice on the time axis.** Because cached K/V are pre-rotated at *absolute* positions,
dropping the oldest column needs NO re-rotation — the surviving columns keep their correct absolute
phase. This is exactly what the absolute-RoPE foundation (T-008) was for.

**The one semantic deviation from `generate()`:** each frame's context-noise is drawn ONCE at commit and
reused while the frame sits in the window, vs `generate()`'s fresh redraw every step. Documented,
defensible (a frame's committed representation is fixed once generated) and the natural structure for
rollout training (context = fixed/detached). So: NOT bit-identical to `generate()`; IS bit-identical to a
full windowed recompute over the same frozen-noised frames, and to a frozen-noise reference rollout.

**Autograd note:** built `@torch.no_grad()` for inference now. The relay-training method will need grad
through the current step with the cache detached between steps (stop-grad TBPTT-1). The concat in
`Attention.forward` (cached K/V + new K/V) is already grad-compatible when the cache is detached; flagged
so that method can lift the no_grad and detach-on-commit without redesign.

## Verification (`src/D_dynamics_model/test_stream_cache.py`) — forward-level is the real gate
1. **Eviction equivalence, NO RNG (primary gate):** stream the model frame-by-frame through the
   persistent commit+evict cache; for each new frame at step t, the cached output must equal a FULL
   windowed forward over frames `[max(0,t-max_ctx) .. t]` with explicit positions, bit-for-bit (TOL 1e-4).
   Reference is independent of the cache (full recompute), so this isolates eviction + absolute-RoPE +
   causal mask. Run T both within and well past `max_temporal_length`.
2. **Long-rollout past the cos/sin table:** T >> max_temporal_length — eviction through unbounded
   absolute positions (the documented RoPE-overflow trap) stays equal to full recompute.
3. **Generate-level (D-021, Merlin):** `generate_streaming` (cached) == `generate_windowed` (the
   independent UNCACHED twin) under a shared `noise_seed`, bit-for-bit, with and without actions. The
   seed is a per-frame noise source keyed on the ABSOLUTE frame id (not RNG call order), so both real
   code paths get identical noise and the only difference is the cache → divergence = a real cache bug.
   This replaces the earlier test-local frozen-noise reimplementation (which could share the
   implementation's bug). Plus: `test_seeded_noise_is_reproducible` (seed determinism + different seed
   changes the rollout) and `test_divergence_is_detectable` (MUTATION test — break eviction, confirm
   the comparison fails → the test is sensitive). AND report the (small) deviation from standard
   `generate()` to confirm the frozen-noise semantics is benign.
4. **Speed sanity (not a gate):** `generate_streaming` faster than `generate_cached` on a long rollout
   (no per-frame cache rebuild).

## Done when
All forward-level equivalence tests green (incl. past-table), generate-level frozen-noise bit-exact,
deviation-from-generate small, speed win shown. Then commit + sync CLAUDE.md. No present-then-stop (infra).
