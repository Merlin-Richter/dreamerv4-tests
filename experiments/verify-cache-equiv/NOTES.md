# Cache-equivalence of the carrying KV-cached rollout

**Provenance:** branch `master`, commit `f91e2a0` (spec→code sync: every-3rd temporal cadence depth=9,
dynamics `logit_scale`, tokenizer sigmoid, `--resume` CLI). CPU, fp32. Task:
`tasks/in-progress/verify-cached-vs-uncached-rollout-identical.md`.

## Question
Is the production carrying rollout (`generate`/`rollout_init`/`rollout_step`, which always uses the
across-time KV cache + absolute-index RoPE) numerically identical to an UNCACHED reference that
recomputes the full current sliding window through `forward(cache=None)` each step (within-window
RoPE, faithful 5th-pass commit, sliding-window eviction), with matched noise?

## Method
`experiments/verify-cache-equiv/probe.py` (measurement) and `src/tests/test_dynamics_cache.py` (gate).
Noise is matched by drawing in the production order (init: 1×`randn_like(context)`; per frame:
`randn(frame)` then on commit `randn_like(z)`) under one `manual_seed` per path, so the cache is the
ONLY difference. Run in fp32, `eval()` (dropout off). Configs: vanilla / memory(n_memory>0,ff9_k>0) /
labeled(n_actions>0), depth 6 (2 temporal layers) and 9 (3), plus a depth-3 (1 temporal layer)
discriminator.

## Result
- **Within the window (rollout never exceeds `max_temporal_length`): cached == uncached up to fp
  (~5e-6).** The KV cache + absolute-index RoPE is a *correct* optimization here. (HOWTO's RoPE caveat
  is fully addressed — the within-window equivalence is exact.)
- **Across sliding-window eviction: cached and the uncached current-window recompute DIVERGE
  materially — O(0.1–1.4) max-abs-diff in latent units, NOT fp noise.** The divergence appears at
  exactly the first frame generated *after* the window first drops a committed frame.
- **Discriminator (clean DICHOTOMY, not a graded depth effect):** a 1-temporal-layer model (depth=3)
  stays *mathematically* exact THROUGH eviction (fp32 residual ≤7.4e-4 → ~3e-7 in fp64); any model with
  **≥2 stacked temporal layers** (depth 6 or 9) diverges O(1) and does **not** shrink in fp64
  (1.41 fp32 == fp64). Magnitude is weight-dependent and **not** monotone in temporal-layer count
  (d6 ≈ d9, sometimes d6 > d9 on random init) — the robust statement is the 1-vs-≥2 dichotomy.
  Independently reproduced by the `critical-claim-verifier` (`indep_probe.py`, fp32+fp64).

Representative numbers (max_ctx=5, T_ctx=2; first post-eviction frame = gen[4]):

| config        | within-window | post-eviction maxdiff |
|---------------|---------------|-----------------------|
| vanilla d6    | exact (~0)    | 0.42                  |
| vanilla d9    | exact (~0)    | 0.65                  |
| memory  d6    | exact (~0)    | 0.28                  |
| labeled d6    | exact (~0)    | 0.28                  |
| depth=3 (1 t) | exact (~0)    | ~0 (8.3e-7) THROUGH eviction |

## Root cause
With ≥2 stacked temporal layers, a committed frame's K/V at the *2nd/3rd* temporal layer is a function
of its *deeper* representation, which (via the *1st* temporal layer) depends on the frames that were
in-window when it was committed — its **commit-time receptive field**. The cache freezes that K/V. After
eviction the current window no longer contains some of those frames, so a fresh windowed recompute
produces a different deep-layer K/V. With only 1 temporal layer the K/V depend solely on each frame's
own spatial-only representation (window-independent) → no effect. Hence the divergence is precisely a
stacked-temporal-layer phenomenon. RoPE is NOT the culprit: cached rotates at the absolute index,
uncached at the within-window index, but both windows are contiguous so relative phases (and V, which
is unrotated) match — confirmed by the depth-3 exactness holding in fp64.

## Interpretation — RESOLVED by Merlin (2026-06-26): the divergence is DESIRED, not a defect
Merlin's verdict: *"It should diverge as soon as eviction starts — that's literally the whole point of
information preservation."* The frozen cache **carries state the sliding window has dropped**; an
uncached current-window recompute, by construction, **cannot** — it only sees the live window. So the
post-eviction gap between (cached) and (windowed-recompute) is exactly the memory mechanism doing its
job: the cached path retains evicted hidden state, the recompute path forgets it. This is the CORRECT
and intended behavior, not a train/inference bug to fix.

Consequences to keep in mind (not actions):
  * The recall eval measures the cached (memory-carrying) path — which is the point.
  * The within-window bit-exactness still matters as a correctness check (the cache must be lossless
    while everything is in-window); that's the hard gate in `test_dynamics_cache.py`.
  * A vanilla (`n_memory=0`) model also diverges post-eviction (frozen deep-layer K/V of plain
    tokens) — so the divergence is the SWA-cache mechanism in general; memory tokens are the part
    *trained* (FF9) to make what's carried be the hidden state. The recall A/B (memory vs vanilla)
    is what tells us whether the carried state is actually useful.

## Artifacts
- `probe.py` — per-frame diff measurement harness.
- `src/tests/test_dynamics_cache.py` — gate test: asserts within-window exactness + read-only-branch
  equivalence + the characterized eviction divergence + the depth-3 discriminator.
