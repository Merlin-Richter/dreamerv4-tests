# ORIENT.md

Rewritten: 2026-06-13 (ESC-009/010 RESOLVED by Merlin → continue; building D-020 cross-frame KV cache)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75. H2 anchored on budget-matched `vanilla_s0`.
- **H3 color — supported (FF7 v1).** Register relay carries static hidden COLOR well past the N=8
  window (EXP-010); confound resolved (EXP-012); the FF7 1-step base-dynamics gain is the LOSS not the
  relay (EXP-014).
- **H3 position — OPEN.** EXP-013 measured blind-occlusion position memory as near-absent (vanilla ≈
  copy-last; FF7 marginally better). **Merlin (ESC-009): the position metric as coded is of uncertain
  strength — NOT frozen as a spine, NOT a hard gate.** Read stands; not litigated further. The next
  method to attempt position retention is the **sequential register-relay rollout training** (IDEAS.md).
- **NOW (D-020): building the rollout substrate that method needs** — a cross-frame sliding-window KV
  eviction cache. Easily verifiable; Merlin-directed. This is infra, not an experiment.

## In flight
**Nothing running.** 4070 free. No cluster (scripts/ deferred). T-012 (cross-frame KV cache) DONE +
verified. Awaiting Merlin's steer on whether to start the relay rollout-training method (the next H3
position step) — NOT starting it unprompted (it's a new method/phase, his call).

## NEXT ACTION
**Report T-012 done; propose the rollout-training method as the next decision and let Merlin steer.**
T-012 gave us the verified efficient-rollout substrate. The natural next step is to DESIGN the
**sequential stop-grad register-relay rollout training** (IDEAS.md) — the candidate H3 *position*
method — built on `stream_rollout_step` (lift no_grad + detach the cache between steps = stop-grad
TBPTT-1). That's a new method/phase, so bring a plan to Merlin rather than barrelling in. Do NOT start
building the training method until he weighs in.

## Recently done
- **T-012 / D-020 — cross-frame sliding-window KV eviction cache — DONE + verified.** `stream_rollout_
  init`/`_step` + `generate_streaming` + `generate_windowed` (uncached twin); gate `test_stream_cache.py`
  9/9 (forward-level eviction bit-exact incl. past-table + actions; cached==uncached-twin under shared
  seed; mutation test catches broken cache; ~1.1× faster). No regression. Tripwire 2 cleared
  (frozen-noise deviation benign on trained vanilla_s0).
- **D-021 (Merlin) — test-validity refinement:** seeded per-frame noise keyed on absolute frame id +
  `generate_windowed` so the cached rollout is bit-checked against a REAL independent non-cache path,
  not a test reimplementation; mutation test proves divergence is detectable. CLAUDE.md synced.
- **ESC-009/010 RESOLVED** (Merlin: "resolve as whatever; continue"). Position metric NOT frozen
  (uncertain strength); EXP-014 read accepted (FF7 gain = loss). GOAL H3-position note added.
- EXP-013 (D-018) position-memory metric built+applied; EXP-014 (D-019) FF7-gain disentangle. Both done.
- T-008/D-017 — within-frame KV cache (`generate_cached`, bit-for-bit). The foundation D-020 builds on.

## Open threads / parked
- **Sequential register-relay rollout training** (IDEAS.md): the H3 *position* method; D-020 prepares
  its efficient-rollout substrate. Starts after T-012 is verified.
- EXP-013 position metric: built, of uncertain strength, parked (Merlin's call). Revisit only if a
  position method needs a yardstick.
- Tokenizer-C KV cache (BOARD, optional; not on the rollout-training path).

## Current worries
1. **Frozen-noise semantics (D-020 tripwire 2):** the streaming cache draws each frame's context-noise
   once at commit instead of redrawing every step — must confirm the divergence from `generate()` is
   negligible, else the semantics choice escalates.
2. **Autograd for later training use (D-020 tripwire 3):** built as no_grad inference infra now; the
   relay-training method will need grad through the step (with detach-based stop-grad on the cache).
   Note the interface so that method isn't surprised.
