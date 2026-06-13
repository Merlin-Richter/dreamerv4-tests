# ORIENT.md

Rewritten: 2026-06-13 (T-012 cross-frame KV cache DONE + verified; framing corrected by Merlin —
purpose is efficient continuous rollouts, NOT training prep)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75. H2 anchored on budget-matched `vanilla_s0`.
- **H3 color — supported (FF7 v1).** Register relay carries static hidden COLOR well past the N=8
  window (EXP-010); confound resolved (EXP-012); the FF7 1-step base-dynamics gain is the LOSS not the
  relay (EXP-014).
- **H3 position — OPEN.** EXP-013 measured blind-occlusion position memory as near-absent (vanilla ≈
  copy-last; FF7 marginally better). **Merlin (ESC-009): the position metric as coded is of uncertain
  strength — NOT frozen as a spine, NOT a hard gate.** Read stands; not litigated further.
- **Big picture (Merlin restated 2026-06-13):** high-level plan is unchanged — *experiment with
  architectures + objectives to find one with persistent memory*. Currently on the **memory-token**
  line; **its capability to persist memory is still UNPROVEN** (color-only so far). New design ideas
  captured in IDEAS.md (2026-06-13): split MEMORY tokens (full-state carrier) from REGISTER scratch;
  **memory-only sufficiency** objective (FF9); "produce vs generate" (training cost is ~2–4×, not
  ÷window, because latents are teacher-forceable — only memory must be produced by running the model).
  These are noted ideas, NOT a committed build. The KV-cache subobjective (done) is the production-path
  engine for the eventual sequential variant.
- **JUST DONE (T-012 / D-020), the first subobjective: cross-frame sliding-window KV eviction cache.**
  We can now run **efficient sliding-window continuous (open-ended) rollouts** via KV caching — O(1)
  attention per step, no per-frame window rebuild. Implemented + verified (9/9). Infra, not an
  experiment. (Built for rollout efficiency; training internals — objectives, gradient graphs — are
  deliberately out of scope for this step.)

## In flight
**Nothing running.** 4070 free. No cluster (scripts/ deferred). **EXP-016 (batch-limit parallelism sweep,
D-023) DONE → blocked on Merlin (ESC-012) present-then-stop.** (ESC-011 RESOLVED — he directed this batch
cut.)

## NEXT ACTION
**Await Merlin's ESC-012 verdict on EXP-016 (batch-limit parallelism sweep).** Decisive read: more
parallelism → MORE speedup — cached/windowed steps/s ratio rises monotone 5.85×(B32)→14.0×(B512) because
only cached scales (frames/s 731→1427) while windowed is throughput-FLAT (~105–126, saturated). At each
method's own VRAM ceiling (cached B=512/52%, windowed B=1024/82% — cached is MORE memory-hungry at N=32,
flipping EXP-015's N=64 asymmetry) cached still wins 13.5× end-to-end. Caveat: abs steps/s ±30% run-to-run
(laptop thermal); ratio+shape stable. **Timing fixed** (Merlin flagged the runs were way too slow / hung
10 min): measure from pre-filled window + predictive VRAM guard that never launches a past-VRAM (sysmem-
thrash) config + `-u`/no-tail live streaming. Offered next cut: same sweep at N=64 (asymmetry flips) or a
repeat for error bars. Don't start follow-ups until he weighs in.

## Recently done
- **T-012 / D-020 — cross-frame sliding-window KV eviction cache — DONE + verified.** `stream_rollout_
  init`/`_step` + `generate_streaming` + `generate_windowed` (uncached twin); gate `test_stream_cache.py`
  9/9 (forward-level eviction bit-exact incl. past-table + actions; cached==uncached-twin under shared
  seed; mutation test catches broken cache; ~1.1× faster). No regression. Frozen-noise deviation from
  generate() benign on trained vanilla_s0 (latent 0.032 / pixel 1.76 < its own seed-to-seed 0.049/2.75).
- **D-021 (Merlin) — test-validity refinement:** seeded per-frame noise keyed on absolute frame id +
  `generate_windowed` so the cached rollout is bit-checked against a REAL independent non-cache path,
  not a test reimplementation; mutation test proves divergence is detectable. CLAUDE.md synced.
- **ESC-009/010 RESOLVED** (Merlin: "resolve as whatever; continue"). Position metric NOT frozen
  (uncertain strength); EXP-014 read accepted (FF7 gain = loss). GOAL H3-position note added.
- EXP-013 (D-018) position-memory metric built+applied; EXP-014 (D-019) FF7-gain disentangle. Both done.
- T-008/D-017 — within-frame KV cache (`generate_cached`, bit-for-bit). The foundation D-020 builds on.

## Open threads / parked
- **Efficient register-relay rollout training** (IDEAS.md) — the big-picture H3 *position* goal that
  the T-012 rollout cache feeds into. Parked deliberately: do it AFTER the rollout substrate, not now.
- EXP-013 position metric: built, of uncertain strength, parked (Merlin's call). Revisit only if a
  position method needs a yardstick.
- Apply `stream_rollout_step` to continuous-rollout call sites (interactive viewer) — efficiency win,
  pending Merlin's direction.
- Tokenizer-C KV cache (BOARD, optional).

## Current worries
1. **Frozen-noise semantics:** the streaming cache draws each frame's context-noise once at commit
   instead of redrawing every step. Confirmed benign on trained vanilla_s0 (within generate()'s own
   seed-to-seed noise) — resolved, but keep in mind for any rollout that's sensitive to exact noise.
