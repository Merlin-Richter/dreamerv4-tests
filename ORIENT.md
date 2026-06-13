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
**Nothing running.** 4070 free. No cluster (scripts/ deferred). **Efficiency subobjective CLOSED** —
EXP-016 ACCEPTED (ESC-012, "strong results"). Merlin directed: clean up + propose next direction.

## NEXT ACTION
**BUILD T-013: memory-token architecture + FF9 v2 loss on the 4070 (D-024).** ESC-013 RESOLVED — Merlin
picked P1 reframed as the **architectural BASELINE** ("this alone won't fix FF7; wanted it first for a
better baseline"), with his own fix for the V-T013 loss shortcut. **FF9 v2 (his design):** per memory
rollout pick horizon j∈{1..k}; path frames t..t+j−1 at **τ=0** (pure noise → NO GT latent anywhere memory
could cheat from); **last frame t+j at sampled τ** (training target; low-τ_j forces memory); **loss on the
last frame only, un-ramped.** Distinct MEMORY token type (registers→scratch); withhold-via-τ=0 (no
`absent_latent` token). Mechanism: within a window frame t+j attends DIRECTLY to frame t's memory → trains
"memory = sufficient full-state object," NOT the cross-window relay (that's option A, layered on next).
Build progress: ✅ architecture (memory tokens) + `_ff9_loss` v2 done, all gates green (FF9 7/7, FF7 5/5,
KV 5/5, stream 9/9; commit 7f4e4a3). **Remaining before the training run:** (3) `train_dynamics_model.py
--ff9` flag + knobs (ff9_k, lambda_ff9, ff9_ramp); (4) `generate_full_state_memory` + dispatch (memory-carry
rollout for frozen-probe eval) + memory-sufficiency probe; (5) train seed0 100ep bs32, committed
config.yaml+run.sh → EXP-017 present-then-stop. Measure: PRIMARY within-window memory sufficiency
(L(mem)≪L(no-mem)) + no base-dynamics regression; POSITIONING frozen-probe color vs FF7/vanilla_s0 (expect
≈FF7). Tripwires (D-024): mem not load-bearing / base regression / color worse than FF7 → halt.

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
