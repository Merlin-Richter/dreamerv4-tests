# BOARD.md — task board

Updated: 2026-06-13 (ESC-009/010 RESOLVED → building D-020 cross-frame KV cache / T-012)

## In progress
- **EXP-017 — FF9 v2 memory-token baseline — TRAINING overnight** (D-024; launched 2026-06-14 late-night per
  Merlin's overnight-run ask). occluded, 100ep bs32 lr3e-4 seed0, n_memory=4 ff9_k=3, --fresh. bg bt57zuxt8 →
  `experiments/EXP-017/train.log`; ckpt `ff9v2_s0.pt` every epoch; ~6-7h. TRAIN ONLY (eval deferred — needs
  generate_full_state_memory). Tomorrow: build that + memory-sufficiency probe + frozen-probe color vs
  vanilla_s0/ff7_k3 → present-then-stop. (Chose FF9 v2 over the relay tonight: relay = bug-prone rushed build
  per V-T014; FF9 v2 is the safe, needed foundation.)
- **T-013 — memory-token architecture + FF9 v2 (BUILT; TRAINING as EXP-017)** (H3 architectural baseline; D-024). ESC-013
  resolved: Merlin picked P1-as-baseline with his FF9 v2 fix (variable-horizon j∈{1..k}, path frames τ=0 /
  no GT leak, last frame sampled-τ, loss on last frame only, un-ramped). Plan `tasks/T-013-plan.md` updated.
  Build progress:
  - ✅ (1) additive MEMORY token type + (2) `_ff9_loss` v2 (random j, τ=0 path, loss on 1..j, un-ramped) in
    dynamics_model.py. Gates: test_ff9_smoke **7/7** (incl. n_memory=0 byte-identity, grad→blocks+memory_tokens,
    injection changes prediction); FF7 5/5, KV 5/5, stream 9/9 — **no regression**. (commit 7f4e4a3)
  - ⏳ (3) `train_dynamics_model.py --ff9` flag + knobs (ff9_k, lambda_ff9, ff9_ramp).
  - ⏳ (4) `generate_full_state_memory` + dispatch (memory-carry rollout, analog of generate_memory) for
    frozen-probe eval; memory-sufficiency probe (L(mem)≪L(no-mem)).
  - ⏳ (5) train seed0 → EXP-017 (the ops-1&2 baseline; sequencing vs going straight to A+B still open w/ Merlin).
- **T-014 — FF9 relay training (operation 3 = write-memory-from-memory) + 50/50 split + mode alternation**
  (Merlin 2026-06-14; folds option A into the FF9 line = A+B). Design note `tasks/T-014-relay-plan.md` written.
  Mode B = detached-carry sliding-window relay (per-step grad forward + FF9 loss, detach+evict+slide, ~200
  steps, small/variable N, reuses T-012 streaming cache for detached context). Memory = activation (cacheable).
  **Verifier V-T014: pure detached carry REFUTED** — preserves state only to the trained rollout depth, then
  drifts to chance (deep-avg detached 0.587 vs BPTT 0.007; only BPTT extrapolates, tbptt-1 partial). Trap:
  within-window FF9 loss→0 is blind to deep-regime drift. Synthetic GRU+static probe; dynamic harder.
  **Blocked on Merlin ESC-014** (gradient design): P-a cheap tbptt-k sweep [rec] / P-c dynamic-state probe /
  P-b train-to-depth detach. Guardrails regardless (deep-hop gate, carry norm, K/V detach, strict-frac knob).
  NEXT on his pick: probe(s) or D-025 → build Mode B → EXP-018 (frozen-probe color n_occ 24/32/48 vs FF9-v2/
  FF7/vanilla). FF9 v2 (ops 1&2) arch+loss built + green.

## Done (2026-06-13)
- **EXP-016 batch sweep — ACCEPTED** (ESC-012, "strong results"). Closes the KV-cache efficiency
  subobjective (T-008→T-012→EXP-015/016).

## Awaiting review
- **EXP-016 — batch-limit parallelism sweep** (D-023) DONE. Fixed N=32, batch→VRAM ceiling. Speedup
  GROWS with batch 5.85×→14.0× (cached scales 731→1427 frames/s; windowed FLAT ~105–126). Cached more
  memory-hungry at N=32 → ceiling B=512 (52%) vs windowed B=1024 (82%); cached still 13.5× end-to-end.
  **Timing fixed** (Merlin flagged too-slow/hang): pre-filled-window measure + predictive VRAM guard +
  `-u`/no-tail streaming. Tool `perf_rollout.py --batch-sweep --sweep-window --batches --outdir`.
  Present-then-stop → **ESC-012** (open). Views: experiments/EXP-016/perf_batch.png, NOTES.md, results.json.

## Done (2026-06-13)
- **EXP-015 — rollout KV-cache perf tool** (D-022) DONE. cached `generate_streaming` flat ~28 steps/s
  (~900 frames/s, B=32) all N; uncached 21→5.7 steps/s; speedup 1.33×→4.79× (N 8→64); cached wins
  memory too. Tool `experiments/EXP-015/perf_rollout.py` (reusable). ESC-011 RESOLVED (→ EXP-016).

## Done (2026-06-13)
- **T-012 — cross-frame sliding-window KV eviction cache (D-020) — DONE.** `stream_rollout_init`/
  `stream_rollout_step` + `generate_streaming` in dynamics_model.py (commit-once + evict-oldest
  persistent cache; pre-rotated K/V → eviction = pure slice). **Gate green:** `test_stream_cache.py`
  9/9 — forward-level eviction equivalence bit-for-bit vs full windowed recompute (in-range, past the
  cos/sin table, with actions) + generate-level cached==uncached-twin + ~1.1× faster than
  generate_cached on a 60-frame CPU rollout. No regression (test_kv_cache 5/5, FF7 smokes 5/5).
  **Tripwire 2 cleared:** on trained vanilla_s0 the frozen-noise deviation from generate() (latent
  0.032 / pixel 1.76) is *smaller* than generate()'s own seed-to-seed noise (0.049 / 2.75) → benign.
  **D-021 (Merlin) test-validity refinement:** added `generate_windowed` (independent uncached twin) +
  seeded per-frame noise (`noise_seed`, keyed on absolute frame id) so the cached rollout is compared
  bit-exactly against a REAL non-cache path (not a test reimplementation); MUTATION test
  (`test_divergence_is_detectable`) proves the comparison catches a broken cache. CLAUDE.md synced.
  Plan: tasks/T-012-plan.md. Infra — no present-then-stop.

## Awaiting review
- *(none open — ESC-009/010 resolved by Merlin: position metric NOT frozen (uncertain strength),
  EXP-014 read accepted. EXP-013/EXP-014 results stand; not gates.)*

## Done (recent)
- **T-010 — play_dynamics_checkpoint FF7 register carry** (2026-06-13). Viewer drove vanilla
  fixed-4-window (no register relay) → random ball after curtain (Merlin's symptom). Refactored
  relay into DynamicsModel.memory_rollout_init/step (generate_memory reuses them); viewer drives
  them for FF7 ckpts. Verified (smokes + probe dry-run + headless reveal 9.9 vs 64.4). CLAUDE.md synced.

## Backlog (post-EXP-012; ESC-008 RESOLVED — Merlin redirected the position metric)
- **Position-memory CONSISTENCY metric (D-018)** — NEXT. Supersedes the "closed-loop/distributional"
  framing. Measure whether the model's believed (x,y) AND velocity stay self-consistent / physically
  coherent over the occluded steps ("what would it predict if revealed now", compared across steps),
  NOT exact GT-trajectory match. Must credit butterfly-divergence (F2) while penalizing forgetting
  (F1: static-center OR wander). Spec **tasks/T-011.md** (onset GT-anchor + best-fit constant-speed
  billiard residual + report-only GT-tracking-horizon; **counterfactual reveal-decode readout** per
  Merlin, NOT state-probe; ceiling/chance/copy-last + GT-floor/forgetting-surrogate calibration).
  **Gate: converge with Merlin → verifier audit → build + FREEZE, BEFORE any H3 position method (§8).**
  - **Status (2026-06-13):** BUILT + VALIDATED + APPLIED (EXP-013). Framing locked (anchored-physical-
    coherence). Verifier audit V-T011 folded; calibration reproduces it; readout faithful (FF7 k=1=1.9px).
    `src/probe/position_consistency.py`. **Result: blind position memory near-absent** — vanilla≈copy-last,
    FF7 marginally better (k1<copy-last; k3 best 1st step). **BLOCKED on Merlin ESC-009** (agree read /
    freeze with coherence-horizon headline / proceed to relay method or wait for EXP-014).
  - **Design caveat (Merlin, 2026-06-13):** open-loop pos_err is doubly artifacted — (1) bounded
    box domain caps error near chance; (2) the curve TURNS OVER at long horizons (h>~20, e.g.
    vanilla 28.2@h16→24.0@h24) because the ball BOUNCES off walls and returns to prior regions, so
    a desynced prediction lands near GT by coincidence — NOT the model recovering. The new metric
    must not credit this (measure self-consistency, not GT-trajectory match).
  - **Merlin's metric critique (ESC-008):** open-loop GT-matched error wrongly penalizes BOTH the
    "ball is center" non-tracker AND the accurate-but-butterfly-desynced tracker. Measure consistency
    of position+velocity over steps instead.
- **Sequential stop-grad register-relay training** (IDEAS.md) — TBPTT-1 relay so training context
  carries real relayed memory tokens (fixes the learned-init mismatch); candidate H3 *position*
  method once position is measurable. Worked out with Merlin 2026-06-13; pending ESC-008 direction.
- **(deferred) FF7 follow-ups from EXP-010:** replicate k=3 2nd seed; 2nd vanilla seed to firm the
  EXP-012 motion claim; position/motion FF7 variant; n_occ-24 near-miss.

## Done (recent)
- **T-009 — FF7 v1 built** (2026-06-12, commit ec45dc1): all 3 acceptance criteria pass
  by artifacts (5/5 unit smokes; 1-epoch train finite; probe dry-run end-to-end through
  `generate_memory`). Spec: tasks/T-009.md.

## Deferred (until H3 method work needs heavy training / long horizons)
- **T-003 — Cluster wrapper scripts (`scripts/`).** Probe suite + H2 baseline run
  locally on the 4070 against existing checkpoints; no cluster needed yet. Until done:
  all cluster interaction is manual by Merlin; orchestrator submits nothing.

## Done (recent)
- **T-008 (D) — KV caching for the dynamics model** (2026-06-13, D-017; done while waiting on
  EXP-012, Merlin-directed). Absolute-position RoPE (on-the-fly, never-reset clock) +
  `generate_cached` (reuses the context window's K/V across the K shortcut substeps that denoise
  one frame; NOT within-frame token caching). Bit-for-bit
  identical to `generate` (seeded) and to the full forward at T beyond the cos/sin table; ~2×
  faster at probe scale. Gate: `src/D_dynamics_model/test_kv_cache.py` 5/5 + FF7 smokes 5/5
  (no training-path regression). Follow `HOWTO/rope_kv_cache_caveat.md`.
  - **Follow-ups (not done, optional):** (a) cross-frame eviction cache (persist finalized
    frames' K/V across rollout steps with sliding-window eviction) — a further speedup that
    freezes the per-frame context-noise redraw (a documented, defensible deviation, so NOT
    bit-identical at generate level); foundation (absolute RoPE) is already in place. (b) KV
    cache for tokenizer C if we ever stream long observed sequences.

## Blocked
*(none)*

## Dropped
- **T-001b** — broad latent-geometry/undertraining diagnosis. Moot: EXP-008 found the
  failure was an inference bug, not a broken/undertrained model (Merlin agreed, ESC-002).

## Done
- W&B integration + plateau fix (D-006, 2026-06-09)
- LPIPS tokenizer adopted, `trained_autoencoder.pt` frozen (EXP-006, 2026-06-10)
- Action-conditioned dynamics trained — failed at rollout (EXP-007, 2026-06-10)
- latent_explorer browser tool (2026-06-11, uncommitted → T-005)
- Protocol adoption + state backfill (D-009, ESC-001, 2026-06-11)
- T-005 — commit debt cleared (latent_explorer + train script committed; tree clean)
- **T-001 / EXP-008 — context-noise rollout diagnostic** (2026-06-11): D-010 cause
  confirmed (inference bug). Report `tasks/T-001-report.md`. → ESC-002 review.
- **ESC-002 / ESC-003 — H1-closure milestone** (2026-06-12): H1 supported; Phase 2
  (H2) planned; architecture understanding corrected (D-011).
- **T-007 — `context_noise`→`context_signal` rename + default 0.9** (2026-06-12):
  applied + smoke-tested + CLAUDE.md synced. Commit f506fef.
- **T-002 — Revisit-consistency probe suite built & FROZEN** (2026-06-12): `src/probe/`,
  frozen at **5503e75** (after D-013 control-key rename). Detector gate green; latent-MSE +
  color/position decomposition; ceiling/chance/matched-horizon-drift controls.
- **T-004 — H2 success criteria pre-registered & LOCKED** (2026-06-12, Merlin-approved):
  `tasks/T-004.md`, D-012. Color ΔRGB headline; latent-MSE secondary; position confounded;
  H3 bar color ΔRGB < ~63 at n_occ ∈ {12,16,24}.
- **EXP-009 / H2 baseline — H2 SUPPORTED** (2026-06-12, ESC-004): the beyond-window cliff
  on the frozen probe. GOAL H2 → supported.
