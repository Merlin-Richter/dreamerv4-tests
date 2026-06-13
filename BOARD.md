# BOARD.md — task board

Updated: 2026-06-13 (EXP-012 done → ESC-008 present-then-stop)

## In progress
*(none — blocked on Merlin's ESC-008 verdict; 4070 idle)*

## Awaiting review
- **EXP-012 — budget-matched vanilla baseline** (D-016) DONE. Confound resolved: FF7 wins are the
  method not budget (color cliff reproduced; vanilla 1-step 4.66px ≈ my_dynamics ≫ FF7 ~1.0px).
  my_dynamics retired. Present-then-stop → **ESC-008** (open). Views in experiments/EXP-012/.

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
  - **Status (2026-06-13):** verifier audit **DONE (V-T011)**, folded in: separates remember/forget at
    n_occ≥8, fixed-S load-bearing, naive hallucination caught; 5 fixes applied (n_occ≥8 headline floor,
    non-degeneracy gate, speed-fixed, lengthen anchor, **ceiling-control-first on vanilla_s0**).
    **BLOCKED on Merlin's framing lock** (credit anchored-physical-coherence/F2 vs stricter near-GT
    bar — verifier proved they're the same choice), then implement → calibrate → freeze.
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
