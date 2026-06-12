# BOARD.md — task board

Updated: 2026-06-12 (post H1-closure milestone, ESC-003 / D-011)

## In progress
- **T-007 — `context_noise`→`context_signal` rename + default 0.9 + comment fix**
  (D-011 cleanup). Inference-only, no retrain. Two usage sites
  (`dynamics_model.py` L63 config, L355 `_denoise_next`). First step of Phase 2.

## Awaiting review
*(none — ESC-002 and ESC-003 both resolved)*

## Backlog (Phase 2 — H2, cheap-signal-first per D-011)
- **T-002 — Build & freeze the revisit-consistency probe suite** (protocol §8, spine).
  On the EXISTING frozen tokenizer + `my_dynamics.pt`; choose inference window N, roll
  out occlusion length k spanning below→above N (no retrain; M<N is free).
  Primary metric: latent-token MSE (predicted reveal latent vs frozen-tokenizer GT
  latent), validated against a pixel-space color/position decomposition. Controls:
  chance floor (no-context/random latent), ceiling (fully visible), no-occlusion drift
  control. "Ball not rendered" tracked as its own failure mode. Freeze before any
  method (H3) experiment.
- **T-004 — Pre-register H2 success criteria in GOAL.md** (with Merlin; after T-002's
  calibration controls measured, before reading the H2 result).
- **(H2 baseline run)** — measure baseline on the frozen probe → present-then-stop.

## Deferred (until H3 method work needs heavy training / long horizons)
- **T-003 — Cluster wrapper scripts (`scripts/`).** Probe suite + H2 baseline run
  locally on the 4070 against existing checkpoints; no cluster needed yet. Until done:
  all cluster interaction is manual by Merlin; orchestrator submits nothing.
- **T-008 — KV caching (D and C) + continuous-RoPE rollout.** Efficiency for long
  rollouts, NOT a prerequisite for H2. MUST follow `HOWTO/rope_kv_cache_caveat.md`
  (cached K/V can't be re-rotated → needs an unbounded, never-reset absolute position
  clock; current fixed cos/sin table is cache-incompatible).

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
