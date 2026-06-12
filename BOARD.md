# BOARD.md — task board

Updated: 2026-06-12 (FF7 go-ahead received; T-009 build in progress)

## In progress
- **T-009 — Implement FF7 v1** (D-014; inline, master): param-free `dynamics_model.py`
  extensions (register inject/return + `generate_memory`) + FF7 loss in
  `train_dynamics_model.py` (window-1 sufficiency, overwrite-real-latents, k flag,
  λ_ff7=1.0). Then EXP-010: smoke + k=1 + k=3 screening (single-seed, per Merlin's
  2026-06-12 protocol edit), frozen probe 5503e75 vs T-004 bar → present-then-stop.

## Backlog
*(empty — next items spawn from the EXP-010 verdict)*

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
