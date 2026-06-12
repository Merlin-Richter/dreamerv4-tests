# BOARD.md — task board

Updated: 2026-06-12 (H2 closed; H3 entered; FF7 designed — pre-context-reset)

## Awaiting Merlin (blocking — see ESCALATIONS ESC-005)
- **FF7 build go-ahead** — first H3 method designed & code-grounded; full v1 in `IDEAS.md`
  "Proposed first attempt". On go-ahead → write **D-014**, spawn build worker (T-009),
  smoke on 4070, run **EXP-010**, present-then-stop. NOT yet committed/built.
- **Harness improvement pick** — code-citation rule only, or rule + a `methods-critic`
  agent (red-team a method design before its D-NNN commits). My write-up in ORIENT §In-flight.

## Backlog (unblocks on the above)
- **T-009 — Implement FF7 v1** (training-procedure change to `train_dynamics_model.py`;
  no architecture change; window-1 sufficiency loss, overwrite-real-latents, k=1, dataset
  with varied curtain timings). Spec to be written at go-ahead. Eval: frozen probe 5503e75.

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
