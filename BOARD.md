# BOARD.md — task board

Updated: 2026-06-12 (FF7 go-ahead received; T-009 build in progress)

## In progress
- **EXP-011 — position-deficit diagnostic** (D-015, no training, local 4070). Confirm/localize
  (tokenizer C vs dynamics D)/disambiguate ((a) no motion learned vs (b) open-loop chaos) the
  general position-tracking deficit Merlin flagged at ESC-006, BEFORE any architecture/retrain.

## Awaiting review
- **EXP-010 — FF7 v1 screening DONE & reconciled** (both arms, 2026-06-13). Supports H3
  (color-only): clears T-004 bar at n_occ 12&16, misses 24; k3>k1; no tripwires. Position
  at chance. → ESC-006 RESOLVED (Merlin redirected to the position-deficit root cause).

## Backlog
- **(post-EXP-011) Position deficit fix OR metric change** — lever picked by EXP-011 results:
  (a)→ base-dynamics fix (temporal-attention density/placement, training, motion loss);
  (b)→ open-loop GT-matched position is the wrong metric, switch to closed-loop/distributional.
- **(deferred) FF7 follow-ups from EXP-010:** replicate k=3 at a 2nd seed; position/motion-
  carrying FF7 variant; n_occ-24 near-miss. Gated behind the position-deficit resolution.

## Done (recent)
- **T-009 — FF7 v1 built** (2026-06-12, commit ec45dc1): all 3 acceptance criteria pass
  by artifacts (5/5 unit smokes; 1-epoch train finite; probe dry-run end-to-end through
  `generate_memory`). Spec: tasks/T-009.md.

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
