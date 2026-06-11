# BOARD.md — task board

Updated: 2026-06-11 (initial backfill)

## In progress
*(nothing actively running — paused on ESC-002 review)*

## Awaiting review
- **ESC-002 — EXP-008 result (present-then-stop, §5).** D-010 supported: EXP-007
  rollout failure is an inference bug (context_noise=0.1 = 90% noise on context);
  high tau_ctx restores ball identity on the existing checkpoint, no retrain.
  Awaiting Merlin: agree?; fix default→0.9 + confirm rollout + proceed to T-002?;
  context_noise semantics (signal-level vs noise-fraction)? Branch paused.

## Backlog
- **T-001b — Broad dynamics diagnosis (b)/(c), IF needed.** Only if the
  context-noise fix does NOT restore ball identity (D-010 tripwire): (b) tokenizer
  latent geometry (adjacent-frame vs random-pair latent distance; ball-position
  traversals — `src/test/latent_explorer` is a start), (c) undertraining (W&B
  loss curve still falling at epoch 100?). Held until EXP-008 reconciled.
- **T-002 — Build & freeze revisit-consistency probe suite** (protocol §8).
  Blocked on: T-001 verdict. observe → occlude k frames → reveal → measure recall
  of ball color/position vs. k. Frozen before any H2/H3 method experiment.
- **T-003 — Cluster wrapper scripts (`scripts/`).** sync_code, submit_job,
  job_status, fetch_logs, wait_for_jobs, pull_results, cancel_job,
  cluster_health, clean_run. Until done: all cluster interaction is manual by
  Merlin; orchestrator submits nothing.
- **T-004 — Pre-register H2/H3 success criteria in GOAL.md** (with Merlin; after
  T-002 exists, before any deciding experiment).

## Blocked
*(none)*

## Done
- W&B integration + plateau fix (D-006, 2026-06-09)
- LPIPS tokenizer adopted, `trained_autoencoder.pt` frozen (EXP-006, 2026-06-10)
- Action-conditioned dynamics trained — failed at rollout (EXP-007, 2026-06-10)
- latent_explorer browser tool (2026-06-11, uncommitted → T-005)
- Protocol adoption + state backfill (D-009, ESC-001, 2026-06-11)
- T-005 — commit debt cleared (latent_explorer + train script committed; tree clean)
- **T-001 / EXP-008 — context-noise rollout diagnostic** (2026-06-11): D-010 cause
  confirmed (inference bug). Report `tasks/T-001-report.md`. → ESC-002 review.
