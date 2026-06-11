# BOARD.md — task board

Updated: 2026-06-11 (initial backfill)

## In progress
- **T-001 — Context-noise rollout diagnostic (EXP-008 harness).** Rescoped by
  D-010 from the broad (a)/(b)/(c) battery to the single decisive cheap test:
  a headless `context_noise` sweep on the existing `my_dynamics.pt`, testing the
  code-read hypothesis that rollout context-noising uses tau_ctx=0.1 as *signal*
  level (=90% noise on context). Spec: `tasks/T-001.md`. Worker builds + smoke-
  tests `experiments/EXP-008/diagnose_context_noise.py`; orchestrator runs the
  full sweep (EXP-008) + interprets. Worktree: (see below). Started 2026-06-11.

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
- **T-005 — Commit uncommitted work.** `src/test/latent_explorer/`, modified
  `src/D_dynamics_model/train_dynamics_model.py`, `CLAUDE.md`, `.claude/` —
  needed for clean provenance of T-001.

## Blocked
*(none)*

## Awaiting review
*(none — ESC-001 resolved)*

## Done
- W&B integration + plateau fix (D-006, 2026-06-09)
- LPIPS tokenizer adopted, `trained_autoencoder.pt` frozen (EXP-006, 2026-06-10)
- Action-conditioned dynamics trained — failed at rollout (EXP-007, 2026-06-10)
- latent_explorer browser tool (2026-06-11, uncommitted → T-005)
- Protocol adoption + state backfill (D-009, ESC-001, 2026-06-11)
