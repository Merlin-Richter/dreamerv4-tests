# BOARD.md — task board

Updated: 2026-06-11 (initial backfill)

## In progress
*(nothing — T-001 is the next action, not yet spawned; see ORIENT.md)*

## Backlog
- **T-001 — Diagnose EXP-007 dynamics rollout failure.** Spawned by D-009.
  Discriminate: (a) shortcut-forcing objective bug (verify against paper Eq. 7,
  check bootstrap target detachment), (b) tokenizer latent geometry (latent
  distance of temporally adjacent frames vs. random pairs; ball-position latent
  traversals — `src/test/latent_explorer` is a starting point), (c) undertraining
  (loss curves still falling at epoch 100?). Local-first (4070); promote only if
  a long run is needed. Acceptance: a written verdict in the task report naming
  the implicated cause(s) with machine-checkable evidence (script + numbers).
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
