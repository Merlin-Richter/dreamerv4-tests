# ORIENT.md

Rewritten: 2026-06-18 (built the cluster wrapper scripts, T-003/D-035).

## What we are doing right now and why
**Active task: cluster interface scripts (T-003), at Merlin's direction ("Its time to work on the
cluster interface scripts").** This is the long-deferred wrapper layer (protocol §6) — needed because
the GridWorld pipeline (tokenizer + vanilla dynamics on the full 6.9 GB set) is overnight/OOM territory
on the 4070 (~25 h local for a 10-ep tokenizer). Building it implicitly resolves ESC-016 Q2 (compute
tier = cluster).

## Status — BUILT this session, awaiting Merlin for the live test
- `scripts/` now holds all 9 verbs + `_common.sh` (connection core) + `cluster.env.example` +
  `job_template.sbatch` + `open_master.sh` + `README.md`. D-035 records the design; `tasks/T-003-plan.md`
  the plan. Offline-verified: arg/guard logic, the AUTH_DEAD/QUOTA/BAD_REF/BAD_CONFIG error contract
  (machine-parseable first stderr line), and sbatch rendering (`submit_job.sh --dry-run`, incl. special
  chars). Two clusters, **no default**: `feranti` (H100) / `galvani` (A100); `--cluster` required.
- **BLOCKED on Merlin for live Phase-1 test** (read-only verbs cluster_health/job_status):
  he must (1) `cp scripts/cluster.env.example scripts/cluster.env` and fill the blanks (hostnames,
  partition, account, …), (2) `scripts/open_master.sh --cluster <c>` (interactive 2FA — I cannot auth).
  Until then the connection assumption (a reusable ControlMaster socket) is unverified.

## Next action (once Merlin unblocks)
Phase 1: run `cluster_health.sh` + `job_status.sh` live → confirm the connection layer. Phase 2:
`sync_code.sh` + a tiny `submit_job.sh` smoke (nvidia-smi) → confirm JOB_ID/log/wait/sacct. Phase 3:
fetch/pull/cancel/clean live-checks. Then first real job = the GridWorld **tokenizer** on the cluster.

## GridWorld research thread (paused under the cluster task; still Merlin-gated)
- **Tokenizer SMOKE COMPLETED** while no session was alive (W&B `zjvhcn4s`, ~17 min on the 300-ep
  subset): val/mse **0.00216**, latent_cos 0.37 (not collapsed). Recon views in `experiments/EXP-023/`.
  Pipeline validated end-to-end on the subset. (Processed per cold-start §1.4 — flagged to Merlin,
  not acted on, since he redirected me to cluster scripts.)
- **ESC-016 Q1 STILL PENDING:** the GridWorld eval design sign-off (D-033, position-first headline) —
  do NOT freeze the eval or wire the model adapter until Merlin blesses it. Q2 (compute tier) answered.
- Uncommitted `src/training/train_tokenizer.py` (+101 lines: LPIPS/W&B/subset for the smoke) +
  `experiments/EXP-023/` + `_gridworld_tok_smoke.log` belong to the GridWorld thread — NOT part of the
  cluster-scripts commits; left for that thread's owner to land.

## Current worries
1. The ControlMaster/2FA connection model in `_common.sh` is the standard pattern but UNVERIFIED
   against the real clusters — if the actual login flow differs (per-command token, multiplex-breaking
   jump host), the connection helper needs rework. The D-035 tripwire covers this; live test settles it.
2. Don't over-build past Phase 1 before the connection layer is confirmed live.
3. GridWorld eval still unblessed (ESC-016 Q1) — gates the downstream dynamics eval, not tokenizer training.

## Parked (pre-pivot threads — resume only if Merlin redirects)
- C1/motion (EXP-021 ckpt ~ep10), exposure-bias/open-loop compounding. ESC-014 op-3 relay open.
- Occluded-line H3 (FF7 color SUPPORTED; FF9 v2 static-color SUPPORTED; position open). Under checkpoints/occluded/.
