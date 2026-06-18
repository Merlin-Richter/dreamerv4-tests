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
  chars). Two clusters, **no default**: `ferranti` (H100) / `galvani` (A100); `--cluster` required.
- **VALIDATED END-TO-END (D-037), via WSL.** Full mini pipeline ran green on ferranti (job 405555,
  COMPLETED): venv-by-hash build (pip on the fixed UTF-8 requirements), `sync_code` checkout,
  cluster datagen (gridworld_mini), tokenizer 1ep train, **W&B synced via the cluster's ~/.netrc**
  (run 2n8ym02n), checkpoint `pull_results`'d back to the laptop, `clean_run` cleaned up. All verbs
  exercised live (cancel_job guard offline-only — nothing cancellable). Connection = a reusable
  ControlMaster socket that MUST live in **WSL** (D-036). Pre-req fixes folded: requirements.txt
  UTF-16→UTF-8, repo-wide LF (.gitattributes), branch pushed to origin.

## Next action — the cluster is READY for the real GridWorld run
T-003 is done. The cluster can now run the real GridWorld **tokenizer** (full data) — this is the
ESC-016 Q2 payoff. NOT starting it unprompted: present-then-stop / awaiting Merlin's go (and note
ESC-016 Q1, the eval-design sign-off, is still open — it gates the downstream dynamics RECALL eval,
NOT tokenizer training). When greenlit: generate full gridworld data on the cluster (or reuse the
local 6.9GB set?) → train tokenizer (LPIPS, ~10ep) → pull → then vanilla dynamics.

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
1. Connection model now CONFIRMED (read-only) — must always run cluster verbs in WSL, never Git Bash
   (separate socket namespaces; that mismatch caused the first AUTH_DEAD). Documented; stay disciplined.
2. Mutating pipeline (submit/sacct/logs/wait/pull) still unverified — needs the one trivial smoke job.
3. GridWorld eval still unblessed (ESC-016 Q1) — gates the downstream dynamics eval, not tokenizer training.

## Parked (pre-pivot threads — resume only if Merlin redirects)
- C1/motion (EXP-021 ckpt ~ep10), exposure-bias/open-loop compounding. ESC-014 op-3 relay open.
- Occluded-line H3 (FF7 color SUPPORTED; FF9 v2 static-color SUPPORTED; position open). Under checkpoints/occluded/.
