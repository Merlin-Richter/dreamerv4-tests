---
name: bash-timeout-long-waits
description: Bash tool kills commands at timeout even with run_in_background (default 120s, max 600s) — never hold a >10-min cluster wait in one call; launch detached + Monitor-poll instead
metadata:
  type: feedback
---

Never hold a long cluster wait (wait_for_jobs.sh / vast_wait.sh / run_experiment.sh) inside a
single Bash tool call. `run_in_background: true` does NOT lift the timeout — the command is
still killed at the timeout (default 120000 ms if unset, hard max 600000 ms), exiting rc=124.

**Why:** 2026-07-10 autoresearch shakedown — the first live loop iteration
(`run_experiment.sh --cluster vast`, ~20 min end-to-end) was launched via
`run_in_background: true` with no timeout set and was killed at exactly 120 s, mid-launch.
The remote job survived (setsid+nohup detached), but the local waiter died silently and could
have orphaned vast_run's launch lock. The earlier "25-min silent shakedown stall" (commit
ffd1f20) showed the same rc=124 signature.

**How to apply:** for anything that outlives ~2 min: (1) launch so the remote side is detached
and survives the local caller (submit_job/vast_run already do this); (2) re-attach with a
Monitor poll loop (e.g. vast_status/job_status every 60 s, emit on terminal state, generous
timeout_ms up to 3600000) or short idempotent re-checks; (3) if a single Bash call must wait,
set timeout explicitly and keep the command's expected runtime under it. Related: the loop's
one-command contract in program.md assumes the agent shell can hold a 20-min foreground
process — flagged to Merlin as a launch/collect split candidate.
