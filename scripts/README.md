# scripts/ — cluster interface wrappers (T-003 / D-035)

The **only** sanctioned way to touch the cluster (protocol §6). No raw ssh/scp/rsync/sbatch
anywhere else. Two clusters, **no default**: `feranti` (H100s) and `galvani` (A100s) — every
verb needs `--cluster {feranti|galvani}` (pick per live fairshare + queue; see `cluster_health.sh`).

## One-time setup
1. `cp scripts/cluster.env.example scripts/cluster.env` and fill in the blanks (host, partition,
   account, etc.). `cluster.env` is gitignored (holds the W&B key + hostnames).
2. Open the master socket (interactive, completes 2FA — **Merlin only**, the orchestrator cannot):
   `scripts/open_master.sh --cluster feranti` (and/or galvani). Persists ~8h.
   A wrapper printing `ERROR: AUTH_DEAD` means re-run this.

## Verbs
| verb | purpose |
|---|---|
| `cluster_health.sh [--cluster both]` | fairshare + queue depth + your jobs + disk/quota (BOTH by default) — run before every submit |
| `sync_code.sh --cluster X <branch> [sha]` | remote git fetch+checkout; echoes `SHA:` (record it) |
| `submit_job.sh --cluster X --name R [--gpus N --hours H] -- <cmd>` | render+sbatch; echoes `JOB_ID:` |
| `job_status.sh --cluster X [ids]` | squeue (no ids) / sacct state+exit+elapsed+MaxRSS (with ids) |
| `fetch_logs.sh --cluster X <id> [--tail N]` | slurm log, running or done |
| `wait_for_jobs.sh --cluster X <ids> [--poll S]` | block until terminal; early-exit (rc 7) on FAIL/Traceback |
| `pull_results.sh --cluster X <run> [--what all\|logs\|metrics\|checkpoints]` | rsync back; *.pt only on demand |
| `cancel_job.sh --cluster X <id>` | scancel — refuses ids not in EXPERIMENTS.md |
| `clean_run.sh --cluster X <run>` | rm runs/<run> — refuses anything escaping the runs/ subtree |

## Error contract (machine-parseable FIRST stderr line)
`ERROR: AUTH_DEAD` (socket down → Merlin re-opens) · `ERROR: QUOTA` (clean superseded runs / escalate)
· `ERROR: BAD_REF` / `ERROR: BAD_CONFIG` (our bug — fix & retry) · anything else → escalate.
Exit codes: 2 BAD_CONFIG, 3 AUTH_DEAD, 4 QUOTA, 5 BAD_REF, 6 generic, 7 job failed (wait_for_jobs).

## Typical flow
```
scripts/cluster_health.sh                         # pick cluster by load
SHA=$(scripts/sync_code.sh --cluster feranti feat/x | sed 's/SHA: //')
JOB=$(scripts/submit_job.sh --cluster feranti --name EXP-024-tok --hours 6 -- \
        python -u src/training/train_tokenizer.py --wandb --epochs 10 --batch-size 32 | sed 's/JOB_ID: //')
# record SHA + JOB in EXPERIMENTS.md immediately
scripts/wait_for_jobs.sh --cluster feranti "$JOB"
scripts/pull_results.sh --cluster feranti EXP-024-tok --what metrics
```
