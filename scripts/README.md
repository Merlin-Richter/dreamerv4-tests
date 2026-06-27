# scripts/ — cluster interface wrappers (T-003 / D-035)

The **only** sanctioned way to touch the cluster (protocol §6). No raw ssh/scp/rsync/sbatch
anywhere else. Two clusters, **no default**: `ferranti` (H100s) and `galvani` (A100s) — every
verb needs `--cluster {ferranti|galvani}` (pick per live fairshare + queue; see `cluster_health.sh`).

## RUN THESE IN WSL (D-036)
These wrappers run in **WSL** — both the master socket and every verb. WSL, Git Bash, and PowerShell
are separate ssh stacks with separate socket namespaces; a socket opened in one is invisible to the
others. So the human opening the master and the orchestrator running the verbs must BOTH use WSL.
(PowerShell can't run these bash scripts and Windows-native ssh has no ControlMaster anyway.)
- **Merlin** opens the master in a WSL terminal.
- **Orchestrator** invokes each verb via WSL, e.g.
  `wsl.exe -e bash -lc "cd /mnt/c/Users/richt/OneDrive/Desktop/Code/transformer && bash scripts/<verb> ..."`.
- Note the deliberate split: local 4070 *training* stays in Windows/Git-Bash (the CUDA venv
  `venv/Scripts/python.exe`); only *cluster orchestration* lives in WSL.

## One-time setup
1. `cp scripts/cluster.env.example scripts/cluster.env` and fill in the blanks (host, partition,
   account, etc.). `cluster.env` is gitignored (holds the W&B key + hostnames).
2. In WSL, open the master socket (interactive, completes 2FA — **Merlin only**, the orchestrator
   cannot): `scripts/open_master.sh --cluster ferranti` (and/or galvani). Persists ~8h.
   A wrapper printing `ERROR: AUTH_DEAD` means re-run this (in WSL).

## Verbs
| verb | purpose |
|---|---|
| `cluster_health.sh [--cluster both]` | fairshare + queue depth + your jobs + disk/quota (BOTH by default) — run before every submit |
| `sync_code.sh --cluster X <branch> [sha]` | remote git fetch+checkout; echoes `SHA:` (record it) |
| `submit_job.sh --cluster X --name R [--gpus N --hours H] -- <cmd>` | render+sbatch; echoes `JOB_ID:` |
| `job_status.sh --cluster X [ids]` | squeue (no ids) / sacct state+exit+elapsed+MaxRSS (with ids) |
| `fetch_logs.sh --cluster X <id> [--tail N]` | slurm log, running or done |
| `wait_for_jobs.sh --cluster X <ids> [--poll S]` | block until terminal; early-exit (rc 7) on FAIL/Traceback |
| `pull_results.sh --cluster X <run> [--what all\|logs\|metrics\|checkpoints]` | rsync back a whole `runs/<run>/` dir; *.pt only on demand |
| `pull_file.sh --cluster X <remote-path> [--dest LOCAL]` | rsync back ONE file from outside `runs/` (e.g. a checkpoint at `checkpoints/<env>/x.pt`); path is repo-relative, mirrors locally by default |
| `cancel_job.sh --cluster X <id>` | scancel — refuses ids not in EXPERIMENTS.md |
| `clean_run.sh --cluster X <run>` | rm runs/<run> — refuses anything escaping the runs/ subtree |

## Error contract (machine-parseable FIRST stderr line)
`ERROR: AUTH_DEAD` (socket down → Merlin re-opens) · `ERROR: QUOTA` (clean superseded runs / escalate)
· `ERROR: BAD_REF` / `ERROR: BAD_CONFIG` (our bug — fix & retry) · anything else → escalate.
Exit codes: 2 BAD_CONFIG, 3 AUTH_DEAD, 4 QUOTA, 5 BAD_REF, 6 generic, 7 job failed (wait_for_jobs).

## Typical flow
```
scripts/cluster_health.sh                         # pick cluster by load
SHA=$(scripts/sync_code.sh --cluster ferranti feat/x | sed 's/SHA: //')
JOB=$(scripts/submit_job.sh --cluster ferranti --name EXP-024-tok --hours 6 -- \
        python -u src/training/train_tokenizer.py --wandb --epochs 10 --batch-size 32 | sed 's/JOB_ID: //')
# record SHA + JOB in EXPERIMENTS.md immediately
scripts/wait_for_jobs.sh --cluster ferranti "$JOB"
scripts/pull_results.sh --cluster ferranti EXP-024-tok --what metrics
```
