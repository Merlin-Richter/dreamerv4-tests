# scripts/ — cluster interface wrappers (T-003 / D-035)

The **only** sanctioned way to touch the cluster (protocol §6). No raw ssh/scp/rsync/sbatch
anywhere else. Three backends, **no default**: `ferranti` (H100s, SLURM), `galvani` (A100s, SLURM),
`vast` (a rented GPU box, e.g. RTX 5090 — **no scheduler**) — every verb needs
`--cluster {ferranti|galvani|vast}`. ferranti/galvani: pick per live fairshare + queue (see
`cluster_health.sh`). vast: there's only one box, so it's just "is it up".

**Two verb families** — pick by backend:
- **SLURM verbs** (`submit_job.sh`, `job_status.sh`, `fetch_logs.sh`, `wait_for_jobs.sh`,
  `cancel_job.sh`) — ferranti/galvani ONLY. They render/parse `sbatch`/`squeue`/`sacct`, which
  don't exist on a Vast rental — don't point them at `--cluster vast`.
- **vast verbs** (`vast_run.sh`, `vast_status.sh`, `vast_wait.sh`, `vast_cancel.sh`) — vast ONLY.
  No scheduler: "submit" launches the command as a detached background process (setsid+nohup)
  over ssh, self-registering its own PID as the job id (there's no SLURM job id to have). One
  job at a time (a single GPU can only run one thing anyway) — `vast_run.sh` refuses to launch
  while a previous run's PID is still alive.
- **Scheduler-agnostic verbs** (`sync_code.sh`, `pull_results.sh`, `pull_file.sh`,
  `cluster_health.sh`) — pure git/rsync, work unmodified for all three backends.

## RUN THESE IN WSL — including vast (D-036)
All three backends' wrappers run in **WSL** — both the master socket and every verb. WSL, Git
Bash, and PowerShell are separate ssh stacks with separate socket namespaces (own known_hosts,
own ControlMaster sockets) — a socket (or a trusted host key) established in one is invisible to
the others. So the human/orchestrator opening a master and the orchestrator running the verbs must
use the SAME shell consistently. (PowerShell can't run these bash scripts and Windows-native ssh
has no ControlMaster anyway.)
- **Merlin** opens the ferranti/galvani master in a WSL terminal (2FA is interactive, the
  orchestrator cannot complete it).
- **Orchestrator** invokes every verb, all three backends, via WSL, e.g.
  `wsl.exe -e bash -lc "cd /mnt/c/Users/richt/OneDrive/Desktop/Code/transformer && bash scripts/<verb> ..."`.
- Note the deliberate split: local 4070 *training* stays in Windows/Git-Bash (the CUDA venv
  `venv/Scripts/python.exe`); only *cluster orchestration* (all three backends) lives in WSL.

**vast's auth is still the exception, just not its shell.** Plain SSH-key auth, no 2FA — the
orchestrator MAY open/re-open the vast master socket itself (`open_master.sh --cluster vast`) and
self-heal an `AUTH_DEAD` without escalating to Merlin, unlike ferranti/galvani.
**Two gotchas specific to vast, both solved by using WSL uniformly:**
1. `pull_results.sh`/`pull_file.sh` need `rsync`, which plain Windows Git Bash doesn't ship —
   another reason to run vast ops in WSL rather than Git Bash, not just consistency.
2. The dedicated key (`VAST_SSH_KEY`) must live in **WSL's own native home** (`~/.ssh/`), NOT
   under `/mnt/c/...` — WSL's `drvfs` mount can't hold real Unix permissions (`chmod 600` silently
   doesn't stick, stays `777`), and `ssh` refuses a world-readable private key outright ("bad
   permissions", auth silently fails). `cluster.env`'s `VAST_SSH_KEY=~/.ssh/id_ed25519_vast` is
   written tilde-relative for exactly this reason — it resolves correctly as long as a
   correctly-permissioned copy of the key exists at `~/.ssh/id_ed25519_vast` in WHICHEVER shell is
   running (WSL's own home, not the Windows-side one Git Bash generated it into). If you ever
   rotate this key, copy the new one into WSL's native `~/.ssh/` too (`chmod 600` there — it's a
   real ext4/tmpfs home, permissions stick normally) and re-run `open_master.sh --cluster vast`.

## One-time setup
1. `cp scripts/cluster.env.example scripts/cluster.env` and fill in the blanks (host, partition,
   account, etc.). `cluster.env` is gitignored (holds the W&B key + hostnames) — see the file's
   comments for the vast-specific keys (`VAST_SSH_PORT`, `VAST_SSH_KEY`, ...).
2. ferranti/galvani, in WSL: open the master socket (interactive, completes 2FA — **Merlin only**):
   `scripts/open_master.sh --cluster ferranti` (and/or galvani). Persists ~8h. A wrapper printing
   `ERROR: AUTH_DEAD` means re-run this (in WSL).
   vast, in WSL too (D-036): `scripts/open_master.sh --cluster vast` (no 2FA — anyone/anything
   holding the key can run this, including the orchestrator).
3. vast only, one-time: `git clone` the repo onto the box at `VAST_REMOTE_PATH` before the first
   `sync_code.sh` (it fetches/checks-out an existing clone, it doesn't create one). Re-clone if the
   instance was `destroy`ed/recycled (disk not persistent — see cluster.env.example's warning).

## Verbs
| verb | backend | purpose |
|---|---|---|
| `cluster_health.sh [--cluster both]` | all | fairshare + queue depth + your jobs + disk/quota (BOTH by default) — run before every submit |
| `sync_code.sh --cluster X <branch> [sha]` | all | remote git fetch+checkout; echoes `SHA:` (record it) |
| `submit_job.sh --cluster X --name R [--gpus N --hours H] -- <cmd>` | ferranti/galvani | render+sbatch; echoes `JOB_ID:` |
| `job_status.sh --cluster X [ids]` | ferranti/galvani | squeue (no ids) / sacct state+exit+elapsed+MaxRSS (with ids) |
| `fetch_logs.sh --cluster X <id> [--tail N]` | ferranti/galvani | slurm log, running or done |
| `wait_for_jobs.sh --cluster X <ids> [--poll S]` | ferranti/galvani | block until terminal; early-exit (rc 7) on FAIL/Traceback |
| `cancel_job.sh --cluster X <id>` | ferranti/galvani | scancel — refuses ids not in EXPERIMENTS.md |
| `vast_run.sh --name R -- <cmd>` | vast | launch detached (setsid+nohup), venv-by-hash bootstrap same as SLURM; echoes `JOB_ID: <name>`; refuses if the box is already busy |
| `vast_status.sh [RUN] [--tail N]` | vast | no RUN = list all runs RUNNING/DONE + last log line; RUN = that run's status + log |
| `vast_wait.sh <RUN> [--poll S]` | vast | block until the run's pidfile is gone; early-exit (rc 7) on FAIL/Traceback, same as wait_for_jobs.sh |
| `vast_cancel.sh <RUN>` | vast | SIGTERM the run's pid — refuses names not in EXPERIMENTS.md |
| `pull_results.sh --cluster X <run> [--what all\|logs\|metrics\|checkpoints]` | all | rsync back a whole `runs/<run>/` dir; *.pt only on demand |
| `pull_file.sh --cluster X <remote-path> [--dest LOCAL]` | all | rsync back ONE file from outside `runs/` (e.g. a checkpoint at `checkpoints/<env>/x.pt`); path is repo-relative, mirrors locally by default |
| `clean_run.sh --cluster X <run>` | ferranti/galvani | rm runs/<run> — refuses anything escaping the runs/ subtree |

## Error contract (machine-parseable FIRST stderr line)
`ERROR: AUTH_DEAD` (socket down → Merlin re-opens on ferranti/galvani; self-heal with
`open_master.sh --cluster vast` on vast) · `ERROR: QUOTA` (clean superseded runs / escalate)
· `ERROR: BAD_REF` / `ERROR: BAD_CONFIG` (our bug — fix & retry) · anything else → escalate.
Exit codes: 2 BAD_CONFIG, 3 AUTH_DEAD, 4 QUOTA, 5 BAD_REF, 6 generic, 7 job failed (wait_for_jobs /
vast_wait).

## Typical flow (ferranti/galvani, SLURM)
```
scripts/cluster_health.sh                         # pick cluster by load
SHA=$(scripts/sync_code.sh --cluster ferranti feat/x | sed 's/SHA: //')
JOB=$(scripts/submit_job.sh --cluster ferranti --name EXP-024-tok --hours 6 -- \
        python -u src/training/train_tokenizer.py --wandb --epochs 10 --batch-size 32 | sed 's/JOB_ID: //')
# record SHA + JOB in EXPERIMENTS.md immediately
scripts/wait_for_jobs.sh --cluster ferranti "$JOB"
scripts/pull_results.sh --cluster ferranti EXP-024-tok --what metrics
```

## Typical flow (vast, no scheduler — in WSL like everything else, per D-036)
```
scripts/cluster_health.sh --cluster vast           # or just check vast_status.sh — no fairshare/queue to read
SHA=$(scripts/sync_code.sh --cluster vast exp/x | sed 's/SHA: //')
scripts/vast_run.sh --name EXP-030-sym -- \
        python -u src/training/train_tokenizer.py --wandb --epochs 10 --batch-size 32
# record SHA + run name in EXPERIMENTS.md immediately
scripts/vast_wait.sh EXP-030-sym
scripts/pull_results.sh --cluster vast EXP-030-sym --what metrics
```
