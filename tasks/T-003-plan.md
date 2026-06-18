# T-003 — Cluster interface scripts (`scripts/`)

Status: DESIGN (awaiting Merlin's connection/SLURM specifics before build)
Owner: orchestrator. Motivated by: ESC-016 (GridWorld pipeline → cluster) + the standing
deferral on BOARD ("Deferred until H3 needs heavy training"). Now needed: the full GridWorld
tokenizer+dynamics pipeline is overnight/OOM territory on the 4070 (HOWTO/cluster.md: ~25h
local for a 10-ep tokenizer on the full 6.9 GB set).

## Why now / what exists
- The research-orchestrator protocol (§6) defines 9 wrapper verbs as the ONLY sanctioned cluster
  interface. They do not exist yet — cold-start `./scripts/job_status.sh` currently cannot run.
- Discovery (this session): NO `~/.ssh/config`, NO galvani/mlcloud host aliases in known_hosts,
  NO ControlMaster socket, NO checked-in sbatch/slurm script, NO lockfile (only `requirements.txt`).
  All cluster access has been manual by Merlin. So the connection + SLURM specifics must come from him.
- Known from W&B metadata (HOWTO/cluster.md): user `mot936`, repo checked out remotely as
  `dreamerv4-tests`. Galvani path `/mnt/lustre/work/martius/mot936/dreamerv4-tests`;
  MLCloud path `/weka/martius/mot936/dreamerv4-tests`. MLCloud ~6× faster (tokenizer 179 vs 30 smp/s).

## Architecture (the part I can build independent of his answers)
```
scripts/
  cluster.env            # SINGLE config file — all site specifics live here (Merlin fills / I template)
  _common.sh             # sourced by every wrapper: load config, ssh_cluster(), error-emit, socket check
  job_template.sbatch    # templated #SBATCH header + venv-by-requirements-hash prologue + run line
  sync_code.sh           # remote git fetch + checkout <branch> [sha]; echoes resolved SHA
  submit_job.sh          # render template, sbatch via socket; echoes JOB_ID:
  job_status.sh          # squeue + sacct (state, exit code, elapsed, MaxRSS)
  fetch_logs.sh          # cat/tail remote job logs (running or done)
  wait_for_jobs.sh       # poll job_status until terminal; early-return on FAILED or Traceback-in-log
  pull_results.sh        # rsync/scp results dir back (checkpoints only on explicit --what)
  cancel_job.sh          # scancel — REFUSES ids not present in EXPERIMENTS.md
  cluster_health.sh      # quota (du/lfs), scratch free, queue depth (squeue) — run before every submit
  clean_run.sh           # rm -rf restricted to the runs/ subtree (refuses paths outside it)
```

### Connection model (standard 2FA-gated ControlMaster — to confirm with Merlin)
- Merlin opens ONE master once and completes 2FA:
  `ssh -M -S <ControlPath> -o ControlPersist=8h -fN <host>`
- Every wrapper reuses it, no re-auth:
  `ssh -S <ControlPath> -o ControlMaster=no <host> '<remote cmd>'`
- Liveness: `ssh -S <ControlPath> -O check <host>`; if it fails → emit `ERROR: AUTH_DEAD` on the
  first stderr line and exit non-zero (NEVER attempt to re-auth — that is Merlin's, §6).
- All wrappers are pure bash around this one `ssh_cluster()` helper; no raw ssh elsewhere.

### Error contract (§6, machine-parseable FIRST stderr line)
`ERROR: AUTH_DEAD` (socket down) | `ERROR: QUOTA` | `ERROR: BAD_REF` | `ERROR: BAD_CONFIG` |
anything else → escalate with full output. Helpers in `_common.sh`: `die_auth`, `die_quota`,
`die_badref`, `die_badconfig`.

### venv-by-lockfile-hash prologue (§6: "job prologue builds/reuses a cached venv keyed on the
lockfile hash"). Plan: `HASH=$(sha256 requirements.txt); VENV=$VENV_ROOT/venv-$HASH`; if absent,
create + `pip install -r requirements.txt`; else reuse. Treat env-build failure as a dependency bug.

## Merlin's answers (2026-06-18)
- **No default cluster.** TWO clusters: **`feranti`** (H100s) and **`galvani`** (A100s). `--cluster`
  is REQUIRED on every verb — the right choice depends on his live fairshare + queue depth at each,
  so `cluster_health.sh` should report BOTH to inform the pick. (W&B-metadata "MLCloud/mlcbm*" = feranti.)
- **Code sync = remote git fetch + checkout** (remote holds a clone, pulls from GitHub origin;
  provenance = resolved SHA).

## Strategy: the config file IS the question
Everything site-specific is isolated into ONE self-documenting `scripts/cluster.env` (two stanzas,
feranti/galvani). The wrapper *logic* is connection-independent. So I build the full scaffold now;
Merlin only (a) fills `cluster.env` values and (b) opens the master socket — then Phase-1 read-only
verbs are testable live. Values needed FROM `cluster.env` (placeholders shipped):
  per cluster: SSH host, user, optional ProxyJump, ControlPath socket, remote repo path, partition,
  account/QOS, GPU gres/constraint, default time, optional `module load` lines, VENV_ROOT, RUNS_DIR.
  global: WANDB_API_KEY (or rely on remote ~/.netrc).

## venv-by-lockfile-hash prologue
No lockfile exists (only `requirements.txt`) → key the cached venv on `sha256(requirements.txt)`:
`VENV=$VENV_ROOT/venv-$HASH`; create+install if absent, else reuse. Env-build failure = dependency bug.

## Build/verify plan once unblocked
- Phase 1: `cluster.env` + `_common.sh` + `cluster_health.sh` + `job_status.sh` (read-only verbs).
  Verify by running `cluster_health.sh` / `job_status.sh` against the live socket — smallest
  blast radius, proves the connection layer before anything mutating.
- Phase 2: `sync_code.sh` + `submit_job.sh` + `job_template.sbatch`, validated with a TINY job
  (e.g. `nvidia-smi` / 1-step smoke), confirm JOB_ID echo + log fetch + wait + sacct.
- Phase 3: `fetch_logs.sh`, `wait_for_jobs.sh`, `pull_results.sh`, `cancel_job.sh` (EXPERIMENTS-guard),
  `clean_run.sh` (runs/-subtree guard). Guards unit-tested locally (refuse-bad-input) before live use.
- Then: D-035 recorded, HOWTO/cluster.md updated, first real cluster job = GridWorld tokenizer.
```
