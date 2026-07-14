# Vast.ai as a third compute backend (no scheduler)

Requested by Merlin 2026-07-10, triggered by a ferranti outage (galvani alive but the autoresearch
loop's queue-latency concern + wanting a dedicated box motivated renting rather than just switching
`--cluster galvani`). Merlin rented an RTX 5090 on Vast.ai and asked how to wire it into the existing
`scripts/` cluster-wrapper discipline.

## Problem
All cluster access is sanctioned only through `scripts/` wrappers (protocol §6), which are built
entirely around SLURM (`sbatch`/`squeue`/`sacct`) on a shared login node. A Vast.ai rental is a
single box you already own outright — no scheduler, no queue, just SSH into a Docker container.

## Design (agreed in-session)
- **Reuse what's scheduler-agnostic**: `sync_code.sh`, `pull_results.sh`, `pull_file.sh`,
  `cluster_health.sh` are pure git/rsync — extended `_common.sh` (added `vast` to the `--cluster`
  whitelist, optional `SSH_PORT`/`SSH_KEY` config, optional `SSH_CALL_TIMEOUT` opt-in hard-bound on
  `ssh_cluster()`) so they work for vast **unmodified**.
- **New verbs for the scheduler-shaped ones**: `vast_run.sh` (submit_job.sh equivalent — detached
  setsid+nohup process, self-registered PID as the job id, one-job-at-a-time via an atomic remote
  `mkdir` lock), `vast_status.sh`, `vast_wait.sh`, `vast_cancel.sh`. `vast_job_template.sh` mirrors
  `job_template.sbatch`'s venv-by-requirements-hash convention minus SBATCH directives.
- **Auth model differs**: plain SSH key, no 2FA — the agent may open/re-open the vast master socket
  itself and self-heal `AUTH_DEAD` (unlike ferranti/galvani, where only Merlin can complete 2FA).
- **Shell**: runs in WSL, same as ferranti/galvani (see "environment correction" below).

## Done means
Dedicated key generated + trusted; repo cloned on the box; a real job (venv build + train-shaped
command) launched, polled, and its results pulled back, end-to-end, through the actual wrapper
scripts (not just raw ssh probing); docs (`scripts/README.md`, `cluster.env.example`,
`CLAUDE.md`, `HOWTO/cluster.md`) updated to describe vast as a third backend.

## RESULT (2026-07-10) — DONE, live-verified

**Built**: `scripts/vast_{run,status,wait,cancel}.sh` + `vast_job_template.sh`; `_common.sh`/
`open_master.sh` extended (additive only — ferranti/galvani behavior unchanged, confirmed by
inspection since `SSH_CALL_TIMEOUT`/`PORT`/`IDENTITY` are all opt-in-empty-by-default).
`scripts/cluster.env`'s `VAST_*` stanza filled in by Merlin (host `83.233.228.250:28631`, direct
connect preferred over the `ssh3.vast.ai` proxy).

**Live-verified**: RTX 5090 (Blackwell, cc 12.0) + CUDA 13.0 driver; venv-by-hash bootstrap
installed `torch==2.13.0+cu13`/`torchvision==0.28.0` and ran correctly on the GPU; full
`vast_run.sh` → `vast_status.sh` → `vast_wait.sh` → `pull_results.sh` cycle; `sync_code.sh` /
`pull_results.sh` reused with zero modification.

**Three real bugs found + fixed via live testing** (not just written and assumed correct):
1. `open_master.sh`'s two `-O check`/`-O exit` diagnostic calls didn't carry `-p $PORT`, so their
   `%p` token expansion defaulted to 22 — misreported a genuinely-alive vast master (port 28631) as
   dead. Fixed: shared `check_opts` array carries the port.
2. **Real TOCTOU race** in the busy-guard: two `vast_run.sh` invocations fired ~1s apart both
   passed the "any live PID?" check before either had registered its own PID, and both launched
   concurrently on the one GPU (reproduced live: pids 2543/2565, then again 2729 vs a correctly
   *refused* second call). Fixed with an atomic remote `mkdir` lock around the check+launch critical
   section (mkdir is atomic even though the ssh transport itself is flaky — see bug 3).
3. **Login-banner contamination in `vast_wait.sh`**: the alive-check captured `2>&1`, and this
   box's ssh mux layer prints a harmless "Connection reset by peer" + login banner on essentially
   every call (see below) — multi-line noise mixed into what should've been a bare `ALIVE`/`DEAD`
   answer, so `[ "$alive" = DEAD ]` never matched and the loop **polled forever past job
   completion**. Caught because Merlin noticed the elapsed time didn't add up ("its beed another 3
   minutes. Something is broken") and pushed back rather than accepting "just flaky" — investigating
   ground truth (the job HAD finished cleanly) found the real logic bug. Fixed: take the last line
   of the captured output as the answer, keep the full capture only for failure diagnostics.

**One design correction mid-session**: initially recommended running vast ops in Git Bash (not
WSL), reasoning that vast's no-2FA auth made the WSL-socket-namespace rule moot. Wrong in practice:
Git Bash has no `rsync` (breaks `pull_results.sh`/`pull_file.sh`), and WSL's `drvfs` mount can't
hold real Unix permissions (`chmod 600` silently doesn't stick), so a key copied to `/mnt/c/...`
gets refused by ssh as world-readable. Fixed by copying the key into WSL's *own* native `~/.ssh/`
(permissions stick correctly there) and standardizing vast on WSL like the other two backends —
`cluster.env`'s `VAST_SSH_KEY=~/.ssh/id_ed25519_vast` needed no change since it was already
tilde-relative.

**Known (documented, not a bug)**: this box's ssh mux layer reliably fails to open a *session*
channel over an established ControlMaster, falling back to a fresh direct connection every single
call (~1-3s tax, occasionally worse). Every call still completes correctly. `vast_run.sh`'s
pid-detection and `vast_wait.sh`'s poll loop are written around this (single remote-side loops
instead of repeated local polling calls, which would otherwise race against short jobs finishing).

**Remote box state at handoff**: clean (`runs/` empty — all smoke-test dirs removed), repo cloned
at `/workspace/dreamerv4-tests`, venv pre-built and cached
(`/workspace/venvs/venv-b05c6eb3f672f99e`). Storage is **NOT persistent across recycle/destroy**
(`workspace_is_volume: false`) — only `stop`/`start` preserves it; always `stop` between sessions.

**NOT done / open for Merlin**: the autoresearch loop's `autoresearch/loop/run_experiment.sh` still
hardcodes the SLURM verbs (`submit_job.sh`/`wait_for_jobs.sh`/`fetch_logs.sh`) against
`--cluster ferranti` — it does NOT yet know how to target vast. If the intent is to actually run the
loop on this rented box (plausible, given the outage was the trigger), that needs a vast-flavored
run_experiment.sh (or a branch by `--cluster`) using `vast_run.sh`/`vast_wait.sh` instead — not
started, needs Merlin's go-ahead on which shape (separate script vs. branching the existing one).
