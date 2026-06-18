# HOWTO: Compute

## Local (this machine)

Windows laptop "ZaubererPC", RTX 4070, repo venv at `venv\` (PowerShell:
`venv\Scripts\activate`; WSL available for bash). Used for development, smoke
tests, small ablations. Local val/mse plateaued ~7.9e-4 on short tokenizer runs;
100-epoch runs are overnight territory → cluster.

**Two local environments, split by concern (D-036):**
- **Windows / Git Bash** — local 4070 *training* (CUDA venv `venv/Scripts/python.exe`).
- **WSL (Ubuntu)** — *cluster orchestration* (all `scripts/` wrappers + the master socket).
  WSL/Git-Bash/PowerShell are separate ssh stacks; the ControlMaster socket only works if the
  opener and the wrapper-runner share one env, so cluster work is standardized on WSL. The
  orchestrator calls verbs via `wsl.exe -e bash -lc "cd /mnt/c/.../transformer && bash scripts/<verb> ..."`.

## Cluster (university, SLURM, H100s)

**Status (2026-06-18): wrapper scripts BUILT (T-003 / D-035) — see `scripts/` + `scripts/README.md`.**
They are the ONLY sanctioned cluster interface (protocol §6). Two clusters, **no default** — every
verb needs `--cluster {ferranti|galvani}`:

| name (Merlin) | GPUs | W&B hosts seen | storage path |
|---|---|---|---|
| **ferranti** | H100 | mlcbm002, mlcbm014 (was labelled "MLCloud") | `/weka/martius/mot936/dreamerv4-tests` |
| **galvani** | A100 | galvani-cn109 | `/mnt/lustre/work/martius/mot936/dreamerv4-tests` |

Both under user `mot936`, repo checked out remotely as `dreamerv4-tests`.

**Before first use (Merlin):** `cp scripts/cluster.env.example scripts/cluster.env`, fill the blanks
(hostnames, partition, account, etc. — `cluster.env` is gitignored), then open the master socket
once per cluster: `scripts/open_master.sh --cluster ferranti` (interactive, completes 2FA; persists
~8h). The orchestrator cannot authenticate — a wrapper printing `ERROR: AUTH_DEAD` means re-open it.
Code sync = remote `git fetch + checkout` from GitHub origin (Merlin's chosen model). The venv is
built/reused on the node keyed on `sha256(requirements.txt)`.

**Live-tested end-to-end** (2026-06-18, ferranti): full pipeline (datagen → train → W&B → pull) green.

## Run tuning notes (learned 2026-06-18, EXP-024 / D-041 — read before the next cluster run)
- **Always pass enough CPUs.** `submit_job.sh --cpus N` → `#SBATCH --cpus-per-task=N` (default 8).
  Without it SLURM gives `cpu=2` and the DataLoader (num_workers auto = `SLURM_CPUS_PER_TASK`, cap 8)
  STARVES the H100 — the 0↔100% util sawtooth, ~35% power. With `--cpus 8` the tokenizer holds **95%+
  util** (i.e. now compute-bound, not input-bound).
- **Tokenizer (LPIPS vgg) sweet spot: `--batch-size 64`.** On the H100 that's **~3.13 it/s,
  ~2.85 min/epoch, 61% VRAM, 95% util**. Because util is already saturated (LPIPS-VGG-bound), a bigger
  batch only *fits* more — it does NOT speed things up. Keep bs64 for the LPIPS tokenizer; re-profile
  separately for dynamics (different compute profile) or if LPIPS is off.
- **Right-size epochs + wall time.** GridWorld tokenizer val-MSE plateaus by ~epoch **3** (latent_cos
  still settling to ~ep10); 15–30 epochs is ample. 30 ep ≈ **1.4 h**, so `--hours 2–3` is enough
  (don't over-request, and don't UNDER-request → the run wall-kills mid-train: 405597 was a 40ep/4h
  that would've died ~ep28).
- **bf16 autocast + TF32 are already in `train_tokenizer.py`** (D-041). `torch.compile` and removing
  per-step `.item()` syncs are the remaining (smaller) compute wins, not yet applied. FlashAttention is
  unavailable (hand-rolled QK-norm/soft-cap attention).
- **venv cache:** keyed on `sha256(requirements.txt)` at `$VENV_ROOT/venv-<hash>`. First run builds
  (torch+CUDA download, a few min); reused after. Don't churn `requirements.txt` casually.
- **W&B auth = the cluster's `~/.netrc`** (user mot936) — leave `WANDB_API_KEY` empty in cluster.env.
- **Scheduling:** jobs landed on an H100 immediately both times (fairshare ~0.39); queue depth (30–60
  pending) hasn't been a problem.

## Pre-T-003 history (manual runs)

Observed throughput (tokenizer, 100 epochs, occluded.npy): galvani-cn109 ≈ 30
samples/s (10.1 h); mlcbm014 ≈ 179 samples/s (1.7 h). Cause of the gap not
established (GPU model / IO not recorded) — record GPU model in future runs.
Dynamics run on mlcbm002: ≈ 585 samples/s, 100 epochs in 31 min.
