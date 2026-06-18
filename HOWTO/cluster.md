# HOWTO: Compute

## Local (this machine)

Windows laptop "ZaubererPC", RTX 4070, repo venv at `venv\` (PowerShell:
`venv\Scripts\activate`; WSL available for bash). Used for development, smoke
tests, small ablations. Local val/mse plateaued ~7.9e-4 on short tokenizer runs;
100-epoch runs are overnight territory → cluster.

## Cluster (university, SLURM, H100s)

**Status (2026-06-18): wrapper scripts BUILT (T-003 / D-035) — see `scripts/` + `scripts/README.md`.**
They are the ONLY sanctioned cluster interface (protocol §6). Two clusters, **no default** — every
verb needs `--cluster {feranti|galvani}`:

| name (Merlin) | GPUs | W&B hosts seen | storage path |
|---|---|---|---|
| **feranti** | H100 | mlcbm002, mlcbm014 (was labelled "MLCloud") | `/weka/martius/mot936/dreamerv4-tests` |
| **galvani** | A100 | galvani-cn109 | `/mnt/lustre/work/martius/mot936/dreamerv4-tests` |

Both under user `mot936`, repo checked out remotely as `dreamerv4-tests`.

**Before first use (Merlin):** `cp scripts/cluster.env.example scripts/cluster.env`, fill the blanks
(hostnames, partition, account, etc. — `cluster.env` is gitignored), then open the master socket
once per cluster: `scripts/open_master.sh --cluster feranti` (interactive, completes 2FA; persists
~8h). The orchestrator cannot authenticate — a wrapper printing `ERROR: AUTH_DEAD` means re-open it.
Code sync = remote `git fetch + checkout` from GitHub origin (Merlin's chosen model). The venv is
built/reused on the node keyed on `sha256(requirements.txt)`.

**NOT yet live-tested** end-to-end (needs Merlin's cluster.env + an open socket); offline-verified:
arg/guard logic, error contract, sbatch rendering (`submit_job.sh --dry-run`).

Observed throughput (tokenizer, 100 epochs, occluded.npy): galvani-cn109 ≈ 30
samples/s (10.1 h); mlcbm014 ≈ 179 samples/s (1.7 h). Cause of the gap not
established (GPU model / IO not recorded) — record GPU model in future runs.
Dynamics run on mlcbm002: ≈ 585 samples/s, 100 epochs in 31 min.
