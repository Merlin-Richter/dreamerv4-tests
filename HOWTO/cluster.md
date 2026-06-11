# HOWTO: Compute

## Local (this machine)

Windows laptop "ZaubererPC", RTX 4070, repo venv at `venv\` (PowerShell:
`venv\Scripts\activate`; WSL available for bash). Used for development, smoke
tests, small ablations. Local val/mse plateaued ~7.9e-4 on short tokenizer runs;
100-epoch runs are overnight territory → cluster.

## Cluster (university, SLURM, H100s)

**Status: all access has been manual by Merlin so far.** The protocol's wrapper
scripts (`scripts/sync_code.sh` etc.) do not exist yet — task T-003. Until they
do, the orchestrator submits nothing; cluster runs are requested via escalation.

Two distinct systems observed in W&B run metadata (both under user `mot936`,
repo checked out as `dreamerv4-tests`):

| system | hosts seen | storage path |
|---|---|---|
| Galvani | galvani-cn109 | `/mnt/lustre/work/martius/mot936/dreamerv4-tests` |
| MLCloud | mlcbm002, mlcbm014 | `/weka/martius/mot936/dreamerv4-tests` |

Observed throughput (tokenizer, 100 epochs, occluded.npy): galvani-cn109 ≈ 30
samples/s (10.1 h); mlcbm014 ≈ 179 samples/s (1.7 h). Cause of the gap not
established (GPU model / IO not recorded) — record GPU model in future runs.
Dynamics run on mlcbm002: ≈ 585 samples/s, 100 epochs in 31 min.
