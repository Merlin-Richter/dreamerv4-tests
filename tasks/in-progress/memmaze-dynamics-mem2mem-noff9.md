# Train mem2mem-rollout memory dynamics on Memory Maze WITHOUT FF9 (memory on, ff9 off)

Requested by Merlin 2026-07-04, right after the vanilla arm landed. Third arm of the memmaze
dynamics campaign: identical to `memmaze-dynamics-mem2mem` (415104) minus the FF9 sufficiency loss —
the memmaze counterpart of the GridWorld `mem2mem-rollout-noff9-fair` result (there: clean no-FF9
MATCHED the FF9 winner; the 50% full-noise rollout mode alone trained memory).

## Config (single-variable ablation vs 415104)
`train_mem2mem.sh 50 4 --lr 1e-4 --no-ff9` + checkpoint/W&B-name overrides:
clip128 bs4 lr1e-4 W=32 512/12/16, n_memory 8, `--mem2mem-frac 1.0 --no-bootstrap`, seed 0.
`--no-ff9` keeps n_memory=8 (config ff9_k stays 3 for ckpt-config parity; the loss term is skipped).
relay-grad-clip OFF (matches 415104); the flag exists if the relay explodes on this longer-relay env.

## Question
Does the GridWorld finding (FF9 unnecessary — rollout noise-mode alone trains memory) transfer to
the real 3D env? Also gates the sparse-memory-tokens design (`tasks/drafts/sparse-memory-tokens.md`):
per-frame FF9 injection is awkward with every-Nth-frame memory, so a working no-FF9 recipe simplifies
that path considerably.

## Done means
Checkpoint `checkpoints/memmaze/dynamics_mem2mem_noff9.pt` pulled + verified, W&B healthy (watch
relay stability without the FF9 scaffold — GridWorld showed init relay explosion ~3x/hop), rollout
sheets via `make_sheets.sh`, provenance in `experiments/memmaze-dynamics/NOTES.md` + EXPERIMENTS
line. Memory CLAIMS wait for the memmaze recall/probe eval — 3-way (vanilla / mem2mem / no-ff9).

## Provenance
- (job id + SHA recorded at submit below)
