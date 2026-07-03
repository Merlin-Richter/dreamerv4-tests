# Train vanilla Dreamer-4 dynamics on Memory Maze latents

Requested by Merlin 2026-07-03. Runs in parallel with `memmaze-dynamics-mem2mem`.

## Goal
The no-memory baseline for the memmaze memory campaign: canonical `DynamicsModel` (n_memory=0),
action-conditioned, trained on the cached frozen-tokenizer latents (n_latents=32, bottleneck_dim=16).
Every future memory claim on memmaze compares against this through the identical eval.

## Prereqs
- `latent-cache-for-dynamics-training` (cache path in trainer) — DONE first.
- `memmaze-dynamics-prep` (actions npy + latent cache on cluster).

## Open config decisions (ask Merlin with throughput data before submitting)
- Model size (GridWorld default is small; memmaze has 32 latent tokens/frame — propose scaling to
  ~tokenizer scale: embedding_dim 512, depth 12, n_heads 16; needs CLI exposure like the tokenizer got).
- `--context-length` (temporal window; compute-quadratic-ish; propose 32?), batch size (search),
  epochs/compute budget.

## Done means
Trained checkpoint `checkpoints/memmaze/dynamics_vanilla.pt` pulled + verified, W&B curves healthy,
qualitative rollout sheet, provenance (job id + SHA + config) in `experiments/memmaze-dynamics/NOTES.md`
+ EXPERIMENTS.md line. (Recall-eval scoring is a separate follow-up task — needs the memmaze
recall/probe eval to exist first.)

## Provenance
- ferranti job 415103 @ SHA 1149bb4 (train_vanilla.sh 50 64, --hours 12), submitted 2026-07-03. bs64 lr3e-4 W32 512/12/16 (41.0M). ~8.5h ETA.
