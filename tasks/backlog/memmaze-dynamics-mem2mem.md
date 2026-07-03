# Train mem2mem-rollout memory dynamics on Memory Maze latents

Requested by Merlin 2026-07-03. Runs in parallel with `memmaze-dynamics-vanilla`.

## Goal
Port the GridWorld headline win to the real 3D memory env: memory-token dynamics trained with the
mem->mem sliding-rollout signal (`experiments/mem2mem/train_mem2mem.py` + `rollout.py`), on the cached
memmaze latents. GridWorld winner config = rollout-heavy + FF9, NO bootstrap (`--no-bootstrap`),
relay-grad-clip OFF by default (watch for the init relay explosion on this longer-relay env; the
`--relay-grad-clip` flag exists if training is unstable — ORIENT 2026-06-29).

## Prereqs
- `latent-cache-for-dynamics-training` (mem2mem trainer consumes the cache).
- `memmaze-dynamics-prep` (actions npy + latent cache on cluster).

## Open config decisions (ask Merlin with throughput data before submitting)
- Same model-size question as vanilla (keep the two arms' transformer config IDENTICAL for a fair
  comparison; only memory/FF9/training-signal differ).
- mem2mem specifics: `--mem2mem-frac` (GridWorld winner: 1.0 rollout-only; 50/50 also worked),
  `--clip-len` (rollout length; GridWorld used 64 = 4x window), n_ctx choices, `--ff9 K --n-memory M`.
- Compute budget: rollout training is ~sequential over the clip -> slower per sample than vanilla.

## Done means
Trained checkpoint `checkpoints/memmaze/dynamics_mem2mem.pt` pulled + verified, W&B healthy (watch
relay stability), qualitative rollout sheet vs vanilla, provenance in
`experiments/memmaze-dynamics/NOTES.md` + EXPERIMENTS.md line. Memory CLAIMS wait for the memmaze
recall/probe eval (follow-up task) — sheets illustrate, never decide.
