# Memory Maze 9x9 dynamics — experiment notes

Campaign kickoff 2026-07-03 (Merlin). Two arms on the frozen memmaze tokenizer's latents
(`checkpoints/memmaze/tokenizer.pt`, 412635, n_latents=32 bottleneck_dim=16):
1. **vanilla** Dreamer-4 dynamics (n_memory=0) — baseline (task `memmaze-dynamics-vanilla`).
2. **mem2mem ROLLOUT-ONLY** [LOCKED by Merlin]: `--mem2mem-frac 1.0 --no-bootstrap` + FF9
   (the GridWorld 411133 winner config) (task `memmaze-dynamics-mem2mem`).
Trained via the NEW latent disk cache (tokenizer never on the GPU during training; see
`tasks/done/latent-cache-for-dynamics-training.md`).

## Window-invariance probe (GridWorld reference numbers, local 4070, fp32)
`probe_window_invariance.py --frames data/gridworld.npy --offset 8`: latent cos-sim mean **0.9975**
(min 0.977), rel-L2 5.8%; decoded recon window-delta MSE **1.33e-6** vs recon-vs-GT ~8.2e-6 (6x below
the recon error). ⇒ arbitrary-offset slicing of cached latents is safe (Merlin's claim confirmed on
GridWorld; memmaze probe in the prep job). JSON: `probe_window_invariance_gridworld.json`.

## Provenance
- Prep job (extract actions/labels from `data/memmaze9x9_raw` + build latent cache + memmaze
  invariance probe): ferranti **job 415098** @ SHA `7d86b8d` (`prep.sh`, --hours 2 --cpus 8).
  Outputs: `data/memmaze9x9_actions.npy` (+ per-key label npys), `data/memmaze9x9.latents-<sha12>.npy`
  (~3GB fp16), probe JSON.
- Training jobs: <fill>
