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

## Prep results (job 415098 @ 7d86b8d, rc=0, ~8 min)
- **npz keys:** action(1001,6 one-hot), agent_dir(1001,2), agent_pos(1001,2), image,
  maze_layout(1001,9,9), reset, reward, target_color(1001,3), target_pos, target_vec,
  targets_pos(1001,3,2), targets_vec — ALL extracted to `data/memmaze9x9_<key>.npy` on /weka.
  **actions: (2900,1001) int64, n_actions=6** (`memmaze9x9_actions.npy`, auto-detected by trainers;
  pulled local along with agent_pos/agent_dir). maze_layout & targets stay on /weka for the eval task.
- **Latent cache built:** `data/memmaze9x9.latents-fe2ff8440036.npy` (2900,1001,32,16) fp16, 452s
  (6.4 eps/s, cache-batch 16).
- **Memmaze window-invariance probe (offset 32 vs 0, 8 eps):** latent cos **0.999648** (min 0.9978),
  rel-L2 2.5%; recon MSE window-delta **1.19e-6** vs recon-vs-GT 7.3e-5 (**60x below the recon
  error** — even stronger than GridWorld). Arbitrary-offset slicing confirmed safe on memmaze.
  JSON: `probe_window_invariance_memmaze9x9.json` (pulled local).

## Calibration (bs/throughput, H100, synthetic latents)
- Round 1 (job 415100 @ 37330e6): **vanilla** 41.0M @ 512/12/16 W=32: bs64 **0.455 s/step,
  140.6 clips/s, 42.6GB** (bs128 OOM) -> use bs64. **mem2mem OOM @ bs8/clip128**: the rollout keeps
  every slide's graph until one backward (~7 slides + FF9 forwards ≈ bs×2×clip activation footprint).
- Round 2 (job 415101 @ 965268b): fine mem2mem ladder bs {1,2,4,6}xclip128, {4,6,8}x96, {6,8,12}x64
  -> <fill>

## Provenance
- Prep: ferranti **415098** @ `7d86b8d`. Calibration: **415100** @ `37330e6`, **415101** @ `965268b`.
- Training jobs: <fill>
