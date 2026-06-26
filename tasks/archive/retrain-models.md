# Train new models for tokenizer and dynamics on Gridworld

Due to some changes in the tokenizer and dynamics code, we need to train new models from scratch.
Use the cluster to train these new models.
One tokenizer model and two dynamics models (vanilla and FF9 enabled)

Train the tokenizer with foreground weighing on.
Put the models into checkpoints/gridworld/

Get the results and models back with pull results

---

## RESULT (2026-06-26) — DONE. 3 models trained on ferranti (H100), pulled to checkpoints/gridworld/.

Code SHA `0a0e070` (spec-synced: depth-9 every-3rd cadence, dynamics logit_scale, tokenizer sigmoid).
EXPERIMENTS.md: `R-gridworld-retrain`. Pipeline (all ferranti, W&B project dreamerv4-gridworld):
- **datagen** (job 410366): fresh `data/gridworld.npy` (1000,200,64,64,3), occ 0.50. Regenerated rather
  than reuse the stale local copy (which was a different 7.3GB shape).
- **tokenizer** `tokenizer.pt` (job 410367): `--fg-weight 2.0` (foreground weighting ON), bs128/30ep.
  Validated: val fg_mse **1.7e-5** (ball faithfully reconstructed, not dropped), latent_cos 0.114
  (escaped collapse), pred_std 0.367 (real content), 0 grad-skips. 15.6M params, depth 9.
- **dynamics vanilla** `dynamics_vanilla.pt` (job 410368): bs256/50ep, val/loss 0.0058. n_memory=0,
  n_actions=2 (action-conditioned, auto from _actions.npy). 7.75M params.
- **dynamics FF9** `dynamics_ff9.pt` (job 410369): `--ff9 3 --n-memory 4`, bs128/50ep, val/loss_diffusion
  0.0015, train/loss_ff9 0.117 (sufficiency term active). n_memory=4, ff9_k=3, n_actions=2. 7.75M.

All three pulled to `checkpoints/gridworld/` (staged via a cp job into runs/ since pull_results only
syncs runs/<run>/), load clean, configs mutually consistent (bottleneck 64 / n_latents 4 match tokenizer).

NEXT (separate, ORIENT NEXT#2): recall A/B (FF9 memory vs vanilla). Models are ready; **gated on the
recall k↔tick alignment sign-off** (flagged atop recall.py) before the numbers are trusted.