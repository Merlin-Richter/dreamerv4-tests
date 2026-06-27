# mem2mem rollout-only, NO FF9 — is the explicit sufficiency loss even needed?

Parallel ablation against `experiments/mem2mem-rollout-boot/` (job 411221). Drops the FF9 sufficiency
term entirely: memory is trained ONLY by the rollout flow loss.

## Hypothesis
Memory-carrying pressure in the rollout already comes from TWO places: (a) the explicit FF9 term, and
(b) the 50% **full-noise** mode, where every latent in the window is pure noise so the new-half flow loss
can only be satisfied by reconstructing the scene from the carried memory. This run removes (a) and keeps
(b): if the noise-mode flow loss alone trains memory to carry hidden state, FF9 is redundant for this task.
Verified offline: with `use_ff9=False`, force_mode="noise", the relay gradient still reaches an evicted
frame's memory construction (|grad|≈9e-4, exactly 0 when the relay is detached) — so memory IS still trained.

## Setup (provenance)
- Branch `exp/mem2mem-rollout-only`, SHA: see EXPERIMENTS / job record. Cluster ferranti (H100), parallel
  with job 411221.
- IDENTICAL to 411221 except `--no-ff9` (rollout-only, bootstrap ON, curriculum ON). Config still carries
  ff9_k=3 / n_memory=4 (model architecture & checkpoint shape unchanged) — only the loss term is dropped.
- Command:
  ```
  python -u experiments/mem2mem/train_mem2mem.py \
    --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
    --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout_noff9.pt \
    --epochs 36 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 --no-ff9 \
    --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-noff9
  ```
- Dropping FF9 removes one forward/slide, so this run is FASTER than 411221 (< ~3h).

## Eval plan
Recall @ window=8, max_k=64 (+ few-step K=2/1), 4-way vs: rollout-boot (411221, flow+ff9),
rollout-only no-boot (`dynamics_mem2mem_rollout.pt`), and the baselines. Question: does retention survive
without FF9?

## Status
- (to fill: SHA, job id, result)
