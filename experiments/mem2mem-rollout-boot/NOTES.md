# mem2mem rollout-only + bootstrap (step-size curriculum) — does the full shortcut ladder help?

Follow-up to `experiments/mem2mem-rollout-only/` (which trained finest-step flow only). Adds the shortcut
**bootstrap distillation** (coarser d/2 steps) to the rollout new-half loss, under a **finest-first
step-size curriculum** so we never distil a coarse step from an untrained finer step.

## What's new vs the rollout-only winner
- `_newhalf_loss` now = full shortcut-forcing diffusion (flow at d_min + bootstrap at coarser d),
  mirroring `DynamicsModel.loss` line-for-line (audited: EXPERIMENTS `V-newhalf-loss`, |diff|=0.0).
  Applied to BOTH clean-context and full-noise (memory-only) modes.
- **Curriculum** (Merlin's idea — early bootstrap from untrained finer steps is wasted compute): train
  d_min only for the first 15%, then unlock one coarser step every 2.5%, finest-first. n_d=8 → fully
  unlocked at training fraction 0.325. Bootstrap forwards are SKIPPED while only d_min is active (whole
  warmup is cheap), keeping wall-clock near the prior ~3h.

## Setup (provenance)
- Branch `exp/mem2mem-rollout-only`, SHA: see EXPERIMENTS / job record. Cluster ferranti (H100).
- Trainer: `experiments/mem2mem/train_mem2mem.py` (bootstrap default ON; curriculum default ON).
- Command (rollout-only, 36 epochs to hold ~3h; bootstrap adds ~1.4× per active step):
  ```
  python -u experiments/mem2mem/train_mem2mem.py \
    --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
    --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout_boot.pt \
    --epochs 36 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 \
    --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-boot
  ```
- Data: 5× `data/gridworld.npy` + frozen `tokenizer.pt` already on cluster (same as job 411133).
- Output ckpt: `checkpoints/gridworld/dynamics_mem2mem_rollout_boot.pt` (distinct).

## Eval plan
- Recall @ window=8, max_k=64 → compare against rollout-only (no-boot) `dynamics_mem2mem_rollout.pt`.
  Retention is already near-ceiling, so the more telling test is **few-step inference**: recall at K=2
  (and K=1) — the bootstrap ladder should help the model take fewer/coarser steps without quality loss.

## Status
- (to fill: SHA, job id, result)
