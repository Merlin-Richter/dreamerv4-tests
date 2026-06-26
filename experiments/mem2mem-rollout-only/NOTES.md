# mem2mem rollout-only — does the sliding mem→mem rollout ALONE match the 50/50 retention win?

Task: `tasks/in-progress/retrain-mem2mem-with-only-rollout-training.md`.

## Question
The headline mem2mem win (`experiments/mem2mem/`, job 410376) trained on a **50/50** per-batch mix of
(a) the normal shortcut-forcing loss on a random ≤N window and (b) the mem→mem sliding rollout. Does
training on the **rollout signal ALONE** (`--mem2mem-frac 1.0`, no normal-window batches) reproduce the
same flat-to-high-k position retention, or does the normal-window loss matter?

## Setup (provenance)
- Branch `exp/mem2mem-rollout-only`, SHA `2951ee688c689d4e32466310bdb9c201e5bcdb0e`.
- Cluster ferranti (H100), **job 411133**, 1 GPU, 6h budget.
- Trainer: `experiments/mem2mem/train_mem2mem.py` (UNCHANGED code — rollout-only is just a flag).
- Command (identical to the 50/50 job 410376 except `--mem2mem-frac 1.0` and the checkpoint name):
  ```
  python -u experiments/mem2mem/train_mem2mem.py \
    --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
    --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout.pt \
    --epochs 50 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 \
    --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-only
  ```
- Data: the 5× `data/gridworld.npy` (5000 eps, datagen job 410371) + frozen `tokenizer.pt` already on
  cluster (both gitignored, untouched by `sync_code`). Same inputs as job 410376.
- Output checkpoint: `checkpoints/gridworld/dynamics_mem2mem_rollout.pt` (distinct, does not clobber
  `dynamics_mem2mem.pt`).

## Eval plan (per task)
Recall test at **window=8, max_k=64** via the new `recall.py --window 8 --max-k 64` CLI → JSON, then
overlay vs vanilla / FF9 / mem2mem(50/50) with `plot_recall.py`. Check whether near-perfect position
retention persists to high k under the tight window. Compare tail (k≥14) position_acc against the 50/50
model (which held ~0.96 flat to k=20).

## Status
- [2026-06-26] submitted job 411133.
