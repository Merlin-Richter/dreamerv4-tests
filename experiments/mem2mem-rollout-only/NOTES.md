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

## Result (2026-06-27) — WIN, rollout-only reproduces (slightly beats) the 50/50 split
Job 411133 COMPLETED (exit 0, 2h51m, 50 epochs). Log confirms `train normal: 0.00000` every epoch →
the normal-window branch never fired, so this is genuinely rollout-ONLY. val(normal) ended ~0.005
(higher than the 50/50 run's 0.0027 — expected, it never trained the normal-window loss; that loss is
the in-window monitor, not the result).

Recall @ window=8, max_k=64 (`recall_dynamics_mem2mem_rollout.json`), position_acc mean over k:

| model                | mean (all k) | tail k≥14 | k=64  |
|----------------------|-------------:|----------:|------:|
| vanilla              |        0.040 |     0.033 | 0.000 |
| FF9                  |        0.375 |     0.111 | 0.125 |
| mem2mem 50/50        |        0.988 |     0.984 | 0.984 |
| **mem2mem rollout-only** |    **0.992** | **0.988** | **1.000** |

Rollout-only holds position recall FLAT ~0.98–1.0 to k=64 under the tight window=8 — at least as good as
the 50/50 model, uniformly high (not periodicity-gamed; copy_last only spikes at the bounce-period k=10/40).
**Conclusion: the mem→mem sliding rollout signal ALONE is sufficient; the normal shortcut-forcing loss is
not needed for the retention win.** 4-way overlay: `compare_w8_k64_4way.png`.

Caveat: in-window next-frame reconstruction (val normal 0.005 vs 0.0027) is slightly worse without the
normal-window loss — small-k position recall is still 1.0, so tracking-in-the-clear is unaffected, but a
50/50 (or small normal-loss fraction) may be the safer default if pixel fidelity matters elsewhere.

## Status
- [2026-06-26] submitted job 411133.
- [2026-06-27] COMPLETED; checkpoint pulled via new `scripts/pull_file.sh`; recall done → WIN (above).
