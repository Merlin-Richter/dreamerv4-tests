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

## Result (2026-06-27) — NEGATIVE: the bootstrap HURT retention; not needed for few-step either.
Job 411221 COMPLETED (2h21m, 36ep, clean/stable: val 0.0047, flow 0.0014). Curriculum ramped 1→8 by
ep11 as designed; `train normal: 0.00000` (rollout-only). BUT recall collapsed vs the plain no-boot
rollout-only winner. position_acc mean (window=8, max_k=64):

| K | no-boot rollout-only (50ep) | boot+curriculum (36ep) |
|---|----------------------------:|-----------------------:|
| 4 | 0.992 | 0.472 |
| 2 | 1.000 | 0.516 |
| 1 | 0.999 | 0.502 |

Two findings:
1. **The plain no-boot model already does single-step K=1 inference near-perfectly (0.999).** The
   bootstrap's whole motivation (few-step inference) does not apply on this task — there was nothing to fix.
2. **Bootstrap+curriculum HALVED retention at every K.** Mechanism: final FF9 train loss is **0.054 vs
   the no-boot run's 0.010** (5× worse memory sufficiency) — recall tracks FF9, not flow. The bootstrap
   drove flow LOWER (0.0014 vs 0.0022) at the cost of memory. Plausible cause: FF9 is normalized by
   `flow.detach()/ff9.detach()`, so lower flow shrinks FF9's effective weight → the bootstrap inadvertently
   down-weighted the memory objective. The model spent capacity on denoising quality, not memory.

Caveats (confounds): boot ran 36 epochs vs no-boot's 50; and the flow↔ff9 normalization interaction above.
A confound-free check would be boot @ 50ep — but since no-boot is already at ceiling incl. K=1, there is
no upside to pursue. CONCLUSION: keep the simple rollout-only + FF9 (finest-step flow, no bootstrap).
Overlay: `compare_w8_k64_ablations.png`. Per-K JSONs: outputs/recall/recall_{boot,noboot}_K{1,2}.json.

## Status
- [2026-06-27] DONE. branch SHA `3be2108`, ferranti job 411221 (36ep, 2h21m). bootstrap+curriculum ON.
  Result above. Checkpoint `dynamics_mem2mem_rollout_boot.pt` (kept for reference; NOT the winner).
