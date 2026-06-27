# Fair bootstrap A/B — does the shortcut bootstrap ladder ACTUALLY hurt retention?

Re-run of `experiments/mem2mem-rollout-boot/` (job 411221, NEGATIVE) with its confounds removed. That
run conflated the bootstrap with three other changes vs the rollout-only winner; this isolates the
bootstrap gradient. See `tasks/in-progress/fair-bootstrap-ablation.md` for the full rationale.

## The confounds we remove
1. **FF9 normalizer dilution** → fixed with `--ff9-norm-flow`: normalize the FF9 term by the pure d_min
   FLOW magnitude (`flow_norm`), not the mixed flow+bootstrap diffusion mean. The mixed mean is dragged
   down by the smaller bootstrap self-distillation term, which silently down-weights memory. (Local
   smoke: `flow 0.113` mixed vs `flow_norm 0.266` pure ⇒ the old code under-weighted FF9 ~2.4×.)
2. **τ distribution** → held IDENTICAL across both arms by using the curriculum d-sampling in BOTH (the
   d-snapped grid that bootstrap requires). It cannot be matched to the *winner* (which used uniform τ),
   because bootstrap's two-half-step target is only well-defined on the d-grid — so the τ-shift is
   intrinsic to bootstrap and is measured separately (Arm A vs winner) rather than removed.
3. **Epochs** → both arms 50ep (matches the winner; the old boot ran only 36).

## Design — 2-arm factorial, 50ep, ferranti H100
- **Arm A (control):** `--boot-loss-off --ff9-norm-flow`. Snapped-τ + curriculum, bootstrap LOSS off
  (coarse-d tokens get flow MSE). Everything the boot run has EXCEPT the bootstrap gradient.
- **Arm B (fair boot):** `--ff9-norm-flow` (bootstrap + curriculum ON by default).
- **A vs B** → pure bootstrap-gradient effect (τ, FF9-weight, epochs all identical).
- **A vs winner** (`dynamics_mem2mem_rollout.pt`, uniform-τ pure-flow 50ep) → the τ-shift effect.

## Commands
```
# Arm B (fair boot)
python -u experiments/mem2mem/train_mem2mem.py \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout_boot_fair.pt \
  --epochs 50 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 \
  --ff9-norm-flow \
  --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-boot-fair

# Arm A (control)
python -u experiments/mem2mem/train_mem2mem.py \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout_bootctrl.pt \
  --epochs 50 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 \
  --boot-loss-off --ff9-norm-flow \
  --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-bootctrl
```

## Eval plan
Recall @ window=8, max_k=64 (+ K=2/1), overlay 4-way: Arm A, Arm B, winner (no-boot rollout-only),
vanilla/copy_last baselines. position_acc mean / tail (k≥14) / k=64.

## Predictions (pre-registered)
- If **B ≈ A ≈ winner**: the bootstrap is free; the old negative was purely the confounds (normalizer +
  epochs). Intuition vindicated — shortcut forcing does not hurt retention.
- If **B < A**: the bootstrap gradient itself hurts memory on this task (would be surprising).
- If **A < winner**: the τ-shift (snapped grid, ~25% τ=0) is the culprit, independent of bootstrap.

## Provenance
- Branch `exp/mem2mem-rollout-only`, SHA `851a7ab8d8f9d88827e3d9588c4ecd4d0ed742d1`. Cluster ferranti (H100).
- **Arm B (fair boot):** job **411502** → `dynamics_mem2mem_rollout_boot_fair.pt` (--ff9-norm-flow, 50ep, 5h).
- **Arm A (control):** job **411503** → `dynamics_mem2mem_rollout_bootctrl.pt` (--boot-loss-off --ff9-norm-flow, 50ep, 4h).
- Data: 5× `data/gridworld.npy` + frozen `tokenizer.pt` already on cluster (same as jobs 411133/411221).
- Compare against: winner `dynamics_mem2mem_rollout.pt` (job 411133) + old unfair boot (411221).

## Status
- [2026-06-27] SUBMITTED. Code verified (probe |diff|=0, autograd relay intact, both arms smoke clean @ 4070).
  Jobs 411502 (B) + 411503 (A) queued on ferranti (queue depth ~241 pending). Awaiting completion → recall A/B.
