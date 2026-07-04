# Vanilla honest-baseline A/B — can a τ/d-allocation change alone teach the transition map?

**Date:** 2026-07-04. **Ordered by Merlin** (follow-up to `experiments/vanilla-inwindow-diagnosis/`:
vanilla learns denoising, never dynamics, because (τ≈0, GT-flow) is ~0.4% of ramp-weighted loss).
Both arms are **vanilla** (n_memory=0), only `sample_tau_d` changes (train mode only; val keeps the
default distribution so `val/loss` is comparable across arms and to the original vanilla 410374).
Implementation: `model_arms.py` via the `--model-module` seam — no spec-backed file touched.

## Arms

**Arm C — `DynamicsModelDCurriculum` (Merlin's design):** finest-first step-size curriculum on
training progress: d_min only for the first 33% of epochs, remaining steps unlocked gradually and
evenly until 66%, everything unlocked after. Progress tracked via eval→train transitions of
`loss()` (one val pass per epoch); total epochs from `$CURR_TOTAL_EPOCHS` (default 50 — matches
`--epochs 50`; MUST be set if epochs change). During warmup every frame gets a GT flow target
(vs 1/8 default), lifting the (GT ∧ τ≤0.1) weighted share ~0.4% → ~2.7% — but transiently, and τ
stays uniform with the ramp applied.

**Arm D — `DynamicsModelTau0Anchor` (agent's design):** with prob **P_ANCHOR=0.5** per frame,
force `(tau_idx=0, d_idx=d_min)` — the frame's own latent is pure noise and the loss is the
ground-truth flow term = plain next-frame-prediction-from-context, sustained the whole run.
Rationale: transplant exactly the pressure that provably teaches the map in the mem2mem trainer
(noise mode: 50% of new-half frames at τ=0 vs GT, ramp applied — recall ~1.0 even without FF9,
see mem2mem-rollout-noff9-fair) into plain diffusion forcing, minus memory tokens, minus rollout.
One knob, magnitude matched to a validated recipe (anchored slice ≈ 18% of ramp-weighted loss).
Loss formula / ramp / bootstrap untouched. NOTE (design record): coarse-d τ=0 GT anchoring was
considered and rejected — x-pred GT at coarse d is only valid for deterministic envs, and the
411133-era result (rollout-only, NO bootstrap, K=4 recall 0.992) shows d_min-anchoring alone
transfers to K=4 inference on GridWorld.

## Pre-registered predictions (before either job runs)

- **Arm D**: teacher-forced 1-step (probe_next_pos.py) pos_acc ≥ 0.9 at ctx ≥ 4 (vanilla today:
  0.08–0.09); free-run tracks GT in-window; recall decays toward copy_last once context leaves the
  window (EXPECTED and desired — that's the honest no-memory baseline). val/loss same ballpark as
  0.0016.
- **Arm C**: partial improvement at best (transient pressure, ramp still ×0.1 on low τ): above
  vanilla's 0.09 but below 0.9 on the teacher-forced probe. If C ≈ D ≈ 1.0 instead, the
  "sustained pressure needed" part of the diagnosis is falsified and the cheaper curriculum
  suffices (also a useful result).
- If BOTH fail → the allocation story is incomplete; next knob is the ramp itself (exempt the
  anchored slice) — needs a new arm.

## Setup / provenance

- Branch `exp/mem2mem-rollout-only`, SHA: see LAUNCH below. Cluster ferranti (H100), 1 GPU each,
  `--hours 4`. Data: cluster `data/gridworld.npy` (5×, 5000 eps, datagen 410371) + frozen
  `checkpoints/gridworld/tokenizer.pt` (both already on cluster, gitignored). Trainer:
  `src/training/train_dynamics.py` (latent-cache path; the two jobs may race-build the cache —
  atomic rename, benign). Config mirrors the r2 vanilla (410374): 50ep, bs256, lr 3e-4 default,
  W=16 default, actions auto (n_actions=2), seed 0.
- Local smokes (4070, 2026-07-04): both arms 2–3 epochs green; curriculum epoch counter verified
  ticking (k=1 → k=8 with CURR_TOTAL_EPOCHS=3); sampler unit checks pass (anchored mass 0.499
  train / 0.0006 eval; curriculum monotone, d_min-only warmup, τ snapped to d-grid).

Commands (identical up to model class / ckpt / name):

```
python -u src/training/train_dynamics.py \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics_vanilla_dcurr.pt \
  --epochs 50 --batch-size 256 --seed 0 \
  --model-module experiments/vanilla-honest-baseline/model_arms.py:DynamicsModelDCurriculum \
  --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-vanilla-dcurr
# Arm D: ...model_arms.py:DynamicsModelTau0Anchor --checkpoint .../dynamics_vanilla_tau0.pt \
#   --wandb-name gw-dyn-vanilla-tau0
```

## Eval plan (when both land)

1. Pull both ckpts (`pull_file.sh`); they load with the BASE DynamicsModel (subclasses add no
   params) → all tooling unchanged.
2. **Primary:** `experiments/vanilla-inwindow-diagnosis/probe_next_pos.py` extended with both
   ckpts — teacher-forced 1-step + free-run vs the old vanilla + memory arms.
3. Secondary: recall (native window + w8 max_k 32) vs vanilla/mem2mem; normal + occlusion sheets.
4. Verdict against the pre-registered predictions above; EXPERIMENTS.md line.

## LAUNCH (filled at submit time)

- (pending)

## Status

- [2026-07-04] designed, smoked locally, submitting.
