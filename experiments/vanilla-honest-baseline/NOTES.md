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
Rationale: transplant the AMOUNT of (pure-noise-input → GT-target) pressure that provably teaches
the map in the mem2mem trainer into plain diffusion forcing, minus memory tokens, minus rollout.
One knob (anchored slice ≈ 18% of ramp-weighted loss, ramp kept at w(0)=0.1 like mem2mem).
**PRECISION (2026-07-04, Merlin's question):** the mem2mem noise mode is NOT per-frame — it is
per-sequence-per-slide (`_sample_modes`: rand(B)<0.5) and hides the WHOLE window (old half AND new
half latents at τ=0), so the carried memory tokens are the only scene carrier; in the winner
no-bootstrap config every noise-mode new-half frame gets the GT flow target. Arm D is per-frame
i.i.d. BY DESIGN, not as an approximation: a memoryless model with a fully-hidden window would
face an unconditionally unpredictable target (optimum = marginal mean frame — mode-averaging, no
dynamics signal); per-frame anchoring keeps ~half the context readable so the demanded pathway is
direct latent temporal attention (+ registers as scratch channels — the pathway Probe 2 showed is
learnable). So mem2mem validates the pressure MAGNITUDE through the memory pathway; whether the
same dose through the latent-attention pathway suffices is precisely what this arm tests.
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

- SHA `fae4e8bae669ce291fb338b71190b907739d884f` (branch `exp/mem2mem-rollout-only`), synced to
  ferranti 2026-07-04 16:45.
- **Arm C (dcurr): job 415190** → `checkpoints/gridworld/dynamics_vanilla_dcurr.pt`, W&B
  `gw-dyn-vanilla-dcurr` (logs `runs/gw-vanilla-dcurr/slurm-415190.out`).
- **Arm D (tau0): job 415191** → `checkpoints/gridworld/dynamics_vanilla_tau0.pt`, W&B
  `gw-dyn-vanilla-tau0` (logs `runs/gw-vanilla-tau0/slurm-415191.out`).

## RESULT (2026-07-04) — both pre-registered predictions CONFIRMED; Arm D is the honest baseline

Both jobs COMPLETED rc=0 (~20/30 min). Final val/loss (default sampler, comparable): old vanilla
0.0016 / **Arm C 0.0019 / Arm D 0.0010** — Arm D's anchor did not hurt (helped) the standard loss.

**Primary — teacher-forced 1-step pos_acc (`results_probe.json`, same seeds as the diagnosis):**

| ckpt | t=2 | t=4 | t=8 | t=15 | free-run j=1..12 |
|---|---|---|---|---|---|
| vanilla (old) | 0.078 | 0.094 | 0.078 | 0.094 | 0.17 → 0.05 |
| **Arm C dcurr** | 0.078 | 0.062 | 0.250 | 0.188 | 0.05–0.22 |
| **Arm D tau0** | 0.844 | 0.984 | 1.000 | 1.000 | **0.98–1.00 flat** |

**Secondary — recall w8 max_k32 position_acc:** Arm D = 1.00/1.00/0.95 at k=2/4/6 (in-window),
collapse to ~chance at k≥8 — the EXACT eviction boundary. Textbook honest baseline: competent
in-window, zero retention past the window (no memory tokens — as designed). Arm C ≈ chance at all
k. Sheets (`sheets_tau0/`): free-run tracks GT through all 12 steps, crisp + right colors — the
`sheet_normal.png` failure Merlin flagged is gone.

**Verdicts:** (1) Arm D ≥0.9 prediction CONFIRMED — sustained per-frame (τ=0, d_min, GT-flow)
pressure through the LATENT-ATTENTION pathway alone teaches the transition map, no memory tokens
needed for in-window competence. (2) Arm C "partial at best" CONFIRMED — the transient curriculum
(even Merlin's stronger 33/66 schedule) does not fix it; sustained pressure is the active
ingredient, validating the diagnosis's allocation story by intervention. (3) The memory-vs-vanilla
comparison should henceforth use Arm D-style vanilla as baseline; the memory arms' remaining edge
is then cleanly PAST-window retention (mem2mem w8: ~1.0 at k=32/64 vs Arm D 0.03-0.08 at k≥8).

**Follow-ups (Merlin decides):** graduate the τ0-anchor into src/+spec (small `sample_tau_d`
change / config knob)? Retrain the memmaze vanilla arm (415103) with it for an honest 3-way?

## K-sweep addendum (2026-07-04, Merlin's ask) — `probe_K.py` / `results_K_sweep.json`

Arm D pos_acc = **1.000 at EVERY K ∈ {1, 2, 4, 128}** (teacher-forced t=4/8 + free-run j1–4):
even a single x-pred step from pure noise is perfect — expected for x-prediction on a
deterministic env once the map exists; the d_min-anchored training generalizes across the whole
d ladder (consistent with 411133: d_min-only training → perfect K=4/2/1). Old vanilla stays
broken at every K incl. K=128 (0.05–0.13, d_true ~2.3) → finer inference sampling cannot rescue
the missing map; the deficit is in the weights, not the sampler.

## Status

- [2026-07-04] designed, smoked locally, submitted 415190 (C) + 415191 (D) on ferranti @ fae4e8b.
- [2026-07-04] both COMPLETED; ckpts pulled; probe + recall w8 + sheets done → predictions
  confirmed, Arm D = honest baseline. Task → done.
- [2026-07-04] K-sweep addendum (above).
