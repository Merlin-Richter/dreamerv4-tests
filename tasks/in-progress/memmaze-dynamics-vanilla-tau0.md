# Retrain memmaze vanilla dynamics with the τ0-anchor (honest baseline)

Requested by Merlin 2026-07-04, after the GridWorld honest-baseline A/B
(`experiments/vanilla-honest-baseline/`: Arm D τ0-anchor fixed the vanilla in-window failure —
teacher-forced ~1.0 vs 0.09; the old objective starves prediction-from-context).

## Goal
The memmaze no-memory baseline retrained with `DynamicsModelTau0Anchor` (per-frame p=0.5 forced
(τ_idx=0, d=d_min, GT flow); `--model-module`, no spec change): the old vanilla arm 415103 trained
the starved objective, so the upcoming 3-way memory comparison would have an unfairly weak
baseline. P_ANCHOR=0.5 kept — matches the τ0-GT dose of the mem2mem arms' noise mode (50% per
slide), so the training-pressure comparison stays apples-to-apples. Known nuance: on memmaze some
anchored frames are epistemically ambiguous (unseen areas) → x-pred trains toward the conditional
mean there; 1-step-ahead with W=32 context is mostly near-deterministic and the other 50% of
frames keeps the full shortcut ladder trained.

## Config
Identical to 415103 (`train_vanilla.sh 50 64`: 512/12/16, W=32, bs64, lr 3e-4, seed 0, latent
cache) with overrides: `--checkpoint checkpoints/memmaze/dynamics_vanilla_tau0.pt` (no clobber),
`--model-module experiments/vanilla-honest-baseline/model_arms.py:DynamicsModelTau0Anchor`,
W&B `memmaze-dyn-vanilla-tau0`. ~8.5h on H100, `--hours 12`.

## Done means
Checkpoint pulled + load-verified, W&B healthy, pre64 sheets rendered + compared vs
`sheets_vanilla/` (expect: better GT-tracking in the early rollout / within-window), provenance
recorded. Enters the memmaze comparison as THE vanilla baseline (3-way becomes 4-way or replaces
415103 — Merlin's call at eval time).

## Provenance
- (filled at submit)
