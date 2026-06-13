# EXPERIMENTS.md — index

> One line per experiment; details in `experiments/EXP-NNN/`. Append-only.
> EXP-001..007 backfilled 2026-06-11. Provenance pre-W&B (before 2026-06-09) is
> approximate and marked as such; from EXP-006 on, provenance is exact (W&B).
> W&B entity: `models-eberhard-karls-universit-t-t-bingen`.

| ID | Hyp | Decision | Provenance | Job/Host | Status | Headline result | Read |
|---|---|---|---|---|---|---|---|
| EXP-001 | H1 | D-001 | pre-git, approximate | local | done | qualitative: coherent char-level Shakespeare | infra validated |
| EXP-002 | H1 | D-003 | pre-git, approximate | local | done | qualitative: single-frame recon OK | B baseline works |
| EXP-003 | H1 | D-003 | ~60a4b67 | local | done | **latent collapse**: all-black recon, latent cos-sim ≈ 1 | negative; spawned D-004 |
| EXP-004 | H1 | D-004 | 9019835..7cb30c1, approximate | local | done | collapse resolved on occluded.npy; latent_cos ↓ 0.19→0.03 across iterations | dense backgrounds fix the optimum |
| EXP-005 | H1 | D-005 | pre-W&B, approximate | local | done | qualitative: plausible short rollouts on bouncing | unconditional dynamics baseline |
| EXP-006 | H1 | D-007 | 58ebfde (A) / 3205e8e (B); W&B 1lzegsxt, rc01geau | galvani-cn109 / mlcbm014 | done | val/mse 3.23e-4 (no LPIPS) vs **1.41e-4 (VGG LPIPS)** | LPIPS adopted; A/B confounded but direction clear |
| EXP-007 | H1, H2 | D-008 | 3205e8e; W&B sm0kr1cf | mlcbm002 | done | val/loss 1.93e-3 but rollouts randomize ball color+position; bg preserved | **Surprise: high** → ESC-001, D-009; cause found in EXP-008 (inference bug, not broken model) |
| EXP-008 | H1 | D-010 | master @ 8cb4c78 | local (4070) | done | gen-MSE 0.0289(τ_ctx0.1)→0.0165(0.99); ball color preserved at high τ_ctx | supports D-010: EXP-007 rollout failure is an inference bug (context_noise=0.1=90% noise on context); 1-line fix, no retrain → ESC-002 |
| EXP-009 | H2 | D-011 | master @ f1cf860 (frozen probe) | local (4070) | done | color ΔRGB cliff at n_occ=7: 16.8→94.4 (ceiling 15.9, chance 109.9); latent-MSE↔color r=0.952; detector gate pass | **supports H2 baseline**: recall collapses to chance exactly when prefix scrolls out of N=8 window (geometry-predicted). Position drift-confounded. Instrument calibrated for H3. Present-then-stop → ESC-004 |
| EXP-010 | H3 | D-014 | master @ ec45dc1; probe 5503e75; W&B 82klng1c (k1) / 17u810q2 (k3) | local (4070) | done | FF7 v1 screening k=1/k=3 (seed 0, 100 ep). color ΔRGB @ n_occ 12/16/24: k1 52/59/80, k3 40/55/65 vs baseline ~chance (108/100/120); ceiling/drift ≤ baseline | **supports H3 (color-only)**: post-window cliff replaced by gentle decay; clears T-004 bar (<63) at n_occ 12&16 both arms, misses 24; k3>k1 (relay holds); position at chance (not retained) → latent-MSE confounded. No tripwires. Present-then-stop → ESC-006 |
| EXP-011 | H3 | D-015 | master @ 9050a80; probe env/detector 5503e75 | local (CPU, no train) | done | position-deficit diagnostic. latent→xy probe R²=0.96 (C encodes pos); 1-step teacher-forced pos_err: my_dynamics 4.5px (>copy-last 3.2) vs ff7_k1/k3 ~1.0px; ff7_k3 open-loop tracks to 14.8px@h12 then →chance | **reframes the position worry**: deficit is in D not C; my_dynamics ≈(a) weak motion model, FF7 ≈(b) open-loop chaos (tracks ~12 steps). Occluded pos-at-chance = dead-reckoning chaos, not memory defect. CONFOUND: FF7-better-dynamics vs my_dynamics-undertrained unidentified → baselines not training-matched. High surprise → ESC-007 |
