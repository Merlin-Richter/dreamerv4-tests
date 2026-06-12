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
| EXP-009 | H2 | D-011 | master @ f1cf860 (frozen probe) | local (4070) | running | (pending) | H2 baseline: revisit-consistency sweep of `my_dynamics.pt` on the frozen probe (N=8, P=3, 64 eps/n_occ); color-recall headline, drift-controlled |
