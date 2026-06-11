# EXP-008 — Context-noise rollout diagnostic

Decision: D-010 (sharpens D-009 cand. (a)) | Hypothesis: H1 (dynamics) | 2026-06-11, local

## Provenance

Local, RTX 4070, inference-only (no training). Deterministic (`--seed 0`).
- Script: `experiments/EXP-008/diagnose_context_noise.py` (committed; see index SHA).
- Dynamics checkpoint: `my_dynamics.pt` (EXP-007 / W&B sm0kr1cf, action-conditioned,
  n_actions=2). **Not modified** — only `model.config.context_noise` mutated per rollout.
- Tokenizer (frozen): `trained_autoencoder.pt` (EXP-006 arm B / W&B rc01geau).
- Data: `occluded.npy` + `occluded_actions.npy`. 6 fixed (seeded) episodes,
  4 GT context frames, 12 generated, K=4 (checkpoint default).
- Command: `python experiments/EXP-008/diagnose_context_noise.py --n-episodes 6`

## Reconciliation

Expected (D-010): at tau_ctx≈0.9–0.99 the decoded rollout preserves ball color and
approximate position for the first several generated frames, and pixel-MSE-vs-GT
drops sharply relative to tau_ctx=0.1; at 0.1 the failure reproduces.

Observed:
- gen-frame pixel-MSE vs GT (mean over 6 eps), **monotonic in tau_ctx**:
  tau=0.1 → 0.0289 | 0.5 → 0.0216 | 0.9 → 0.0184 | **0.99 → 0.0165** (−43% vs 0.1).
- Per-frame curve: at tau=0.9 the FIRST generated frame is near-perfect (MSE 0.0046)
  and error accumulates over the 12-frame rollout (→0.038); at tau=0.1 the first
  generated frame is already broken (0.022) and stays high. So conditioning works
  at high tau; the residual at high tau is ordinary autoregressive drift.
- Visual (images/): at **tau=0.1** the generated balls take random colors that do
  not match the GT (e.g. ep307 GT blue → rollout magenta/cyan/green) — the exact
  EXP-007 failure. At **tau=0.99** the generated ball keeps the GT color across all
  12 generated frames (ep307 stays blue; magenta-GT eps stay magenta; red→red;
  cyan→cyan), with mild late lightening/position drift.

Surprise: **mild.** Direction exactly as predicted; the only twist is that pixel-MSE
(background-dominated) needed ≥6 episodes to show the monotonic trend (the 2-episode
smoke was noisy and briefly looked non-monotonic). Visual evidence is unambiguous.

Hypothesis impact: **D-010 supported.** The EXP-007 rollout failure is an
**inference-only bug**, not a broken/undertrained dynamics model. The codebase
convention is tau = signal level (loss: `z_tilde=(1-tau)*noise+tau*z1`); the rollout
context-noising `ctx_noised=(1-tau_ctx)*noise+tau_ctx*context` with the default
`context_noise=0.1` therefore feeds the model **90% noise on the context frames**,
destroying the ball color/position it is meant to condition on → it emits a
plausible-but-random ball. Using near-clean context (tau_ctx≈0.9–0.99) restores
ball-identity preservation **on the existing checkpoint, with no retraining**.

This **reframes EXP-007**: H1's dynamics component is much closer to working than the
EXP-007 "dynamics broken" read suggested. (D-009's stated tripwire was about val/loss
leaking context; the actual reframing is the rollout path, not val/loss — but the
effect is the same: EXP-007's pessimistic dynamics verdict is substantially revised.)

Tripwires checked:
- D-010 primary ("high tau_ctx does NOT restore ball identity → not the cause"):
  **NOT triggered** — high tau_ctx DOES restore it.
- D-010 secondary ("even tau=0.99 diverges within 1–2 frames → suspect geometry"):
  **NOT triggered** — color holds across all 12 generated frames; only late drift.
- D-008 ("rollouts fail to preserve per-episode constants with visible context"):
  the cause is now identified (inference bug), so this is on track to clear once fixed.

## Decisive read

The EXP-007 dynamics model is not broken — the rollout was reading 90%-noise context
because `context_noise=0.1` is a *signal* level in this codebase (tau=1 is clean), so
the intended "light" corruption is actually near-total. With near-clean context
(tau_ctx≈0.9–0.99) the same checkpoint preserves ball color and approximate position
through the rollout; pixel-MSE on generated frames falls 43% and the first generated
frame becomes near-perfect. Residual late-rollout drift remains (ordinary
autoregressive accumulation) and is a separate, smaller problem. **Recommended fix: a
one-line change to the rollout context-noise default (≈0.9), then a confirmation
rollout of the EXP-007 checkpoint — no retraining needed to validate the H1 dynamics
baseline.** Latent-geometry (b) and undertraining (c) diagnoses (T-001b) are NOT
needed for this failure.

Next: PRESENT + ESCALATE (ESC-002) → hard stop for Merlin's review (§5).
View: `images/_sheet_tau0p10.png` vs `images/_sheet_tau0p99.png` (all 6 eps at a
glance); cleanest single case `images/ep307_s7_tau0p10.png` vs `..._tau0p99.png`.
