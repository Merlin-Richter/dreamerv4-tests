# Why the vanilla dynamics model can't predict square positions even IN-window (no occlusion)

**Date:** 2026-07-04. **Provenance:** branch `exp/mem2mem-rollout-only` @ `17dde99`;
checkpoints `checkpoints/gridworld/` (SHA-1688818-era: `dynamics_vanilla.pt`, `dynamics_ff9.pt`,
`dynamics_mem2mem.pt`, `dynamics_mem2mem_rollout_noff9_clean.pt`), frozen `tokenizer.pt`.
Probes run local (4070), B=64 env seeds, K=4. Scripts: `probe_next_pos.py`, `probe_tau_sweep.py`;
raw numbers in `results_next_pos.json`, `results_tau_sweep.json`.

**Question (Merlin):** `sheet_normal.png` shows vanilla gets colors right but not square positions,
fully revealed, in-window — nothing to do with memory. mem2mem (rollout-only, no-ff9) tracks the
square even occluded. Same compute. Why?

## TL;DR

The vanilla diffusion-forcing objective almost never *demands* next-frame prediction. ~75% of
sampled frames have enough signal (τ) that the square's position can be read out of the frame's
**own noisy latent** — a denoising task, no dynamics needed. The only samples that force
position-from-context are (ground-truth flow target AND τ≈0), which is **1.3% of frames, and the
ramp weight `w(τ)=0.9τ+0.1` cuts that to ~0.4% of total loss weight**. The other τ=0 frames (a full
25% of frames have τ_idx=0!) get the *bootstrap* target — self-distillation of the model's own
finer-step chain, which contains no new ground truth. So vanilla learns "denoise + paint the right
colors" (val/loss 0.0016, looks great) and never learns the transition map. FF9 / mem2mem learn it
because their extra terms put ~all of their weight on exactly the starved case: reconstruct a frame
whose own latent is pure noise, against ground truth.

Inference then runs *exactly* in the starved regime — every generated frame starts at τ=0 — so the
model emits a positional smear, and the subsequent denoise steps (which it IS good at) sharpen the
smear into a crisp square at a near-truth-but-wrong cell. Hence: right colors, wrong position, and
it isn't copying the last seen cell either.

## Evidence

### Probe 1 (`probe_next_pos.py`) — the failure is in-window learning, not rollout compounding

Teacher-forced ONE-step prediction from an all-real, fully-REVEALED context (action=0 everywhere;
no occlusion ⇒ zero memory demand; read-only `rollout_step`, decode, exact readout):

| model (ckpt) | t=2 | t=4 | t=8 | t=15 | copy_rate | col_acc |
|---|---|---|---|---|---|---|
| vanilla | 0.078 | 0.094 | 0.078 | 0.094 | ~0.05 | 1.00 |
| ff9 | 0.922 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 |
| mem2mem 50/50 | 0.656 | 0.984 | 1.000 | 1.000 | 0.00 | 1.00 |
| m2m rollout no-ff9 | 0.875 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 |

- Vanilla ≈ chance-ish (exact-cell chance 1/36≈0.028) at EVERY context length; more context does
  not help; mean Chebyshev distance ~2.3; `copy_rate`~5% ⇒ it is NOT freezing the last observed
  cell — it paints a right-colored square at a loosely-related position.
- Free-run from 4 ctx (the `sheet_normal` setting) is the same story from step 1 (0.17 → 0.05 by
  j=12); no cliff ⇒ not error compounding, it never had the map.
- This also re-reads the recall curves: vanilla's ~chance at the earliest scored k (=2) was never
  a *memory* deficit — it fails before memory is even in play.

### Probe 2 (`probe_tau_sweep.py`) — mechanism: denoise-shortcut, and the architecture is innocent

Real 8-frame episode, context near-clean (τ_ctx grid point of 0.9), LAST frame's own latent noised
to a swept τ, ONE forward at finest d (exactly the training flow term), decode x̂₁, read the square:

- **vanilla**: pos_acc 0.36 @τ=0 → 0.50 @τ=0.13 → 1.00 @τ=0.99. flow MSE **0.258 @τ=0** vs 0.0078
  @τ=0.99. Its τ-averaged val loss (0.0016) is dominated by frames where denoising suffices; the
  τ≈0 slice is ~100× worse and nothing in the objective pushes it down.
- **ff9, same forward, learned-init memory tokens (NO carried memory)**: pos_acc **1.000 at τ=0**,
  flow MSE 0.0042 (62× below vanilla). ⇒ position-from-context is fully learnable through the
  plain latent/temporal-attention pathway in-window. Not an architecture/capacity/attention-mask
  problem; purely a training-pressure problem.

### The weight-share arithmetic (from `sample_tau_d` + the loss, 2M samples)

- P(ground-truth flow target, i.e. d=d_min) = 1/8; all coarser d get the bootstrap (stop-grad
  self-distill) target.
- P(GT flow AND τ≤0.1) = **0.0128**; after ramp weight w(τ): **0.42% of total loss weight**.
- P(τ_idx=0, any d) = 0.249 — a quarter of frames are pure noise, but ~all of them are trained
  against the model's own bootstrap chain, not ground truth. GT correctness can only seep in
  through the 0.4% slice.
- The ramp's rationale in the spec ("low-τ frames carry little signal; flow collapses to the
  mean") is imported from stochastic-env reasoning. GridWorld is DETERMINISTIC: at τ=0 the target
  is exactly predictable from context — the ramp starves precisely the one term that teaches
  dynamics.
- Secondary factor: even at τ=0, a mispositioned square costs little latent MSE next to painting
  the correct background/color — which is also why color is learned perfectly (static, dominates
  MSE, readable from any context frame) while position (precise 2-frame relational read +
  extrapolation) never forms.

### Why FF9 / mem2mem escape

Their extra terms are ~100% "τ=0 vs ground truth": FF9 sets ALL path latents of the mini-window to
pure noise and reconstructs 1..k frames against GT (normalized to diffusion magnitude — can't be
diluted away); mem2mem rollout training puts 50% of new-half frames at τ=0 with GT targets, with
committed self-generated context. Getting position right is the *only* way to reduce those terms.
And per Probe 2, the skill generalizes back into the plain no-carried-memory forward.

## Implications (for Merlin — decisions, not made here)

1. **The current vanilla is a weak baseline, and that CONFOUNDS the memory story.** "Memory tokens
   help retention past the window" needs a baseline that is competent IN-window and fails only
   PAST it. Today's vanilla fails in-window, so recall gaps vs mem2mem overstate the memory
   effect. Reviewers/critics would find this exactly the way this probe did.
2. **The memmaze campaign inherits the confound.** The just-finished memmaze vanilla arm (415103)
   trained the same objective; part of its "diverges from GT within a few steps (expected — no
   memory)" sheet behavior is plausibly this same in-window starvation, not missing memory. The
   3-way memmaze comparison (vanilla / mem2mem / no-ff9) should be interpreted with this in mind.
3. **Cheapest honest-baseline fix (no memory tokens, no arch change):** give vanilla the same τ=0
   GT pressure the memory arms get — e.g. with prob p (~0.25–0.5) force (τ_idx=0, d=d_min) so the
   frame is trained as pure next-frame prediction from context against GT; and/or drop the ramp
   for the d_min flow term (deterministic env ⇒ the "collapses to the mean" rationale doesn't
   apply). Both are small `sample_tau_d`/loss-weight changes — but they touch spec-backed `src/`,
   so they need a spec edit (Merlin).
4. Optionally the same "noise-mode" idea as a vanilla-rollout arm (mem2mem-style sliding rollout
   training minus the memory tokens) would make training-regime vs memory-tokens a clean 2-factor
   design.
