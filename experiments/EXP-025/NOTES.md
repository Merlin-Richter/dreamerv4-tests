# EXP-025 — GridWorld tokenizer v3 (foreground-weighted + training-stability fixes)

Hyp: H-gridworld (backbone) | Decision: D-043 | Branch: feat/motion-prediction @ 725a988
Cluster: ferranti (H100) | Job: 408737 | W&B: transformer-C-tokenizer / gridworld-tok-v3

## Purpose
Produce a USABLE frozen GridWorld tokenizer — one whose latents actually encode the
moving colored square, not background-only. Supersedes EXP-024 (gridworld-tok-v2, run
5d38b2nc), which FAILED.

## Why EXP-024 failed (re-diagnosis from its W&B curve — see D-043)
NOT pure sparse-target collapse. The curve shows the ball WAS being learned, then a loss
explosion destroyed it:
- ep8 val/mse 0.00226 → ep9 **0.000613** (latent_cos 0.29; latents becoming content-bearing)
- ep10 **EXPLOSION**: val/mse 0.0380 (62×), train/mse 0.0387 (17×), pred_std 0.37→0.25
- ep11–12 partial recovery; ep13–29 plateau at val/mse ~0.0037, never below ep9's 0.0006
- the "dropped ball" recon strips were from the POST-explosion ep29 checkpoint
Mechanism: at the loss minimum the Adam 2nd-moment shrinks → one batch lands an oversized
step → blowup. clip_grad_norm_(1.0) (already on) clips the gradient, not the Adam-scaled
step, so it can't catch it. The single-checkpoint overwrite then discarded the good ep9 model.

## What changed vs EXP-024 (D-043)
1. `--adam-beta2 0.95` — faster-adapting 2nd moment (mechanistic fix for the low-loss blowup).
2. `--grad-spike-mult 5.0` — skip the optimizer step on a non-finite / >5×-EMA pre-clip grad
   norm (backstop for the spike-shaped manifestation).
3. best-checkpoint by `val/fg_mse` — canonical `tokenizer.pt` holds the BEST ball-encoding
   model; `tokenizer_last.pt` holds the latest. An explosion can no longer discard the good model.
4. per-step W&B logging (`--log-every 50`) of step_mse/loss/grad_norm + per-epoch grad_norm/skips,
   so an intra-epoch explosion is VISIBLE (30 epoch points couldn't show it). Auto-step axis +
   global_step/epoch fields.
5. `--fg-weight 10` — modest foreground (ball) upweighting. DEMOTED from "the fix" to "a nudge":
   the ep9 evidence shows plain MSE already encoded the ball. val/fg_mse + val/bg_mse logged
   regardless (validity guard, D-042).

Else identical to EXP-024: datagen 3000×200 (6×6 env, D-038) → tokenizer 30ep bs64 lr3e-4
LPIPS-vgg --fresh → 6 recon strips.

## Verification gate (Merlin's instruction — do NOT trust low MSE)
Report success ONLY when the recon strips VISIBLY show the colored square at the right cell +
colour, distinct from the background — confirm one square is a different colour from the bg.
Cross-check: val/fg_mse dropped substantially (not background-only); grad_norm stayed bounded
(no explosion, or spikes were skipped & logged); val/mse went BELOW the ep9 0.0006.

## Reconciliation
(pending — fill on completion)
Expected: <from D-043> no explosion; val/mse < 6e-4 and improving; val/fg_mse low; ball visible.
Observed:
Surprise:
Hypothesis impact:
Tripwires checked:
Next:

## Precursor
`experiments/EXP-025-fgval/` holds an INTERRUPTED local α=30 validation attempt (tok_a30.pt,
stopped mid-ep2 on 2026-06-18) — superseded by this cluster run, kept only as a breadcrumb.
