# Fair bootstrap A/B (re-run the boot ablation without the confounds)

## Why
The earlier `mem2mem-rollout-boot` (job 411221) came back NEGATIVE — bootstrap "halved retention".
But that run was NOT a clean ablation. Reading the diff (2951ee6 winner → 3be2108 boot) showed the
bootstrap was confounded with THREE other changes:
1. **FF9 normalizer dilution.** FF9 is scaled by `diffusion.detach()/ff9.detach()`. In the boot run
   `diffusion` is the *mixed* flow+bootstrap per-token mean; the bootstrap self-distillation term is
   much smaller than the flow x-prediction loss, so it drags the mean down → silently down-weights FF9.
   The rollout-only WINNER used pure flow as the basis (larger) → gave FF9 more weight. (Smoke confirms
   the gap live: `flow 0.113` mixed vs `flow_norm 0.266` pure → ~2.4× under-weighting.)
2. **τ distribution shift.** Winner sampled new-half τ ~ `U{0..K_max-1}` (uniform). Bootstrap REQUIRES
   the d-snapped grid (its two-half-step target is only well-defined there), which piles ~25% of
   clean-mode tokens at τ=0 (vs <1% before) → much noisier new half. This is intrinsic to bootstrap.
3. **36 vs 50 epochs.**

So "shortcut forcing hurts retention" is unsupported. The bootstrap *gradient* (a stop-grad target the
winner already satisfies — it nails K=1 at 0.999) shouldn't hurt. This task isolates it.

## What "done" means
A clean 2-arm factorial on ferranti, both 50 epochs, both with the FF9 normalizer pinned to the pure
d_min flow magnitude (`--ff9-norm-flow`), both using the curriculum d-sampling (so τ is held identical):
- **Arm A (control):** `--boot-loss-off` — snapped-τ + curriculum, bootstrap LOSS off (coarse-d tokens
  get flow MSE). Isolates everything-but-the-bootstrap-gradient.
- **Arm B (fair boot):** bootstrap ON (default).
A vs B = pure bootstrap-gradient effect (τ identical). A vs the existing winner = the τ-shift effect.

Eval: recall @ window=8, max_k=64 (+ K=2/1), 4-way vs the winner (`dynamics_mem2mem_rollout.pt`) and the
old unfair boot. Decision: if B ≈ A ≈ winner → bootstrap is free, the old negative was the confounds
(intuition vindicated). If B < A → the bootstrap gradient genuinely hurts memory on this task. If A <
winner → the τ-shift is the culprit, not the bootstrap.

## Code (all behind flags, default behavior byte-identical — reversible)
- `experiments/mem2mem/rollout.py`: `_newhalf_loss` now also returns `flow_norm` (pure d_min flow
  magnitude); `mem2mem_rollout_loss(..., ff9_norm_flow=False)` picks the FF9 normalizer basis.
- `experiments/mem2mem/train_mem2mem.py`: `--ff9-norm-flow` and `--boot-loss-off` flags; `--no-bootstrap`
  still = winner repro (d_min only). Header + per-epoch log show `flow_norm` and effective bootstrap.
- Verified: `verify_newhalf_loss/probe.py` (|diff|=0, 4 cases), `test_autograd.py` (relay grad intact),
  local 1-epoch smoke of both arms.

## Status
- [2026-06-27] in-progress. Jobs submitted to ferranti (IDs + SHA recorded in the experiment NOTES).

## Result (2026-06-29) — DONE
Both jobs completed clean on ferranti (50ep, rc=0). Recall w8 max_k64 (K=4/2/1, n_roll=64):
Arm A (control, boot OFF) K4 0.998 / K2 1.000 / K1 1.000; Arm B (fair boot) K4 0.968 / K2 0.980 / K1 0.999;
winner 0.992; old unfair boot (411221) 0.472. Pre-registered verdict CONFIRMED: B≈A≈winner (the old
"bootstrap halves retention" was the confounds — chiefly the FF9-normalizer dilution: final FF9
0.013/0.0105 vs old boot's 0.054); B<A small/consistent (~3pts, not catastrophic); A≥winner so the
τ-shift is benign. Few-step: Arm A already perfect @K1/K2 ⇒ bootstrap motivation moot on GridWorld.
Decision unchanged: keep rollout-only + FF9 + x-prediction (bootstrap is free but pointless here).
Full write-up: experiments/mem2mem-rollout-boot-fair/NOTES.md + compare_w8_k64_4way.png.
