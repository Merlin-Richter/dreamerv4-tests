# EXP-030 (+031/032) — FF9 rollout-training NOTES

## Purpose
Test whether training the memory->memory relay (op-3) on the gradient path (FF9 rollout-training,
D-048) makes the FF9 memory tokens carry DYNAMIC hidden state (square position) through occlusion past
where the untrained relay / the no-memory baselines fail. Gated by EXP-029 P1 (recall horizon ~ train
depth for dynamic state) -> train the relay to the eval depth.

## Inference semantics (READ THIS before interpreting the A/B)
Three inferences, all through the FROZEN recall core (D-045), env-direct episodes (N=64/k, n_ctx=8):
- **windowed** (EXP-028 corrected): plain sliding-window `generate_cached(plain=True)`. Memory tokens
  carried WITHIN the 16-frame window via temporal attention; latents are the main carrier. The model
  has the full latent window. Vanilla uses this (no memory).
- **relay** (D-048, the inference rollout-training trains): `generate_updating_memory`. A 2-frame
  `[noise-source | new]` window, the persistent memory token RE-WRITTEN and carried each step (op-3/B2).
  Source = pure noise (A1) => **memory is the ONLY carrier**; there is NO latent window. This is the
  honest "is memory a sufficient state" probe (the imagination north-star), but it is HARDER than
  windowed because it discards the latent history. NB: under relay, the curtain-UP "control" is NOT a
  free-run-in-the-clear (source is still noise) -> it is memory-only with a curtain-up action, so its
  position is also memory-limited, not a dead-reckoning ceiling. Don't read the relay control as the
  EXP-027 in-the-clear control.
- **snapshot** (B1, EXP-017/028 ref): frozen memory snapshot, static state only.

The clean attribution of "does rollout-training help" = **FF9+rollout vs FF9-no-rollout under the SAME
relay inference** (isolates training). The relay-vs-windowed gap is the inference difference.

## Baseline established this session: FF9 (no rollout) under RELAY (untrained B2 relay)
`recall_env_ff9_norollout_relay.json` (FF9 v2 checkpoint from EXP-028, gridworld-ff9-s0/dynamics_ff9.pt).
The untrained updating relay **collapses within ~2-3 hops**:
- position_acc (chance 0.028): k1 0.67 -> k2 0.42 -> k3 0.17 -> k4 0.08 -> ~chance (<=0.06) for k>=5.
- color_acc (chance 0.25): k1 0.98 -> k4 0.61 -> k8 0.28 (~chance) -> stays ~0.25-0.34.
=> The "before" picture: re-writing the memory each step WITHOUT training the relay corrupts it almost
immediately (matches V-T013-eval: B2 drifts). This is the bar FF9+rollout must clear under relay.
For windowed-inference baselines reuse experiments/EXP-028/recall_env_{vanilla,ff9}.json (vanilla cliffs
at the 16-window; FF9 decays to chance ~k28).

## Runs (pending; eval via EVAL_RUNBOOK.md when checkpoints land)
- EXP-030 (job 409752): FF9+rollout h24, window16, clip28, tbptt12, tail, +ff9 3, warmup20.
- EXP-031 (job 409754): FF9+rollout h44 deep, clip48, tbptt16.
- EXP-032 (job 409753): vanilla window-32 control.

## Reconciliation (fill when results land)
Expected (D-048): FF9+rollout under relay holds position past k=4 (clears the untrained-relay collapse)
and past the 16-window; h44 reaches further than h24 (P1 horizon~depth); in-window (k<=8) NOT regressed.
Observed: TBD. Surprise: TBD. Hypothesis impact: TBD. Tripwires: TBD. Next: TBD.
