# EXP-007 — Action-conditioned dynamics on CurtainsEnv (backfilled)

Decision: D-008 | Hypotheses: H1 (dynamics), H2 (informative) | 2026-06-10, cluster

## Provenance

W&B: [sm0kr1cf "denim-waterfall-2"](https://wandb.ai/models-eberhard-karls-universit-t-t-bingen/transformer-D-dynamics/runs/sm0kr1cf)
Commit: 3205e8e | Host: mlcbm002 | Runtime: 31 min, 100 epochs
Data: `occluded.npy` + `occluded_actions.npy` (n_actions=2, 1 action token/frame)
Tokenizer: `trained_autoencoder.pt` (frozen; = EXP-006 arm B / rc01geau)
Config: embedding_dim 256, depth 8, n_heads 16, n_latents 4, bottleneck_dim 64,
n_registers 4, max_sampling_steps 128, inference_steps 4, context_noise 0.1,
context_frames 4, max_temporal_length 16, lr 3e-4, bf16, batch 32.
Artifact: `my_dynamics.pt` (2026-06-10 19:47 local ≈ run finish 17:47 UTC).

## Reconciliation

Expected (from D-008): rollouts preserve ball color and background always, and
ball position when context has ≥2 visible frames.
Observed: train/loss 2.87e-3, val/loss **1.93e-3** — healthy-looking curves. But
decoded rollouts (4 context frames + autoregressive generation, interactive
viewer): from the first generated frame, **ball color and position are
randomized**; background gradient **is preserved**. Control: decoding random
bottleneck tokens produces *no* ball at all — so the model has genuinely learned
to predict "latents containing some ball", just not *which* ball *where*.
Rollout screenshots from the journal were not archived (provenance gap; future
experiments save these under `experiments/EXP-NNN/`).
Surprise: **high.**
Hypothesis impact: H1 dynamics component NOT yet supported. H2: suggestive (this is
exactly what no-memory looks like) but **confounded** — failure occurs even with
fully visible context, so it cannot currently be attributed to occlusion-memory
rather than a broken dynamics model.
Tripwires checked: D-008 "rollouts failing to preserve per-episode constants" —
**triggered.**

## Decisive read

The dynamics model is not yet a usable H1 baseline; the failure is not subtle. The
loss/rollout mismatch suggests either the shortcut-forcing objective rewards
something rollouts don't exercise (e.g. bootstrap self-chasing), or the tokenizer's
latent geometry makes per-episode constants hard to carry (adjacent frames mapping
to distant latents), or plain undertraining (31 min). The random-latents control
rules out "decoder hallucinates balls"; the model is doing *something* right.
Candidate causes are written as (a)/(b)/(c) in D-009 with a diagnosis plan.

Next: ESCALATE → ESC-001. Verdict received: diagnose first (T-001), probe suite
second. → D-009.

---
**UPDATE 2026-06-11 (EXP-008 / D-010):** Cause found. This rollout failure is an
**inference-only bug**, not a broken or undertrained dynamics model. The rollout
context-noising feeds 90% noise on the context frames (`context_noise=0.1` is a
*signal* level in this codebase). With near-clean context (tau_ctx≈0.9–0.99) the
SAME checkpoint preserves ball color/position. So this NOTES' "dynamics not yet a
usable H1 baseline" read is substantially revised — the model is much closer to
working than it looked here. See `experiments/EXP-008/NOTES.md`, ESC-002. (Append
only; original reconciliation left intact above for the record.)
