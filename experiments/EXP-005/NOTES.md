# EXP-005 — Unconditional dynamics on BouncingBall (backfilled)

Decision: D-005 | Hypothesis: H1 | ~2026-06-01, local
Provenance: approximate; artifact `dynamics_bouncing.pt` (2026-06-01). A later
short logged run exists: W&B `transformer-D-dynamics/fq5wpzk9` (2026-06-09,
killed, val/loss 2.48e-3 @ epoch 9, commit 7cb30c1). Pre-W&B training not logged.

Purpose: first shortcut-forcing dynamics model, unconditional, on frozen C
tokenizer, `bouncing.npy`.

Expected: plausible short rollouts.
Observed: qualitative rollouts via interactive viewer
(`train_dynamics_model.py --test-checkpoint`); judged plausible enough to proceed.
No archived rollout images or probe metrics — by current standards this experiment
is under-evidenced.
Surprise: none recorded.
Hypothesis impact: H1 dynamics component provisionally working on the *memoryless*
env. Note (added 2026-06-11): in hindsight, "plausible" was never verified beyond
eyeballing; EXP-007's failure means this judgment should not be leaned on.
Next: action-conditioned training on CurtainsEnv (D-008) after tokenizer upgrade.
