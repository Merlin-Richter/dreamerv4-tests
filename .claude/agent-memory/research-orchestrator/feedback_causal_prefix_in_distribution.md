---
name: causal-prefix-in-distribution
description: Shorter inference context than training length is IN-distribution for a causally-masked transformer
metadata:
  type: feedback
---

A causally-masked (autoregressive) transformer trained on length-N sequences is trained on EVERY
prefix length 1..N simultaneously — each position only attends to its own history. So running inference
with a SHORTER sliding context window (e.g. window 8 on a model trained at max_temporal_length 16) is
fully IN-distribution, NOT out-of-distribution.

Why: Merlin's correction (2026-06-24) — I called window-8 inference on the GridWorld dynamics model
"mildly OOD." He: "context window 8 is absolutely in-distribution, thats how transformers work with
masked training." Causal masking = the model has seen short contexts during training.
How to apply: don't hedge sliding-window / shorter-context inference as OOD for causally-masked models.
A smaller window legitimately accelerates the memory cutoff in demos without confounding the result.
(Does NOT extend to LONGER-than-train contexts — RoPE/position-table limits still bind there; see the
generate_cached RoPE caveat.) Relates to [[ground-claims-in-code]].
