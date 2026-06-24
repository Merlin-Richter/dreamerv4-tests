---
name: verify-inference-path
description: For memory/architecture models, confirm the eval's inference path matches TRAINING before trusting recall numbers
metadata:
  type: feedback
---

When evaluating a model with a special architecture (FF9 memory tokens, FF7 registers, etc.), the
choice of INFERENCE function is load-bearing and easy to get wrong — verify it matches how the model
was TRAINED before trusting any number.

Why: 2026-06-24, EXP-028. I evaluated FF9 v2 via generate_full_state_memory (a frozen-snapshot special
case: writes ONE memory snapshot from the prefix, predicts every frame from a 2-frame [noise|new] window,
NO attention to recent frames → no dead-reckoning). That gave position_acc k=1 = 0.00 (absurd — k=1 is
trivially in context) and period-10 spikes, and I wrongly reported "FF9 solves static memory but not
dynamic position." Merlin: "laughably wrong." The model is actually TRAINED (_ff9_loss) with per-frame
memory tokens that attend causally to prior frames' memory tokens through the temporal channel — so the
correct inference is the NORMAL autoregressive rollout (generate_cached, memory_in=None), where memory
tokens are carried in the sliding window exactly like the frame latents. Both generate AND generate_cached
were short-circuiting to the snapshot path via a config guard; needed generate_cached(plain=True) to
bypass. With the right inference FF9 ≫ vanilla on DYNAMIC position too (0.94 vs 0.52 in-window).
How to apply:
- A k=1 / in-context sanity number that is near-zero is a RED FLAG the inference path is wrong, not a
  real result. Sanity-check the trivial end of every recall curve first.
- Read the training loss (e.g. _ff9_loss) to see how the special tokens are fed, then pick the inference
  that reproduces that token flow. Don't reason from the method's name/docstring or from priors.
- Dispatch guards (use_full_state_memory / use_register_memory) can silently route a "normal" generate
  call to a special-case rollout — check what generate/generate_cached actually do for the model's config.
- For a stronger-than-expected or reversed claim, run critical-claim-verifier on the inference before
  it becomes a headline. Relates to [[ground-claims-in-code]].
