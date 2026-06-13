---
name: feedback-sequence-subobjectives
description: Merlin wants one subobjective nailed in isolation before the next, but keep the big-picture end-goal visible — don't scrub it
metadata:
  type: feedback
---

Focus on ONE subobjective at a time and finish it cleanly before moving to the next — don't try to do
the whole arc at once. BUT keep the big-picture end-goal visible in the strategy docs; don't over-correct
and scrub it when narrowing scope.

**Why:** On the cross-frame KV cache (T-012), Merlin said "I don't want you to think about training
objectives or gradient graphs, just that we could run efficient sliding-window continuous rollouts using
KV caches." I over-corrected and stripped ALL training framing everywhere. He clarified: the big-picture
aim *is* efficient register-relay rollout training (IDEAS.md); he just wanted the rollout-cache
subobjective done first, in isolation — "big picture we are aiming for [it], just not right this second."

**How to apply:** When he scopes you to a subobjective: (1) build/verify just that piece, keep premature
downstream implementation detail (e.g. gradient-graph/stop-grad specifics) out of THAT piece's code +
docstrings; (2) but in ORIENT/plan, keep the end-goal as a parked big-picture line so the sequencing is
visible and the work doesn't look aimless. Both at once = wrong; neither = wrong. See [[feedback-milestone-reevaluate]].