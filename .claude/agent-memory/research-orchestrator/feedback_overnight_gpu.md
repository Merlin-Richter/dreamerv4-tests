---
name: feedback-overnight-gpu
description: Merlin wants idle overnight GPU time used for any relevant training run, even a known-imperfect method
metadata:
  type: feedback
---

When it's late / the session is wrapping up and the 4070 would otherwise sit idle, Merlin wants
**something relevant training overnight** — and will explicitly accept a **known-limited / imperfect
method** to get a model by morning ("Can be using my reduced detached relay. Would still be interesting
to have a model trained with that.").

**Why:** overnight hours are free compute; a trained artifact (even of a method we know has flaws) is
worth more than an idle GPU. He'd rather have a result to look at than wait for the perfect setup.

**How to apply:** when wrapping up late with nothing running, proactively offer/launch a reasonable
overnight run. BUT weigh it against [[feedback-ground-claims-in-code]] discipline: don't ship a *rushed,
likely-broken* complex build that would waste the night (e.g. the bug-prone detached-relay loop the
verifier flagged). The right move was to run the SAFE, already-built, smoke-green thing (FF9 v2 baseline)
and defer the risky build — pick the highest-value run that is *low-risk to launch unattended*
(checkpoints every epoch, OOM-checked, no interactive prompts; skip --wandb if login could hang). State
the tradeoff to him; he values the peer judgment call.
