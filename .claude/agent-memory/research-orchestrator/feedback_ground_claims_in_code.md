---
name: ground-ml-claims-in-code
description: Verify architecture/training-dynamics claims against the actual code BEFORE asserting them in design dialogue
metadata:
  type: feedback
---

Before asserting how the model behaves, or what an architectural/training change would
require, **read the relevant module and cite it (`file:line`)**. Do not reason from ML
priors alone during method design.

**Why:** During H3/FF7 design (2026-06-12) I made two confident conceptual errors —
(1) "scale the prediction horizon k to span the occlusion" and (2) "registers need
recurrent output→input feedback wiring plus a detach or the write gets no gradient."
Both were wrong, and both were resolved the instant I read `dynamics_model.py`
(register_tokens is a broadcast learned init; temporal attention is *position-wise*, so
each register slot is already its own causal channel through time → the carry exists with
no wiring). Merlin caught both and flagged it shook his confidence in my ML judgement. The
failure was treating plausible intuition as evidence — a violation of the harness's own
"trust artifacts, not reports" directive applied to my own mental model.

**How to apply:** In any method/architecture discussion, the moment a claim depends on how
a specific module works (attention masking, token layout, what's read out, gradient paths),
open the file and verify before stating it. Prefer a code citation over a confident
sentence. This is cheap (one Read) and is the root-cause fix — not offloading ML reasoning
to a subagent. A fresh read-only "methods critic" at design-finalization gates is an
optional backstop, but the primary fix is this verification discipline. Related:
[[no-privileged-data-constraint]].