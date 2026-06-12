---
name: feedback-measurement-validity
description: Merlin treats measurement-validity details (can we actually measure the quantity we claim?) as first-class, especially for the probe suite
metadata:
  type: feedback
---

Merlin scrutinizes whether a metric can actually be measured cleanly before trusting
it. He raises measurement-validity objections that must be designed for, not
hand-waved.

**Why:** When approving "measure color + position recall" for the probe suite
(2026-06-12), he flagged: "we care about the color of the circle, but we don't know
where the circle is in the rollout images to fetch its color and evaluate. This is a
crucial testing detail." I.e. color recall is entangled with localizing the
(possibly mispredicted/absent) ball in the decoded rollout frame.

**How to apply:** When designing any probe/metric, explicitly state how the predicted
quantity is extracted and what happens in edge cases (ball absent, mislocated). For
this project the resolution leans on the dual metric: the latent-space probe reads
color/position from the predicted latent detection-free, while pixel-space gives the
visual headline + a blob-detection cross-check; "ball not detectable" is tracked as
its own failure mode. Relates to [[feedback-milestone-reevaluate]].
