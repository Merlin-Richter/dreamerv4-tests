---
name: project-autoresearch-generality-rule
description: Merlin's rule for the autoresearch harness — editable-layer changes must be environment-GENERAL; env-specific exploits are cheating even when they score well
metadata:
  type: project
---

For the autoresearch harness (ColorField tiers): changes the loop (or I) make to the editable
layer (model/objective/training) MUST be environment-general — they may not encode knowledge of
this specific environment's structure.

**Why:** Merlin, 2026-07-07: "the model and training ideas need to work in any environment...
increasing loss on every fifth frame during training is cheating. If the auto researcher
implements a general approach such that new revealed information is weighted more, that's fine,
but it may not use its knowledge about the specifics of this environment." The harness exists to
find TRANSFERABLE training/architecture ideas (promotion ladder: colorfield-sym → pixel
colorfield → memmaze); env-specific hacks score without transferring.

**How to apply:** when writing or reviewing editable-layer changes (mine or the loop's), ask
"would this mechanism be well-defined and sensible in a different environment?" Loss weighted by
phase-index/known periodicity/board size/palette = cheating. Loss weighted by prediction error,
information gain, or measured novelty = fine. The rule belongs in program.md and in the human
review of kept diffs. Related: [[feedback-spec-edit-delegation]].
