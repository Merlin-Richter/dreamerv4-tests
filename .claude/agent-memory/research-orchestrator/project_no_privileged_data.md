---
name: no-privileged-data-constraint
description: Hard H3 constraint — the model never gets privileged data; methods must generalize across environments
metadata:
  type: project
---

The H3 memory work has two non-negotiable constraints (Merlin, 2026-06-12):
- **No privileged data to the model/training, EVER.** The model sees only environment
  observations + reward + data the env generated. No ground-truth hidden state (ball
  color/position/etc.) may be fed to the model as a training or inference signal.
- **Must generalize across environments** — no environment-specific hacks.

**Why:** the research goal is a universal memory mechanism for world models, not a
bouncing-ball-specific demo. A privileged crutch wouldn't transfer to real video / other
envs, so it's disqualified even as a stepping stone.

**Eval exception:** our measurement instrumentation (the revisit probe) *may* read the
sim's hidden color/position to **score** recall — that's measurement, not a model input.
Forbidden only as a signal to the model.

**How to apply:** reject any proposed objective that decodes/reconstructs from privileged
labels (these were FF2/FF4 in [[IDEAS.md]], now marked forbidden). Prefer self-supervised
forcing functions whose target is the env's own future observations (e.g. FF7
single-timestep-sufficiency). When proposing experiments, state explicitly how the signal
is obtained from obs+reward only. Related: [[feedback-measurement-validity]].