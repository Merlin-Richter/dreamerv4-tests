---
name: gridworld-metric-semantics
description: What the GridWorld recall metrics actually test — colour = static memory, position = memory + reasoning
metadata:
  type: project
---

In the GridWorld memory eval, BOTH colour and position are memory tests, but of different kinds —
do NOT frame position as "the only memory metric."

- **Colour recall = static memory.** The square's colour is fixed, so retaining it through occlusion
  is pure static-state retention. A memoryless copy-last baseline passes it trivially (freezing the
  last frame preserves a static attribute), BUT a real dynamics model can still FAIL it (hallucinate a
  different colour) — that is a genuine memory failure. So colour is an easy/static memory metric, not
  a no-op.
- **Position recall = memory + reasoning.** Position changes every hidden step, so the model must
  retain the last observed state AND simulate the motion forward (bounce/wall reflections) across k
  occluded steps — "reasoning inside memory," not just retention. Copy-last fails it except at the
  bounce period (k≡9 mod 10 on the 6×6 env).

Why: Merlin's correction (2026-06-24) after EXP-026, where I'd written "position is the ONLY memory
metric" because copy-last aces colour. That conflated "copy-last passes it" with "it's not a memory
test." This mirrors the occluded-line H2/H3 split (colour = static hidden state; position = dynamic).
How to apply: report BOTH curves as memory results — colour as the static-retention check (does the
model keep a fixed attribute, vs copy-last as a trivial-but-passable floor), position as the harder
dynamic-reasoning headline (beat copy-last per-k). Relevant to [[no-privileged-data-constraint]] and
[[feedback-measurement-validity]].
