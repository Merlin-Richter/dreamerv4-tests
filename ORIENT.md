# ORIENT.md

Rewritten: 2026-06-13 ~07:00 (ESC-006 resolved → redirected; EXP-011 diagnostic launching)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75; T-004 H3 bar = color ΔRGB < ~63 at
  n_occ ∈ {12,16,24}.
- **H3 — FF7 v1 supports it for COLOR (EXP-010).** Both arms move the post-window color
  cliff off chance, clearing the bar at n_occ 12 & 16; k=3 > k=1; no degradation tripwires.
- **NEW redirection (Merlin, ESC-006):** position is at chance even in OPEN rollout — the
  base model never learned to track motion (predates FF7; my_dynamics & EXP-009 same). So
  "can a memory method retain position/momentum?" is UNPROVEN until the base model can do
  position in the clear. **Diagnose before fixing.** → D-015 / EXP-011.

## In flight
**EXP-011 — no-training position-deficit diagnostic** (D-015, local 4070, building inline).
Goal: (i) confirm/quantify deficit, (ii) LOCALIZE tokenizer C vs dynamics D (linear-probe
latents→xy), (iii) disambiguate (a) "never learned motion" vs (b) "learned motion, open-loop
chaotic desync from GT trajectory". Components: GT ball kinematics (states vx,vy — zero
inference); copy-last & chance position baselines vs model OPEN-loop pos_err vs horizon;
closed-loop/teacher-forced 1-step pos_err along trajectory; linear probe of frozen tokenizer
latents → (x,y); qualitative rollout look. Artifacts → experiments/EXP-011/.

## NEXT ACTION
Build + run EXP-011 (no training → no present-then-stop gate is *required* mid-build, but the
RESULT is a §5 present-then-stop: reconcile in EXP-011/NOTES.md, build a view, decisive read on
(a) vs (b) and C-vs-D, ESC-007, stop for Merlin). My priors (D-015): lean (b) chaos + position
decodable from latents (deficit, if any, in D) — but the tripwire is: if position is NOT
decodable from tokenizer latents, the bottleneck is C and that reframes all H3 position work.

## Current worries
1. **Open-loop GT-matched position is metric-ambiguous** — could read chance under (a) OR (b).
   The closed-loop + linear-probe components exist precisely to break that ambiguity; make sure
   they actually do before declaring a verdict.
2. **Don't overfit the diagnostic to my prior.** I predicted (b); run the copy-last comparison
   honestly — if the model only matches copy-last, that's (a) and it overturns the FF7 framing.
3. Reuse the frozen probe's detector/env exactly (detect_ball, make_probe_episode, states) so
   numbers are comparable to EXP-009/010 — do NOT introduce a second detector.
