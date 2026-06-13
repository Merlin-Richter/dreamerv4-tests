# ESCALATIONS.md

> One entry per open question for the human. Resolutions are written back
> verbatim-in-substance; steering not written back here evaporates.

## ESC-001 | 2026-06-11 | RESOLVED
Context: EXP-007 — action-conditioned dynamics on CurtainsEnv reached healthy
val/loss (1.93e-3) but rollouts randomize ball color and position from the first
generated frame (background preserved; random latents decode to no ball).
D-008 tripwire triggered. Simultaneously: adoption of the
research-orchestrator protocol with backfilled state files.
Access points: W&B run
https://wandb.ai/models-eberhard-karls-universit-t-t-bingen/transformer-D-dynamics/runs/sm0kr1cf ;
decisive read in `experiments/EXP-007/NOTES.md`. (Rollout screenshots from the
journal not archived — noted as provenance gap.)
Question: (1) Is the H1→H3 hypothesis framing in GOAL.md right? (2) What comes
first: dynamics-failure diagnosis or the §8 probe suite? (3) How were cluster runs
submitted so far / how to record? (4) Metrics sourcing for backfill?
Urgency: blocking (no further work until answered).

Resolution (Merlin, 2026-06-11):
1. GOAL.md H1–H3 framing approved as proposed.
2. "Diagnose the dynamics failure first (it blocks everything), probe suite
   after." → D-009, T-001 before T-002.
3. Cluster runs were done manually by Merlin so far; record wrapper scripts as a
   pending board task → T-003.
4. Experiment information is to be fetched from the W&B API (MCP/`wandb` Api);
   entity `models-eberhard-karls-universit-t-t-bingen`. Pre-W&B experiments stay
   qualitative — no invented numbers.
Applied to: GOAL.md (created), DECISIONS.md D-009, BOARD.md T-001..T-004.

## ESC-002 | 2026-06-11 | OPEN — present-then-stop (EXP-008 review)
Context: T-001 diagnosis of the EXP-007 dynamics rollout failure. A cold-start code
read of `dynamics_model.py` (D-010) found the likely cause; EXP-008 (inference-only
tau_ctx sweep on the existing `my_dynamics.pt`, no retraining) confirmed it.

**Finding (decisive read):** The EXP-007 dynamics model is **not broken**. The
rollout was conditioning on ~90% noise. In this codebase `tau` is the *signal* level
(loss: `z_tilde=(1-tau)*noise+tau*z1`, tau=1 clean), but the rollout context-noising
`ctx_noised=(1-tau_ctx)*noise+tau_ctx*context` with the default `context_noise=0.1`
puts 90% noise on the context frames — the intended "light" corruption is actually
near-total. The model therefore can't read ball color/position from context and emits
a plausible-but-random ball. With near-clean context (tau_ctx≈0.9–0.99) the SAME
checkpoint preserves ball color across the rollout; gen-frame pixel-MSE falls 43%
(0.0289→0.0165) and the first generated frame becomes near-perfect (0.0046 at 0.9).
This is an **inference-only, one-line-fix bug — no retraining needed** to validate
the H1 dynamics baseline. It substantially revises EXP-007's pessimistic verdict.

**Access points (low-friction view):**
- Side-by-side, all 6 episodes at a glance: `experiments/EXP-008/images/_sheet_tau0p10.png`
  (broken) vs `experiments/EXP-008/images/_sheet_tau0p99.png` (fixed).
- Cleanest single case (blue-ball episode): `experiments/EXP-008/images/ep307_s7_tau0p10.png`
  vs `experiments/EXP-008/images/ep307_s7_tau0p99.png`. (Top row GT, bottom row
  rollout, red line = context/generation boundary.)
- Numbers: `experiments/EXP-008/results.json`. Full reasoning: `experiments/EXP-008/NOTES.md`.

**Question:** (1) Do you agree with the read? (2) Next step — my recommendation:
change the rollout context-noise default to ~0.9 (one line; possibly a quick 0.9 vs
0.95 vs 0.99 sweep to pick), re-run the EXP-007 checkpoint rollout to confirm the H1
dynamics baseline qualitatively, then proceed to the probe suite (T-002). Skip the
broad latent-geometry/undertraining diagnosis (T-001b) — not needed for this failure.
(3) A semantics question for you: should `context_noise` keep meaning "signal level"
(so 0.9 = light noise) or be inverted to mean "noise fraction" (so 0.1 = light)? The
latter matches the variable name + comments but touches the train/test story — your
call since it's a convention choice. (4) Note: residual late-rollout drift remains at
high tau_ctx (ordinary autoregressive accumulation) — separate, smaller, later.

**Harness/process note:** I built+ran the EXP-008 diagnostic *inline* rather than
spawning a worker subagent. The protocol (§4) pushes worker delegation, but the Agent
tool here carries an explicit guardrail ("don't spawn unless the user asks; cold
spawns re-derive context — the expensive path"). For a ~140-line single-threaded task
I was blocked on anyway, neither delegation benefit (context economy, parallelism)
applied, so I kept it inline. Flagging the tension: if you want me to use the
worker/worktree loop even for small tasks (to exercise the harness), say so and I'll
switch; otherwise I'll reserve workers for large or parallelizable implementation.

Urgency: blocking — per §5 every experiment ends in a hard stop for your review; I am
NOT starting the fix, the confirmation rollout, or T-002 until you weigh in.

Resolution (Merlin, 2026-06-12):
- Agrees with the decisive read: the context-noise fix has solved the dynamics-model
  rollout failure. "Good find."
- Agrees T-001b (broad latent-geometry / undertraining diagnosis) no longer makes
  sense — dropped.
- Considers this to complete **H1**.
- Calls a **milestone conversation** to plan the next trajectory and review the tasks
  needed to get there. → ESC-003.
- NOT answered here: Q3 (context_noise "signal-level" vs "noise-fraction" semantics)
  and the mechanics of applying the fix — carried into the ESC-003 milestone.
Applied to: ESC-003 (milestone, below). GOAL.md H1 status + the code fix to be
applied only after the milestone concludes (§7).

## ESC-003 | 2026-06-12 | RESOLVED — H1-closure milestone, Phase 2 planning
Context: H1 complete (ESC-002). Milestone to plan Phase 2 (H2) and reevaluate tasks.
Three design questions answered + two architecture corrections from Merlin.

Resolution (Merlin, 2026-06-12):
- **Probe metric:** "Both, pixel primary" → refined in discussion to latent-token MSE
  as the candidate primary (decoder/detection-free), validated against a pixel-space
  color/position decomposition. He flagged the crucial measurement detail: we can't
  read the ball's color without knowing where the (possibly mispredicted/absent) ball
  is — latent-MSE sidesteps this; "ball not rendered" is its own failure mode.
- **Probe scope:** color + position together in v1.
- **context_noise fix:** keep tau=signal-level convention, rename→`context_signal`,
  fix comment, default 0.9 (T-007).
- **Don't mechanically continue the backlog — reevaluate** (saved to agent memory).
- **Two architecture corrections (he was right, I conceded on the merits):**
  (1) M<N inference needs no retrain (RoPE relative). (2) A sliding-window transformer
  has no persistent state — no "boundary to carry across"; info older than N−1 frames
  is simply absent. The dynamics `generate()` already slides the window, so the
  beyond-window regime is reachable today with no new architecture/retrain.
- **KV cache** = efficiency, not a prerequisite. RoPE/KV-cache rotation-continuity
  trap flagged → `HOWTO/rope_kv_cache_caveat.md`.
- **H3 reframed:** open-ended end-goal ("force encoder and/or dynamics to include
  hidden info in the latent space; how is up in the air; try many, keep what sticks"),
  not a single pre-registered hypothesis. Don't over-formalize it.
- **Verdict:** "you should have enough now to get to work on H2. Continue as you see
  fit. I'm always available."
Applied to: GOAL.md (H1→supported, H2 corrected, H3 reframed), DECISIONS.md D-011,
BOARD.md (tasks rescoped), HOWTO/rope_kv_cache_caveat.md, ORIENT.md (rewritten),
agent memory (milestone-reevaluate, measurement-validity).

## ESC-004 | 2026-06-12 | OPEN — present EXP-009 (H2 baseline cliff) + T-004 pre-registration
Context: EXP-009 (frozen probe @ f1cf860, 64 eps/n_occ) is the H2 baseline on the vanilla
sliding-window model `my_dynamics.pt`. This is a present-then-stop gate (§5): every
experiment ends here for your review, even a clean expected result.

### The result (decisive read)
H2's premise is **supported**, cleanly. Hidden-color recall is at the ceiling while the
color-carrying prefix is inside the N=8 window and collapses to the chance floor the
instant it scrolls out — a sharp cliff between n_occ=6 and n_occ=7 that matches the
sliding-window geometry to the frame (prefix exits when reveal index 3+n_occ > 9). The
matched-horizon drift control rules out ordinary autoregressive degradation (drift color
ΔRGB stays 17–40 while occluded jumps to ~110). latent-MSE tracks color at r=0.952, so it
is validated as the detection-free headline metric; position is drift-confounded and demoted
to secondary. Detector gate green (p99 0.65px). This is exactly the calibrated baseline H3
methods must beat: the bar for "memory" is moving the post-window cliff off the chance floor.

### Access points
- Numbers: experiments/EXP-009/results.json ; full reconciliation: experiments/EXP-009/NOTES.md
- Visual (GT top / prediction bottom, red line = context|gen boundary, yellow box = reveal):
  experiments/EXP-009/sheet.png — n_occ=2,6 colors match GT; n_occ=8 GT magenta → predicted
  green (the cliff). (n_occ=12,24 single samples coincidentally land near GT; the 64-ep
  aggregate is the evidence, not those frames.)
- Headline: color ΔRGB ceiling 15.9 / chance 109.9 ; occluded 16.8 (n_occ=6) → 94.4 (n_occ=7)
  → 116.0 (n_occ=8). latent-MSE 0.27 ceiling / 0.88 chance.

### The question for you (T-004 pre-registration — lock BEFORE any H3 method runs)
I propose we register these H2 baseline criteria now, while no H3 method exists (keeps
pre-registration honest):
1. **Headline metric:** color ΔRGB at the reveal frame, occluded vs matched-horizon drift.
   Secondary/validating: latent-token MSE (r=0.95 with color). Position: reported but flagged
   drift-confounded, NOT a success metric.
2. **Baseline claim (H2):** vanilla sliding-window recall = chance for n_occ >= N-P+? (here
   n_occ>=7 at N=8,P=3). I.e. no hidden-state retention past the window. Supported by EXP-009.
3. **H3 success definition:** a method "retains hidden state" if, at n_occ well beyond the
   window (propose n_occ in {12,16,24}), its color ΔRGB is significantly below the chance
   floor (propose: < halfway between ceiling and chance, i.e. < ~63), under the identical
   frozen probe and matched-drift control.
Do you agree with (1)-(3), and specifically the n_occ>>N test points and the "halfway to
ceiling" H3 bar? Adjust any threshold you'd set differently. After your call I'll write
T-004 + update GOAL H2 status, then we're clear to start H3 method exploration.

### ESC-004 RESOLVED (Merlin, 2026-06-12)
- "Yes I agree. This proofs H2." → H2 declared **supported** (GOAL updated, D-012, T-004 locked).
- T-004 pre-registration criteria (1)-(3) approved as proposed (color headline, latent-MSE
  secondary, position confounded; H3 bar = color ΔRGB < ~63 at n_occ ∈ {12,16,24}).
- Naming feedback: `drift_by_occ` is a bad name for the curtain-stays-up control ("has
  nothing to do with occlusion"). → D-013: renamed to `matched_horizon_drift` (code +
  EXP-009 artifact migrated in place, numbers preserved; probe re-frozen).
Applied to: GOAL.md (H2→supported + criteria), DECISIONS.md D-012/D-013, tasks/T-004.md,
src/probe/ (rename + README re-freeze), experiments/EXP-009/results.json (key migrated).

## ESC-005 | 2026-06-12 | OPEN — two items awaiting Merlin before H3 build (context-reset checkpoint)
Context: H2 closed (ESC-004). H3 entered; first method (FF7) designed and code-grounded with
Merlin in live dialogue. Recorded here so a fresh session resumes without re-litigating.

1. **FF7 build go-ahead.** Design converged & code-grounded; full v1 scheme in `IDEAS.md`
   → "Proposed first attempt — FF7 v1". Key points: single-timestep-sufficiency loss
   (window-1 rollout: predict next k=1 from one frame, latents overwritten with the real
   frozen-tokenizer latent so the register is forced to be the carrier); **training-procedure
   change to `train_dynamics_model.py` ONLY — no architecture change** (registers already
   carry via position-wise temporal attention, dynamics_model.py:110-121); arbitrary-action
   coverage from dataset curtain-timing diversity (v1), adversarial actions later; eval on
   frozen probe 5503e75, ≥2 seeds, T-004 bar. Question: go ahead to write D-014 + spawn build?

Urgency: medium — H3 build is paused on (1). Nothing else is in flight.

Resolution (Merlin, 2026-06-12):
1. **Build go-ahead given**: "Continue by building v1." → D-014 written; building FF7 v1
   (T-009), to run as EXP-010, present-then-stop per §5.
2. Item 2 (harness pick) **withdrawn by Merlin via direct edits**: question deleted from this
   file; protocol updated by him (orchestrator model → opus; ≥2-seed standing order REMOVED —
   so EXP-010 screens single-seed, replication only on promise; §10 code-ownership added).
   No methods-critic agent adopted; the code-citation hard rule stays in force via agent
   memory (feedback_ground_claims_in_code).
Note recorded at build time (detail in D-014): code-grounding surfaced one correction to the
converged design — registers do NOT persist across `generate()` steps (each forward re-expands
the learned tokens, dynamics_model.py:282; only latents carry between steps, :405), so the FF7
relay additionally needs a **param-free inference change** (carry + inject register states —
the exact interface the FF7 training rollout trains). "train_dynamics_model.py ONLY" was too
strong; "no architecture change / no new params" still holds. Same carrier, same loss, same
frozen eval. Will be re-flagged at the EXP-010 present-then-stop.
Applied to: DECISIONS.md D-014, BOARD.md (T-009 in progress), ORIENT.md, IDEAS.md (FF7 row
correction).

## ESC-006 | 2026-06-13 | OPEN — present EXP-010 (FF7 v1 screening) — present-then-stop
Context: EXP-010, the first H3 method (FF7 v1, D-014), finished overnight (both arms) while no
session was alive. k=1 and k=3, single seed, 100 ep each, frozen probe 5503e75, vs the T-004
bar (color ΔRGB < ~63 at n_occ {12,16,24}; EXP-009 baseline is at chance there). This is a §5
present-then-stop gate.

### The result (decisive read)
**H3 is supported in its first attempt — but the win is hidden-COLOR retention specifically,
not full hidden-state retention.** Both FF7 arms replace the baseline's sharp post-window cliff
(color recall → chance the instant the color-carrying prefix scrolls out of the N=8 window)
with a smooth, gentle decay that stays well below the chance floor far past the window. Against
the pre-registered T-004 bar (<63 at n_occ 12/16/24): both arms clear it at n_occ 12 and 16
(k=1 52/59, k=3 40/55) and both narrowly miss only at n_occ 24 (k=1 80, k=3 65 vs bar 63 —
k=3 by 2 points). k=3 beats k=1 at every beyond-window point, confirming the in-pass relay
rationale. No D-014 tripwire fired: window-1 register-carry inference did NOT degrade base
dynamics (FF7 ceiling/drift controls are equal-or-better than the EXP-009 baseline), and there
is no loss interference (healthy val loss, near-perfect ceiling control).

**The honest caveats, surfaced not buried:**
1. **Color only; position is at chance.** The model carries the ball's *color* through
   occlusion but not its position (pos_err ≈ chance for both arms). A register relaying a
   static attribute is a believable mechanism; integrating hidden *motion* is not happening.
2. **The secondary metric does not corroborate the headline.** latent-token MSE stays near
   chance for the FF7 arms and barely separates from its drift control — because latent-MSE is
   dominated by the at-chance position (latentMSE↔posErr r≈0.95; latentMSE↔color r≈0.7..0.8).
   This is exactly the position-confound T-004 anticipated when it pre-registered color as the
   headline and latent-MSE as merely validating. Read on latent-MSE alone this looks near-null;
   the color decomposition is what reveals the retention. I judge the headline (color) the
   correct metric here per T-004, but you should know the two disagree.
3. **Surprise (favorable):** k=1 vastly exceeded its pre-registered expectation. I predicted
   k=1's chained relay was untrained and would decay to chance by n_occ 12; instead it held far
   below chance through 16. The "untrained chained interface" reasoning was too pessimistic —
   single-frame sufficiency + param-free register-carry relays color much further than expected.

### Access points (low-friction view)
- **Headline chart (open this first):** `experiments/EXP-010/headline.png` — color recall vs
  n_occ, all 3 series + ceiling/chance/T-004-bar reference lines; baseline cliff vs FF7 gentle
  decay obvious at a glance. (Also `comparison.html` — same curves + latent-MSE panel, no deps.)
- **Frame sheets (GT top / prediction bottom):** `experiments/EXP-010/k1/sheet.png`,
  `experiments/EXP-010/k3/sheet.png`.
- **Tables + numbers:** `experiments/EXP-010/comparison.md` ; raw `k1/results.json`,
  `k3/results.json`. **Full reconciliation:** `experiments/EXP-010/NOTES.md`.
- **W&B:** k=1 https://wandb.ai/models-eberhard-karls-universit-t-t-bingen/transformer-D-dynamics/runs/82klng1c
  ; k=3 .../runs/17u810q2 (project transformer-D-dynamics).

### The question for you
(1) Do you agree with the read — FF7 v1 demonstrates genuine hidden-**color** retention well
beyond the window, a clear win over the baseline's total collapse, with position unretained?
(2) Is "color retained, position at chance, latent-MSE flat" enough for you to call this H3
*progress/support*, or do you want position retention (full hidden state) before crediting H3?
(3) Direction for next, my recommendation: this single-seed screen is promising enough to (a)
replicate the better arm (k=3) at ≥2 seeds for stability, and (b) decide whether to push the
*method* on position next (e.g. an FF7 variant that forces dynamics/motion state into the
register, not just static color) vs. first hardening the color result (longer training / the
n_occ-24 near-miss / adversarial action coverage). I lean: confirm k=3 with a second seed AND
start designing the position-carrying variant, since color-only is a partial H3 win and
position is the harder, more interesting half. Your call on whether to spend seeds first or
method-iterate first.

Urgency: blocking — per §5 I am not starting the next decision, seeds, or any FF7 variant
until you weigh in. Nothing is in flight; the 4070 is idle.

### ESC-006 RESOLVED (Merlin, 2026-06-13)
Steering (verbatim-in-substance): The important point is that **even in an OPEN rollout the
position is as bad as chance — the model did not learn to track movement in general.** Whether
this memory approach can or cannot memorize position/momentum is therefore **unproven**, because
the model can't do position even in the clear. Cause unknown — maybe the ball moves only a
little per step (but even so the model should predict it); maybe not enough temporal attention
or it comes too late (e.g. instead of "s,s,s,t,s,s,s,t" try "s,t,s,s,t,s,s,t,s"); maybe it never
learned the concept of a movable object; maybe just more training — "maybe, maybe not." The
previous experiments have the same issue, my_dynamics too, even with curtains down. **"We might
want to fix that first."** Greenlit the diagnostic-first plan: "yes go."
He did NOT give a clean verdict on the H3 color-only result itself (Q1/Q2) — the redirection to
the movement-tracking root cause supersedes it for now; H3 status stays "supported (color-only),
position open" pending EXP-011.
Applied to: DECISIONS.md D-015 (EXP-011 no-training diagnostic: confirm/localize/disambiguate
the position deficit before any architecture change or retrain), GOAL.md (H3 note + position-
tracking-blocker), ORIENT.md (rewritten), BOARD.md (EXP-011 in progress).
