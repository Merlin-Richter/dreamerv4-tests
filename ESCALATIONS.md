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

## ESC-002 | 2026-06-11 | RESOLVED — present-then-stop (EXP-008 review)
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

## ESC-004 | 2026-06-12 | RESOLVED — present EXP-009 (H2 baseline cliff) + T-004 pre-registration
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

## ESC-005 | 2026-06-12 | RESOLVED — two items awaiting Merlin before H3 build (context-reset checkpoint)
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

## ESC-006 | 2026-06-13 | RESOLVED — present EXP-010 (FF7 v1 screening) — present-then-stop
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

## ESC-007 | 2026-06-13 | RESOLVED — present EXP-011 (position-deficit diagnostic) — present-then-stop
Context: EXP-011 (D-015, no training) ran to answer your ESC-006 redirect — did the model never
learn motion (a), or learn it and desync in open loop (b), and does the deficit live in the
tokenizer C or the dynamics D. This is a §5 present-then-stop; it is also high-surprise +
raises a methodological issue, so flagging hard.

### The result (decisive read)
Your worry was right about *one* model and reframed for the rest. Three findings:
1. **The tokenizer is fine — position is fully encoded in the latents.** A linear probe reads
   ball (x,y) off the frozen tokenizer latents at R²=0.96 (median 2.7px). So the deficit is in
   the dynamics model D, not the encoder C. (This was my D-015 tripwire; it did NOT fire — the
   info reaches D, so registers/FF7 on D can in principle carry position.)
2. **my_dynamics is genuinely a weak motion model — failure (a).** With *perfect* GT context,
   its 1-step position prediction is 4.5px, WORSE than just freezing the ball (3.2px). It never
   learned good motion. (Consistent with EXP-009's own ceiling 5.6px — no bug; the "1.1px" I'd
   cited before was the detector on GT frames, not the model.) So the "position at chance" partly
   traces to the **baseline being undertrained/weak**, exactly as you suspected the model "can't
   even do it in general."
3. **But motion IS learnable here, and the FF7 checkpoints already track it well — failure (b),
   not (a).** FF7 1-step is ~1.0px (4.5× better than my_dynamics, 3× better than freezing).
   Open-loop, ff7_k3 tracks the moving ball to 4.6px@h4, 10px@h8, 14.8px@h12, only saturating
   near chance by h≈16+ — gradual open-loop compounding, i.e. chaos, not inability to model
   motion. **EXP-010's "position at chance" was misleading:** it measured position only at the
   reveal frame, which for the drift control sits at horizon = n_occ, so the n_occ 12/16/24
   points were always already in the saturated regime. The horizon-resolved curve shows FF7
   does track position; the snapshot hid it.

**What this means for the H3 position question:** it is NOT doomed by a base-capability wall.
Position is in the latents; a trained model tracks motion 1-step to ~1px and open-loop for ~12
steps. The reason occluded position hits chance is that **dead-reckoning a bouncing ball through
occlusion with zero feedback is chaotic** — one bounce-timing error desyncs the exact GT
trajectory. Color (static) survives occlusion; exact position (chaotic) can't be expected to
under an open-loop GT-matched metric. That's a *measurement* problem (we'd need a closed-loop or
distributional position metric), not proof the memory approach fails on position.

### The catch I have to flag (methodological)
I cannot tell from this diagnostic whether FF7's far-better dynamics come from the **FF7 loss**
or simply from the FF7 runs being **trained 100 epochs while my_dynamics (old baseline) was
trained less/differently**. Provenance of my_dynamics is older/approximate. Consequence:
**EXP-009 (H2 baseline = my_dynamics) and EXP-010 (FF7, fresh 100-ep) are not training-matched.**
The color-memory conclusion still holds (the sliding-window cliff is architectural — a better-
trained vanilla model still can't see past its window), but a clean H3 comparison wants a
budget-matched vanilla baseline, and my_dynamics should probably be retired as the baseline.

### Access points (low-friction view)
- **Headline chart:** `experiments/EXP-011/headline.png` — open-loop pos_err vs horizon for all
  3 models + copy-last + chance; title carries the two killer numbers (1-step 4.5 vs 1.0px,
  latent probe R²=0.96).
- Numbers: `experiments/EXP-011/results.json`. Full reconciliation: `experiments/EXP-011/NOTES.md`.

### The question for you
(1) Do you agree with the reframing — base model *can* track motion (FF7 ~1px 1-step, ~12 steps
open-loop); occluded position-at-chance is dead-reckoning chaos + a weak old baseline, not a wall?
(2) The methodological fork — my recommendation: **train a budget-matched vanilla baseline (same
100 ep / data as FF7) to (i) retire my_dynamics, (ii) cleanly attribute the dynamics improvement,
and (iii) re-anchor H2/H3 comparisons.** This is the first cluster-worthy / overnight-worthy run;
alternatively keep screening locally. Agree?
(3) For the H3 *position* question specifically: do you want to (a) switch the position metric to
closed-loop / distributional (measure memory, not chaos), (b) treat color-only as the H3 result
and move on, or (c) design an FF7 variant aimed at position before deciding? My lean: (2) first
(we need a clean baseline regardless), then a closed-loop position metric, then judge position.

Urgency: blocking — per §5 I am not starting the next decision (baseline retrain, metric change,
or any variant) until you weigh in. Nothing is in flight; the 4070 is idle.

### ESC-007 RESOLVED (Merlin, 2026-06-13)
- **Agrees with the reframing** ("You seem to be right"): tested ff7_k3 interactively in
  play_dynamics_checkpoint.py — motion looks reasonable. So the base model CAN track motion;
  occluded-position-at-chance is not a base-capability wall.
- **New task (inserted):** play_dynamics_checkpoint.py appeared to NOT carry the FF7 register/
  memory tokens forward — after ~4 curtain frames it produced a random image. Asked to check
  whether the script handles what FF7 needs and add it if missing. → **T-010 (DONE):** confirmed
  the viewer used the vanilla fixed-4-window path (no register carry); refactored the relay into
  reusable DynamicsModel.memory_rollout_init/step (generate_memory now a thin loop over them) and
  drove them in the viewer for use_register_memory checkpoints. Verified: 5/5 FF7 smokes, probe
  dry-run matches EXP-010, headless reveal test color dRGB 9.9 (memory) vs 64.4 (vanilla off).
- **"Other than that, I agree!"** → taken as agreeing with the ESC-007 recommendations:
  (Q2) train a budget-matched vanilla baseline — **D-016 / EXP-012 launched** (local 4070, vanilla
  --ff7 0 --fresh at the exact EXP-010 budget; train+probe+EXP-011-rerun, present-then-stop);
  (Q3) baseline first, then a closed-loop/distributional position metric, then judge position.
Note: EXP-010 had no saved config.yaml (provenance gap); EXP-012 fixes this with committed
config.yaml + run.sh. CUDA is only visible via venv/Scripts/python.exe here (bare python is
CPU-only) — recorded in HOWTO so future runs use the venv; EXP-011 ran on CPU but reconciles
with EXP-010's GPU numbers, so its conclusions stand.
Applied to: DECISIONS.md D-016, dynamics_model.py + play_dynamics_checkpoint.py (T-010),
CLAUDE.md, experiments/EXP-012/*, EXPERIMENTS.md, BOARD.md, ORIENT.md.

## ESC-008 | 2026-06-13 | RESOLVED — present EXP-012 (budget-matched vanilla baseline) — present-then-stop
Context: EXP-012 (D-016) is the clean control ESC-007 agreed to: a fresh vanilla dynamics model
(--ff7 0 --fresh) at the EXACT EXP-010 FF7 budget (occluded, 100 ep, bs32, lr3e-4, seed0; val loss
0.0066 = FF7's 0.0065), to retire my_dynamics and break the FF7-loss-vs-training-budget confound
EXP-011 surfaced. Frozen probe 5503e75 + the EXP-011 motion diagnostic rerun across all 4 models.
This is a §5 present-then-stop gate.

### The result (decisive read)
**The confound is resolved: FF7's wins are the method, not the training budget — on BOTH axes.**
1. **Color / H2 — stands, on a clean baseline.** The budget-matched vanilla reproduces the sharp
   architectural color cliff: hidden-color ΔRGB ≈ ceiling (15) while the prefix is inside the N=8
   window, then jumps to ≈ chance (98→108) the instant it scrolls out (n_occ 6→7), and stays at
   chance (105–110) through n_occ 24. It does NOT reproduce FF7's beyond-window retention (FF7
   40–65, below the T-004 bar of 63 through n_occ 16). The D-016 tripwire (vanilla beyond-window
   color < 63) did NOT fire. → EXP-010's color memory is the FF7 register relay, not budget.
   my_dynamics retired; vanilla_s0 is the H2/H3 baseline.
2. **Motion — the FF7 loss, not budget, makes a good 1-step dynamics model.** Teacher-forced
   1-step pos_err: vanilla_s0 **4.66px** ≈ my_dynamics 4.51px ≫ ff7_k1 1.02 / ff7_k3 0.96px. Both
   vanillas are *worse* than freezing the ball (copy-last 3.19px); both FF7 arms beat it ~3×.
   Open-loop, ff7_k3 tracks far longest (8.5px@h8 vs vanilla 18.5). Latent→xy probe R²=0.96 (C
   encodes position; reproduced).

**The honest correction (mild surprise — one sub-prediction refuted):** I predicted (D-016, from
EXP-011) that the budget-matched vanilla would substantially beat my_dynamics at motion — that
my_dynamics's weakness was "partly undertraining." **Refuted:** two independently-trained vanillas
land at the same ~4.5px. Motion weakness is intrinsic to the vanilla setup here, not undertraining;
the FF7 loss is what fixes it. (Color was exactly as predicted. No halt-tripwire fired — vanilla
≈ my_dynamics is consistent, not a seed/data bug; a 2nd vanilla seed would confirm if you want it.)

**Bonus finding worth your eye:** FF7 sharpens the *base* 1-step dynamics ~4.6×, not just the
memory relay. That's a bigger claim about the single-timestep-sufficiency objective than "it
carries static color" — it looks like a dynamics regularizer. (The diagnostic's FF7 1-step uses
the relay-inference path, so loss-vs-relay isn't fully disentangled yet.)

### Access points (low-friction view)
- **Color headline (open first):** `experiments/EXP-012/headline_color.png` — vanilla cliff →
  chance vs FF7 sub-bar decay, all on the identical frozen probe, with ceiling/chance/63-bar lines.
- **Motion headline:** `experiments/EXP-012/headline_motion.png` — open-loop pos_err vs horizon
  (4 models) + the 1-step teacher-forced bar inset (the attribution number) + latent-probe note.
- **Probe frame sheet:** `experiments/EXP-012/sheet.png`. Numbers: `results.json` (probe) +
  `diagnostic.json` (motion). Full reconciliation: `experiments/EXP-012/NOTES.md`.
- **W&B:** exp012-vanilla-s0 (project transformer-D-dynamics).

### The question for you
1. Agree H2 is now cleanly anchored on a training-matched baseline and my_dynamics is retired?
2. Agree the confound is resolved — FF7's color AND motion wins are the method, not budget
   (so EXP-009/EXP-010 conclusions are retroactively trustworthy)?
3. This closes the ESC-007 baseline action. The agreed Q3 path was: baseline → closed-loop/
   distributional position metric → judge position. My recommendation: proceed to design that
   position metric next (it's cheap and unblocks honest position claims), and treat the
   **sequential stop-grad register-relay training** idea we worked out today (IDEAS.md) as the
   leading H3 *position* method to try once we can measure position. These are complementary —
   the metric tells us how to MEASURE memory, the relay is a METHOD to improve it. Agree, or
   redirect (e.g. relay-first, or a 2nd vanilla seed for the motion claim first)?

Urgency: blocking — per §5 I am not starting the next decision (position metric, relay method, or
seed) until you weigh in. Nothing is in flight; the 4070 is idle. (Independent KV-cache work
T-008/D-017 was committed during the EXP-012 wait; does not touch this result.)

### ESC-008 RESOLVED (Merlin, 2026-06-13)
1. **Agreed** — H2 cleanly anchored on the budget-matched baseline; `my_dynamics` retired; `vanilla_s0`
   is the H2/H3 baseline.
2. **Agreed** — the confound is resolved; FF7's color AND motion wins are the method, not budget;
   EXP-009/EXP-010 conclusions are retroactively trustworthy.
3. **Redirect on the position metric / interpretation (verbatim-in-substance):** the way position is
   currently measured (open-loop GT-matched error) is bad — a model that predicts "ball is center"
   always gets high loss, AND a model that predicts accurately but suffers early butterfly-effect
   desync also scores badly. A better objective is to measure how well the predicted **position AND
   velocity stay CONSISTENT over the occluded timesteps**: run the curtain down and, at every
   timestep, check "what would the model predict if we reveal now" → read out (x,y,dx,dy), and
   compare across consecutive steps (e.g. step t vs t−1) for consistency — "or something like that,
   or something else idk, but the currently proposed [open-loop GT-matched] idea is bad."
   → The previously-proposed "closed-loop/distributional position metric" is superseded by this
   **belief-self-consistency** direction. Design to be converged with Merlin + checked by the new
   `critical-claim-verifier` agent BEFORE build+freeze (§8: the metric is the spine — a logged
   decision; pre-registration honesty = lock before any H3 position method runs). → D-018.
**New tooling note (Merlin):** the `critical-claim-verifier` agent is committed. Use it when
developing plans for difficult code, or difficult architecture/objective ideas, where an unbiased
independent opinion is wanted. (Not for routine work.)
Applied to: DECISIONS.md D-018, BOARD.md (metric task refined), ORIENT.md (rewritten).

## ESC-009 | 2026-06-13 | RESOLVED — present EXP-013 (position-memory metric: built, validated, applied) — present-then-stop
Context: the D-018 / T-011 position-memory consistency metric you redirected us to (ESC-008) is built
(`src/probe/position_consistency.py`), verifier-audited (V-T011, 5 fixes folded), framing-locked
(anchored-physical-coherence), validated, and applied to the H3 baseline (vanilla_s0) + FF7 k1/k3.
This is a §5 present-then-stop. (Note: a parallel orchestrator is running EXP-014, the loss-vs-relay-vs-
window 1-step disentangle, D-019 — independent of this; I stayed hands-off it and committed path-scoped.)

### Instrument validated (the weak result below is the MODELS, not the metric)
- Synthetic calibration reproduces the V-T011 audit exactly: billiard residual GT 0.77 (floor), F2 0.79
  (passes), hallucination caught (onset 22px), forgetting surrogates 4.9–10.8 (≫floor). Speed-fixed load-bearing.
- Readout faithfulness confirmed: FF7's belief at the FIRST blind step = 1.9px (seed) / 4.0px (mean) —
  reproduces its known ~1px short-horizon skill. found_rate 1.00.

### The result (decisive read) — blind position memory is near-absent
Per-k belief-vs-GT err (mean/20 seeds); copy-last = freeze ball at last-seen position:
```
            k=1  k=2  k=3  k=4  k=5   k=12
copy-last   5.7  8.5 11.1 13.8 16.3  30.7
vanilla_s0  5.7  8.6 12.3 14.6 18.5  20.6
ff7_k1      5.3  7.3 10.8 12.0 15.8  21.3
ff7_k3      4.0  7.9 11.9 14.5 19.4  24.5
```
1. **vanilla_s0 ≈ copy-last** (k1–4): it freezes the ball — ZERO motion propagation through blind
   occlusion (residual 14.85 > even the frozen surrogate 6.6: a weak open-loop model wanders, worse
   than freezing).
2. **FF7 retains only marginally more than freezing:** ff7_k1 ~1–3px below copy-last at every horizon
   (small but real); ff7_k3 best at the first blind step (4.0 vs 5.7) then decays to ≈copy-last. Both
   become physically incoherent (belief teleports ≫ speed) by k≈5. Coherence horizon ~1 step.
3. **Corrects EXP-011:** its "FF7 tracks ~12 steps" was curtain-UP (the model sees its own generated
   ball — visual feedback). Under TRUE occlusion (no feedback) the blind horizon is ~1–4 steps.
4. **H3 story:** the register relay carries STATIC color indefinitely (EXP-010) but NOT dynamic
   position/velocity through blind occlusion. The FF7>copy-last position signal (esp. k1) is real but tiny.

### Honest caveats
- Single 16-step billiard residual is a poor headline (averages coherent-early + incoherent-late);
  the per-k curve + copy-last + coherence-horizon is the right summary. Proposing coherence-horizon as
  the frozen headline (your call). The residual still ranks correctly (ff7_k1<ff7_k3<vanilla).
- Late-tail artifact (your flag, observed): k>5 vanilla dips BELOW copy-last (k12: 20.6<30.7) —
  wandering prediction coincidentally re-approaches GT as copy-last drifts away. Only k≤~5 trustworthy.

### Access points
- `experiments/EXP-013/headline_position.png` (the curve), `per_k_curve.json`, per-model `*_posmem.json`,
  `NOTES.md` (full read), `experiments/verify-T011-scorer/prod_calibrate.json` (instrument validation).

### The question for you
1. Agree with the read — blind position memory is near-absent (vanilla=copy-last; FF7 marginally
   better, esp k1); the EXP-011 optimism was assisted curtain-up tracking, not blind memory?
2. Freeze the metric? I propose freezing the readout+scorer at this commit with the **coherence-horizon
   + per-k curve** as the headline (not the single billiard residual). Adjust if you'd summarize differently.
3. Direction — my recommendation: this is exactly the motivation for the **sequential stop-grad
   register-relay training** (IDEAS.md): the FF7 single-timestep-sufficiency loss teaches color-carry
   but not motion-carry; a relay trained to carry dynamic state is the natural next method, now that
   blind position is measurable. Proceed to design/build it, or do you want to iterate the metric
   summary first / something else? (Also: EXP-014's loss-vs-relay verdict may inform the method — worth
   waiting for it?)

Urgency: blocking — per §5 I am not starting the relay method or freezing until you weigh in. Nothing of
mine is in flight; 4070 free (modulo the parallel EXP-014).

### ESC-009 RESOLVED (Merlin, 2026-06-13)
Steering (verbatim-in-substance): "I am not sure whether the position metric is a strong evaluation
metric the way it was coded. But I don't want to walk in circles. Just resolve the escalations as
whatever. I want to continue." → His call:
- **Reservation recorded, not litigated.** Merlin is not confident the EXP-013 position-memory
  consistency metric, *as coded*, is a strong evaluation instrument. He explicitly does NOT want to
  re-iterate or re-design it now (no circles).
- **Do NOT ceremonially freeze it as the H3 position spine.** It is "built, of uncertain strength."
  The EXP-013 read (blind position memory near-absent: vanilla_s0 ≈ copy-last; FF7 only marginally
  better, esp. k1) stands as the current best read but is NOT treated as a hard pre-registered gate
  that future methods must clear. If a position method later needs a yardstick, the metric's strength
  is revisited then — not now.
- **Direction: continue.** Next work = **KV cache for sliding-window rollouts** (easily verifiable),
  as preparation for rollout training (the eventual sequential register-relay method). → D-020.
Applied to: GOAL.md (H3 position note — metric of uncertain strength, not frozen), DECISIONS.md D-020,
BOARD.md, ORIENT.md. EXP-013 left as-is (result stands; not a frozen gate).

## ESC-010 | 2026-06-13 | RESOLVED — present EXP-014 (disentangle FF7 base-dynamics gain) — present-then-stop
Context: Independent thread (D-019), parallel to the D-018/ESC-009 position-metric work — chosen to NOT
touch the metric spine. Resolves the open ORIENT worry #4 / EXP-012 "bonus finding": FF7 sharpens 1-step
teacher-forced pos_err ~4.6x (vanilla_s0 4.66 >> ff7 ~1.0px), but that ~1px was produced through the
register-RELAY inference path (generate() dispatches use_register_memory=True -> generate_memory(),
a window-1 relay), NOT the <=7-frame windowed attention the vanilla_s0 number used — so the gain conflated
the FF7 loss, the relay, and window size. EXP-014 (analysis-only, no training, existing checkpoints) runs
each model's 1-step teacher-forced prediction through BOTH inference paths on the IDENTICAL GT window.
This is a section-5 present-then-stop gate.

### The result (decisive read)
**FF7's base-dynamics improvement is the LOSS, not the relay inference.** FF7 weights run through the
plain windowed path with *no relay at all* (learned-init scratch registers — the exact forward FF7's own
main diffusion loss uses) already hit **1.04 px** (k3) / 1.63 px (k1), versus vanilla_s0's **4.73 px** —
a 4.5x / 2.9x improvement in *windowed* 1-step dynamics from the single-timestep-sufficiency objective
alone. The register relay is a *secondary, arm-dependent* contributor: it barely moves the well-trained
k3 (1.04->0.99) and closes the residual gap on the weaker-trained k1 (1.63->1.00). And the relay *requires*
the FF7 loss — vanilla_s0 forced through the relay is *worse* (5.34 px > its own 4.73), so the window-1 +
carried-register inference hurts weights not trained for it. Net: the EXP-012 "bonus finding" stands and
sharpens — **FF7's single-timestep-sufficiency loss is a genuine dynamics regularizer on the windowed
weights, independent of any memory-relay inference trick.** All three D-019 tripwires checked clear; the
relay-path column reproduces EXP-011/012's FF7 numbers and the vanilla path reproduces vanilla_s0, so the
cross-path differences are real (harness validated, not drift).

**Complement to ESC-009/EXP-013:** that experiment found position memory through *true blind occlusion*
is near-absent (vanilla ~ copy-last; FF7 only marginally better). Together: the FF7 loss buys a clean
1-step dynamics model, but that 1-step accuracy does NOT translate into dynamic *position* memory through
occlusion — FF7 relays static color, not motion. The two same-day findings are consistent and reinforce.

### Access points (low-friction view)
- **Headline chart (open first):** experiments/EXP-014/headline.png — grouped bars, vanilla vs relay
  path per model, with copy-last (3.19px) and vanilla_s0-windowed (4.73px) reference lines. The FF7
  no-relay bars at ~1px vs vanilla_s0's 4.73px bar is the whole story at a glance.
- Numbers: experiments/EXP-014/results.json. Full reconciliation: experiments/EXP-014/NOTES.md.
  Script: experiments/EXP-014/disentangle.py (reuses EXP-011 diagnostic infra + frozen probe 5503e75).

### The question for you
1. Agree with the read — FF7's 1-step base-dynamics gain is the LOSS (a dynamics regularizer), with the
   relay a small secondary effect, so the EXP-012 bonus claim is confirmed (not an inference artifact)?
2. Does this change how you want to frame/pursue the relay method? It says: expect most 1-step accuracy
   from the FF7 objective itself; the relay's job is *retention* (color beyond-window, EXP-010), not raw
   1-step accuracy. (ESC-009 explicitly flagged this verdict as possibly informing the relay-method call.)
3. Anything to follow up: a 2nd-seed check of the loss-regularizer effect, or fold this into the eventual
   EXP-012/H3 writeup as a sub-result?

Urgency: non-blocking for the metric track (independent); per section 5 I am not starting any follow-on
from this result until you weigh in. Nothing of mine is in flight; 4070 free.
Coordination note: this thread kept strict isolation from the metric work — no edits to
src/probe/position_consistency.py / EXP-013, path-scoped commits only (never git add -A), append-only
state edits with my own IDs (D-019, EXP-014, ESC-010), ORIENT/BOARD left to the metric orchestrator.

### ESC-010 RESOLVED (Merlin, 2026-06-13)
Resolved together with ESC-009 ("just resolve the escalations as whatever. I want to continue"). The
read **stands and is accepted**: FF7's ~4.6× 1-step base-dynamics gain is the LOSS (a dynamics
regularizer on the windowed weights), with the register relay a small, arm-dependent secondary effect
that requires the FF7 loss. No follow-up (2nd-seed check) requested now; fold into the eventual H3
writeup as a sub-result when we get there. The relay's job is *retention* (beyond-window color), not raw
1-step accuracy — carried forward as context for the upcoming rollout-training method. Move on. → D-020.

## ESC-011 | 2026-06-13 | RESOLVED — present EXP-015 (rollout KV cache perf tool) — present-then-stop
Context: You asked for a basic, reusable perf tool for the rollout KV cache (T-012): cached vs no-cache,
GPU, with a batch dimension (training-relevant), reporting rollout-step throughput, memory, and where
time goes, across context-window sizes. EXP-015 (D-022): `experiments/EXP-015/perf_rollout.py`,
`generate_streaming` (cache) vs `generate_windowed` (matched uncached twin), B=32, N∈{8,16,32,64},
~8s/config. §5 present-then-stop.

### The result (decisive read)
**The cache makes continuous-rollout throughput INDEPENDENT of context length, and wins on memory too.**
Cached holds flat at **~28 steps/s (~900 frames/s at B=32, ~35 ms/step) from N=8 to N=64**, while the
uncached path degrades 21→5.7 steps/s (47→175 ms/step) because it re-encodes the whole window every
step. Speedup widens with context: **1.33× (N=8) → 4.79× (N=64).** Memory: cached grows only with the
window (113→218 MB alloc) and stays far below windowed, which balloons to 496 MB alloc / **4708 MB
reserved** at N=64 (large transient window tensors → allocator churn). Profiler (N=32): the cache cuts
total CUDA work 3.7× (889 vs 3268 ms / 20 steps); both paths are ~56% compute / ~40% memory-bound
(elementwise/cat/norm — RMSNorm is the biggest single memory-bound op). The ~40% "memory" split is a
profiler proxy for time-waiting-on-memory; exact HBM-stall % would need Nsight Compute.
**Mild favorable surprise:** I predicted the persistent cache would cost MORE memory; it costs LESS
(opposite). No D-022 tripwires fired.

### Access points
- Plot (open first): `experiments/EXP-015/perf.png` (throughput / ms-per-step / peak-MB vs N, both methods).
- Numbers: `experiments/EXP-015/results.json`. Full reconciliation: `experiments/EXP-015/NOTES.md`.
- Rerun at other settings: `venv/Scripts/python.exe experiments/EXP-015/perf_rollout.py --batch B --windows ... --budget S`.

### The question for you
1. Is this the basic perf info you wanted, or do you want another axis (e.g. a batch-size sweep at fixed
   N, since training throughput vs B is the natural next cut — the tool takes `--batch`)?
2. Anything to read differently — e.g. the ~40% memory-bound split, or the windowed reserved-memory
   blowup (allocator churn) worth a deeper look?
Urgency: blocking per §5 — not starting follow-ups until you weigh in. Nothing in flight; 4070 idle.

### ESC-011 RESOLVED (Merlin, 2026-06-13)
Tool accepted; took the offered next cut. Steering (verbatim-in-substance): "run the perf again but with
a significantly higher batch-size. I want to see if we get more speedup as we do more parallelism — try
to be close to the GPU memory limit for each approach and compare their steps/s." → A batch-limit
parallelism sweep: push batch toward each method's OWN memory ceiling (cached uses less memory → fits a
larger batch) at a fixed context window, and measure whether the cached-vs-windowed speedup grows with
parallelism. → D-023 / EXP-016 (local 4070), present-then-stop per §5.
Applied to: DECISIONS.md D-023, EXPERIMENTS.md (EXP-016), BOARD.md, ORIENT.md.

## ESC-012 | 2026-06-13 | RESOLVED — present EXP-016 (batch-limit parallelism sweep) — present-then-stop
Context: the batch cut you asked for (ESC-011) — fixed N=32, push batch toward each method's OWN VRAM
ceiling on the 8 GB 4070, cached (`generate_streaming`) vs windowed (`generate_windowed`), does the
speedup grow with parallelism. NOTE on the runs you saw cancelled: the original tool ran 4 rollout passes
per config and measured from a near-empty window, so it was far too slow, and at batch ≥1024 it crossed
8 GB into WDDM system-RAM fallback (ms→minutes — the 10-min "stuck on windowed 2048"). Both fixed (see
"timing fix" below); this read is from the clean bounded run.

### The result (decisive read)
**Yes — more parallelism gives MORE speedup, not less, because only the cached path scales; the uncached
path is throughput-flat.** At N=32 the cached/windowed steps/s ratio rises MONOTONICALLY with batch:
**5.85× (B=32) → 10.2× (64) → 13.3× (128) → 13.5× (256) → 14.0× (512).** Cached end-to-end throughput
climbs 731→1427 frames/s as batch goes 32→512; windowed is **flat at ~105–126 frames/s for every batch**
— it is saturated on re-encoding the full 31-frame window each step, so extra parallelism buys it nothing.
Mechanism: windowed's wasted per-step recompute scales with batch while cached does O(1) work/step, so the
gap widens as you parallelize.
**On "close to the memory limit for each approach":** at N=32 the asymmetry FLIPS vs EXP-015's N=64 result.
Here cached uses MORE memory per batch (it holds a persistent K/V cache for 31 frames × all layers/heads),
so cached hits its VRAM ceiling SOONER — cached maxes at **B=512 (4232 MB, 52% VRAM)**, windowed reaches
**B=1024 (6716 MB, 82%)**. So windowed fits 2× the batch — but it doesn't matter: cached at B=512 (1427
frames/s) still delivers **13.5× the end-to-end throughput** of windowed at B=1024 (105 frames/s). Whether
cached or windowed fits the bigger batch is N-dependent (windowed's transient ballooned at N=64; cached's
cache dominates at N=32).
Two of my D-023 predictions were wrong, both surfaced: I expected the ratio to COMPRESS toward 1 at high
batch (it GROWS) and expected cached to fit a BIGGER batch (it fits a smaller one at N=32). Surprise: mild,
favorable — the cache looks even better than predicted on speed.

### The honest caveat (measurement stability)
Absolute steps/s drifts ~30% run-to-run on this laptop GPU (thermal/clock throttling across back-to-back
sweeps — an earlier identical run read cached B=512 at 2090 vs 1427 frames/s here). The cached-vs-windowed
RATIO and the qualitative shapes (cached scales, windowed flat) are stable, because the two methods are
timed back-to-back under the same thermal state. Read the speedup ratio + the shapes as the result, not the
absolute frames/s. Memory numbers are stable.

### Access points
- **Plot (open first):** `experiments/EXP-016/perf_batch.png` — L: frames/s vs batch (cached scales,
  windowed flat); M: steps/s vs batch with the green speedup line 5.9×→14×; R: peak reserved MB vs batch
  with the 8 GB GPU line + each method's ceiling.
- Numbers: `experiments/EXP-016/results.json`. Full reconciliation: `experiments/EXP-016/NOTES.md`.
- Rerun: `venv/Scripts/python.exe -u experiments/EXP-015/perf_rollout.py --batch-sweep --sweep-window N
  --batches ... --budget 2 --outdir experiments/EXP-016`.

### The timing fix you asked for (so the next run isn't slow)
Root cause was twofold: (a) 4 rollout passes/config + measuring from a near-empty window (long run needed
to reach steady state, exploding at big batch); (b) launching configs past 8 GB → WDDM sysmem-fallback
thrash (the 10-min hang). Fixes: measure from a PRE-FILLED window (context = N−1) so every step is
steady-state → a short budget-capped pass suffices (warmup 4 + calibrate 3 + 8–64 measured steps, peak
memory captured in the same pass — no separate memory pass); and a **predictive VRAM guard** that
extrapolates reserved-MB and STOPS escalating a method before the next batch would cross 0.92×VRAM (so a
thrashing config is never launched). Also: `-u` + writing straight to file (no `tail` pipe) so progress
streams live. A clean full sweep now runs in a few minutes and never hangs.

### The question for you
1. Does this answer it — speedup grows with parallelism (→14× at B=512) because cached scales and windowed
   is throughput-flat; and at each method's own VRAM ceiling cached wins 13.5× end-to-end despite fitting
   half the batch at N=32?
2. Want another cut? Natural follow-ups: the same batch sweep at a DIFFERENT N (e.g. N=64, where the
   memory asymmetry flips and cached may fit the bigger batch), or a quick 2–3× repeat at one batch to pin
   the ±30% thermal variance with error bars. I lean: not needed unless you want the N=64 contrast — the
   parallelism question is answered.
Urgency: blocking per §5 — not starting any follow-up until you weigh in. Nothing in flight; 4070 idle.

### ESC-012 RESOLVED (Merlin, 2026-06-13)
"Yes these are strong results! Accepted." The EXP-016 read stands: more parallelism → more speedup
(5.9×→14×) because only the cached path scales while windowed is throughput-flat; at each method's own
VRAM ceiling cached wins 13.5× end-to-end despite being more memory-hungry at N=32. No follow-up cut
requested (N=64 contrast / error bars not asked for). Directed: **clean up + propose the next direction.**
→ The KV-cache rollout substrate (T-008 → T-012 → EXP-015/016) is now complete and characterized; this
closes the efficiency subobjective. Next = the H3-position method it was built for (see ORIENT / the
proposal). Applied to: ORIENT.md, BOARD.md, ESCALATIONS (header markers tidied).

## ESC-013 | 2026-06-13 | OPEN — FF9 design audited (REFUTED + fixed); A/B reframed before build
Context: you chose option B (memory-token split + FF9 memory-only sufficiency), color-first. I wrote the
design note (`tasks/T-013-plan.md`) and — per §4 (novel objective + arch change) — ran it past the
`critical-claim-verifier` BEFORE recording a build decision. The verdict changes the strategic picture, so
I'm stopping to check with you rather than rubber-stamping a build.

### Verifier verdict (V-T013): REFUTED as specified — but with a clean fix + a strategic catch
1. **Loss shortcut (fixable, fix folded into the plan).** FF9 copied FF7's successor setup: successor
   frames carry their OWN real latents noised at τ~Uniform, loss ramp-weighted `0.9τ+0.1`. So the dominant
   ramp-favored (high-τ) part is solvable by locally denoising each successor — **memory is non-load-bearing
   except in the low-τ tail the ramp down-weights.** Empirical probe: memory's max benefit ~61% of the loss
   but concentrated at τ≈0.1 (+0.177) and ≈0 at τ≈0.9 (+0.002). Withholding the latent does NOT *force*
   full-state memory under the inherited ramp. **Fix:** clamp τ LOW on FF9 successors / flatten-invert the
   FF9 ramp; strongest — supervise frame 1 at τ≈0 (pure noise) so the whole target must come from memory.
   (Done in the revised §3.)
2. **Single-hop credit limit (NOT fixable in v1 — the strategic catch).** Even with the τ fix, FF9 v1's
   TBPTT-1 gradient trains read + 1-hop write but NOT preserve-across-N-hops. Prediction: **FF9 v1 reproduces
   FF7's split** — static COLOR survives, dynamic POSITION and beyond-window *depth* do not improve over FF7.
   Depth/position need the **sequential relay = option A**, the one you deprioritized. So the verifier says
   **A and B are COMPLEMENTARY, not alternatives:** B fixes *what* memory should encode (full state, via the
   objective + τ fix); A fixes *how* memory learns to preserve across hops (credit assignment). Neither alone
   is predicted to deliver depth/position.
Confirmed sound: no main-loss corruption, the absent-latent placeholder is fair, env deterministic (loss
well-posed). Artifacts: `experiments/verify-T013/`, EXPERIMENTS row V-T013.

### Where this leaves us — 3 coherent paths (I recommend P1)
- **P1 (recommended): build FF9 v1 WITH the low-τ fix and run it color-first** (single seed, 100 ep,
  n_occ {12,16,24,32,48}, vs FF7 + vanilla_s0). Cheapest informative step that tests YOUR chosen direction
  (B) with the verifier's fix. Framed as a diagnostic: a *win* (flatter color than FF7) means full-state +
  low-τ buys depth single-hop; an *informative null* (≈FF7) empirically confirms the credit limit is the
  blocker → clean green-light for P3. Either way the decision advances. ~one overnight run on the 4070.
- **P2: isolate the low-τ fix on the EXISTING FF7 first** (tiny change, no new architecture) — does
  stronger low-τ successor pressure alone improve FF7's color depth? Cheapest possible, separates the
  loss-fix from the memory-token change — but doesn't test the full-state-memory idea you actually want.
- **P3: go straight to FF9-objective + sequential relay (A+B combined)** — the verifier implies this is what
  depth/position actually need. Biggest build (sequential relay training with the KV cache in the loop),
  highest risk/reward; skips the isolation step so a failure is harder to localize.

**My recommendation: P1.** It's the minimal step that honours your B choice + the verifier's fix, and its
result (win or informative null) is exactly what decides whether we must escalate to P3 (A+B). The strongest
counter: the verifier's analysis already *predicts* P1≈FF7 on depth, so P1 may spend an overnight run
confirming an analytic prediction — if you find that argument decisive, jump to P3. I judge the empirical
check cheap insurance (and color *might* improve single-hop since it's static), but it's your call since it
touches the A/B framing you set.

### The question for you
Which path — P1 (recommended), P2, or P3 — and do you accept the low-τ fix as folded in? On your answer I'll
record D-024 and either build (P1/P2) or write the A+B relay design for a second verifier pass (P3).
Urgency: blocking per §4 (read verdict before deciding) + §7 (verdict bears on the A/B choice you set).
Nothing in flight; 4070 idle. I will NOT record D-024 or start building until you weigh in.

## ESC-014 | 2026-06-14 | OPEN — relay-training credit design (V-T014 REFUTES pure detach) — present-then-stop
Context: you specified Mode B (op-3 relay) with the memory carry DETACHED ("can and should be detached"). I
wrote it up (T-014) and ran it past critical-claim-verifier, which built a decisive synthetic probe. This
is a §5 present-then-stop / §7 escalation: the result is surprising AND reopens a design choice you set.

### The result (decisive read)
**Detached carry does NOT give a stable deep relay — it preserves state only up to the trained rollout
depth, then drifts to chance.** Synthetic relay (GRU writer, a hop-0 secret never re-supplied, per-step
recover loss; train depth 32, eval to 200), recovery MSE (chance ≈0.98):
```
depth   no_relay  detached(ModeB)  tbptt1   bptt
 ≤31      0.98       ~0.0003        0.0004   0.0001   <- in-window: detach looks PERFECT (= BPTT)
 100      0.98        0.149         0.077    0.0018
 199      0.98        1.08          0.451    0.0135   <- past train depth: detached ≈ chance; BPTT flat
```
deep-avg(100/150/199): detached **0.587** vs BPTT **0.007** (84× worse) vs tbptt1 0.255. Detached drift
d199/d16 = 3589×. **The trap:** within-window FF9 loss→0 *green-lights* detached, but it's blind to the
drift that happens exactly in the deep regime op-3 exists to serve. Mechanism (your Bellman analogy, now
qualified): the detached carrier is a *consistency* fixed point with **no per-step content anchor** (unlike
Q-learning's observed reward), so it's free to slowly rotate/shrink its code while every step's loss stays
satisfied. **Only BPTT-through-time extrapolates; 1 hop (tbptt-1) only partially helps.**
Caveat: synthetic GRU + STATIC secret — proves the *credit mechanism* fails (architecture-general), not a
production hop-count; the real DYNAMIC-state relay is strictly harder (no copy attractor).

### The reconciliation (so we're precise about what's refuted)
Refuted = "detach gives a relay sufficient across many hops *without training that deep*." NOT refuted =
"train the per-step loss across the full N-step rollout, detached, and it holds within that N." Your
"~200 steps" reading IS the latter and is viable to depth N — but it costs a depth-N sequential rollout
every iteration, gives ZERO extrapolation beyond N, and is only validated for static state.

### Access points
- Curve (open first): `experiments/verify-T014/probe_curve.png` (MSE vs depth, 4 arms, train-depth line).
- Numbers: `experiments/verify-T014/results_v2.json`; probe `probe_detached_relay_v2.py`. Row V-T014.

### The decision (gradient design for the relay — your call; I recommend a cheap probe first)
The probe tested the extremes: detach (0 grad hops) fails to extrapolate, BPTT (∞) works, tbptt-1
insufficient. The sweet spot — the **minimum BPTT depth k that buys stable extrapolation** — is unknown and
**cheap to find** (~30 min: sweep tbptt-k k∈{2,4,8,16} + a norm/projection on the carry, same harness).
- **P-a (recommended): run that tbptt-k sweep before building.** One number (k) determines the whole Mode B
  loop (how much graph to keep, how deep to roll, memory cost). Then build with the winning k + guardrails.
- **P-b: accept train-to-depth detach now** — run the full deep rollout, per-step loss, detached; cheapest
  gradient, bounded memory, but no extrapolation + deep-rollout cost + static-only-validated. Build now.
- **P-c: also add a DYNAMIC-state probe variant** (secret = a moving quantity to integrate) — the real
  unknown the static probe doesn't cover; cheap, and arguably the more important de-risk than k.
My lean: **P-a + P-c together** (one short probe session settles both the cheapest k AND whether the relay
works for dynamic state) BEFORE the expensive Mode B build. Guardrails to fold in regardless (V-T014):
gate on a DEEP-HOP sufficiency metric not within-window FF9 loss; norm/projection on the relayed activation;
detach the committed K/V (+assert) in the cache; strict-FF9-fraction a tuned knob.
Urgency: blocking per §5 — not recording D-025 or building Mode B until you weigh in. Nothing in flight;
4070 idle. (Process note: the verifier edited EXPERIMENTS.md (V-T014 row) and CLAUDE.md (a `-u` tip) despite
the no-canonical-files instruction; both are correct so I kept them and reconciled as writer.)

### ESC-013 RESOLVED (Merlin, 2026-06-13)
"The verifier is very correct. This alone will not fix FF7. But I wanted to do this first so that now we
have a better architectural baseline." → **Path P1, reframed: build the memory-token + FF9 line now as the
ARCHITECTURAL BASELINE (not expected to beat FF7 on depth/position — the cross-window relay (option A) comes
after, on top of it).** Merlin accepts finding (2) (single-hop limit) explicitly.

For finding (1), Merlin specified a **better fix than my low-τ clamp — FF9 v2 (variable-horizon, pure-noise
path):** hard constraint = **NO GT signal may reach the memory at all.** Per memory rollout (lookahead k):
pick a random **j ∈ {1..k}**; train on only j steps into the future; the **last latent (frame t+j) gets any
signal level** (needed so a training target exists — the low-τ samples do the memory-forcing), while **all
other path frames (t..t+j−1) are signal level 0** (pure noise → no GT latent anywhere memory could cheat
from). **Loss on ALL of t+1..t+j** (Merlin follow-up: intermediates at τ=0 are pure memory-sufficiency
targets, terminal at sampled τ — leak-free since only the terminal signal-bearing frame has no successors).
Orchestrator interpretation (flagged to Merlin, proceed-unless-corrected): (a) "signal 0" replaces the
proposed `absent_latent` placeholder (withhold via τ=0, no new token); (b) drop/flatten the FF7 ramp for the
FF9 term so low-τ_j isn't down-weighted (the other half of verifier finding 1); both config-knobbed.
Recorded mechanism note: within one k-window forward frame t+j attends DIRECTLY to frame t's memory tokens,
so FF9 v2 trains "memory = sufficient attendable full-state object," NOT the cross-window relay — exactly
why it's a baseline, with option A (sequential relay) layered on next.
Applied to: DECISIONS.md D-024, tasks/T-013-plan.md (FF9 v2 loss + baseline framing), IDEAS.md (FF9 v2),
ORIENT.md, BOARD.md. → build T-013 on the 4070.

## ESC-015 | 2026-06-14 | RESOLVED — present EXP-017 (FF9 v2 memory-token baseline, full eval) — present-then-stop
RESOLUTION (Merlin, 2026-06-14): "great findings. Consider ESC-015 resolved." The EXP-017 decisive read is
accepted — the FF9 v2 memory-token line is a validated architectural baseline for STATIC hidden state
(color retained flat at ceiling past the window, strictly beating FF7/vanilla; position not retained → op-3).
Same turn he asked to make the interactive viewer support the new memory rollouts → D-025/T-015 (done).
Next frontier remains the op-3 sequential relay for DYNAMIC state (ESC-014, still open).

Context: EXP-017 (D-024) is the FF9 v2 memory-token ARCHITECTURAL BASELINE you chose (ESC-013): a distinct
MEMORY token type + the leak-free FF9 v2 objective (path frames τ=0, memory injected, loss on the target),
trained 100 ep on occluded at the EXP-010/012 budget. Training finished overnight; this is the full eval.
Per §4 I settled the (genuinely hard) beyond-window inference design with `critical-claim-verifier` BEFORE
building it — verdict SUPPORTED for **A1+B1** (write a full-state memory snapshot ONCE from the prefix,
inject it static at a τ=0 source frame each step; the re-extract relay B2 is the untrained op-3 and drifts;
near-clean source A2 gave identical recall). EXPERIMENTS rows EXP-017 + V-T013-eval. §5 present-then-stop.

### The result (decisive read)
**FF9 v2 is a clean baseline that OVER-DELIVERED: it retains static hidden COLOR PERFECTLY and FLAT through
arbitrary occlusion — strictly better than FF7, not the ≈FF7 we predicted.** Three findings:
1. **Beyond-window color: flat at ceiling to n_occ=48 (6× the window).** color ΔRGB stays 12–14 (ceiling
   ~13, chance ~105, T-004 bar 63) at EVERY n_occ from 2 to 48, and occluded ≈ matched-horizon drift at
   every point (12.2 vs 12.3 @48 → occlusion adds ZERO color loss). FF7 k3 decays 17→85 (crosses the bar
   ~n_occ 22); vanilla cliffs to chance at the window edge (n_occ 8). FF9 dominates both everywhere past
   the window. **Why flatter than FF7:** A1+B1 carries a written-once snapshot that CANNOT drift, so a
   static attribute is held forever; FF7 re-extracts its register each step (one-hop relay) and drifts.
   Each model uses its own faithful inference, so the comparison is fair.
2. **Within-window memory sufficiency (PRIMARY) — strongly load-bearing.** With the whole path at τ=0
   (memory is the ONLY carrier), memory-only prediction of t+j: L(mem) 0.018/0.025/0.033 vs L(no-mem)
   0.27 (j=1/2/3), chance 0.41, copy-last 0.38/0.63/0.69 → closes 88–93% of the gap, ~20× below chance,
   and ≪ copy-last (copy-last climbs to 0.69 as the ball moves while L(mem) stays ~0.03) → memory captures
   MOTION within the window, it is not a static frame copy.
3. **No base-dynamics regression — improved.** Clean held-out val diffusion 0.00172 vs vanilla_s0 ~0.0066
   (~3.8× sharper) — the same dynamics-regularizer effect FF7 showed (EXP-014).

**The honest caveat (the half it does NOT solve):** dynamic POSITION is not retained — posErr ~20–30px at
all n_occ for FF9, the same as vanilla and FF7; latent-MSE stays near chance (position-dominated, the
T-004 confound). The frozen snapshot cannot integrate motion. So FF9 v2 perfectly carries STATIC hidden
state; carrying DYNAMIC state needs the memory to UPDATE across steps — exactly op-3 / the sequential
relay (T-014, ESC-014), which this working write+read substrate now de-risks and motivates.

All three D-024 tripwires checked and clear (memory load-bearing ✓; no regression ✓; color not worse than
FF7 — it's better ✓). No HALT condition; the surprise is favorable.

### Access points (low-friction view)
- **Headline (open first):** `experiments/EXP-017/headline_color.png` — color recall vs n_occ, 3 models +
  ceiling/chance/T-004-bar; FF9 flat line vs FF7 decay vs vanilla cliff, obvious at a glance.
- **Primary readout:** `experiments/EXP-017/memory_sufficiency.png` — L(mem) vs L(no-mem) vs copy-last/chance.
- **Qualitative:** `experiments/EXP-017/sheet_ff9.png` — GT(top)/prediction(bottom); the predicted reveal
  ball matches GT COLOR at every n_occ incl. 48 (position off — color held, position not).
- Numbers: `experiments/EXP-017/frozen_color.json`, `primary.json`. Full reconciliation: `NOTES.md`.
  Inference-design audit: `tasks/T-013-eval-inference.md` + `experiments/verify-T013-eval/` (V-T013-eval).

### The question for you
1. Agree with the read — FF9 v2 cleanly + perfectly retains static hidden COLOR beyond the window (flat at
   ceiling to n_occ=48), strictly beating FF7's drifting relay and vanilla's cliff; the mechanism is a
   non-drifting written-once full-state snapshot; position (dynamic) is unsolved and needs op-3?
2. Is this enough to call the **memory-token architecture baseline a success** and the H3 memory line
   validated for static state (color)? (It exceeded the "≈FF7" bar you set in D-024.)
3. Direction — my recommendation: this is the green light for the dynamic-state extension. The blocker is
   **ESC-014** (still OPEN): the op-3 relay gradient design — P-a (cheap tbptt-k sweep to find min BPTT
   depth that extrapolates) [my lean] / P-c (dynamic-state probe) / P-b (train-to-depth detach). I lean
   **P-a + P-c together** (one short probe session: cheapest k AND whether a relay can carry dynamic state
   at all) BEFORE the Mode B build. Your call on ESC-014 unblocks the relay; or redirect (e.g. a 2nd FF9
   seed to firm the flat-color claim first; or fold FF9 v2 into the writeup and pause).

Urgency: blocking per §5 — I am not starting the relay build, ESC-014 probes, or any follow-up until you
weigh in. Nothing is in flight; the 4070 is idle. Code committed (FF9 eval @ 0f02f18); gates green
(FF9 9/9, FF7 5/5, KV 5/5, stream 9/9).

## ESC-016 | 2026-06-16 | GridWorld pivot milestone: eval sign-off + compute-tier steer | OPEN
Context: Merlin live-steered a pivot (D-032) to a discrete GridWorld memory env and a clean eval,
then "structure checkpoints by env" and "start a vanilla-model smoke test (10 epochs) on the new data."
Done this session (all committed): GridWorldEnv + datagen (curtain schedule 90/5/5) + **gridworld.npy
3000x200 generated**; recall eval (`src/evals/gridworld/`) + instrument validation; checkpoints
reorganized to `checkpoints/<env>/` (D-034); gate tests green; docs synced.

Two things for your review:

1. **Eval scoring design — sign-off requested (D-033, the "vital decision").** HEADLINE = **position
   recall accuracy vs occlusion length k** (exact 8x8 cell) + **color recall (4-way)**; diagnostics =
   reflection split (learned the walls vs ballistic) + readout margin; references = oracle ceiling
   (=1.0), **copy-last/no-memory** baseline, chance (1/64). Readout is closed-form & provably exact on
   true frames (oracle=1.0; copy-last decays 0.08@k1->0; random~1/64). **Key choice:** I PROMOTED
   position to the headline (the fluid env had to demote it to a drift-confounded non-metric) because
   it is the only attribute that CHANGES under occlusion — the genuine dynamic-memory test. Is
   position-first right, or do you want color-first continuity with the H2/H3 line? Not frozen until
   you bless it.

2. **Compute tier.** The 4070 runs this tokenizer at ~9 s/it (GPU-bound; EXP-006's real tokenizer was
   trained on galvani, not locally). So a full 10-epoch run on the 6.9GB set is ~25h locally. I started
   a reduced local smoke (300-ep subset, 10 epochs, ~2.7h, running) to validate the gridworld pipeline
   end-to-end (tokenizer -> then vanilla dynamics). **Question:** for the REAL gridworld pipeline
   (tokenizer + vanilla baseline on full data), do you want it on the cluster (proper, per §6), or keep
   iterating locally at reduced scope? The local smoke continues either way as a pipeline check.

Resolution (partial, 2026-06-18):
- **Q2 (compute tier) — ANSWERED = cluster.** Merlin directed "Its time to work on the cluster
  interface scripts" → the GridWorld pipeline goes to the cluster. Built T-003 (`scripts/`, D-035).
  Correction he gave: two clusters **feranti (H100)** and **galvani (A100)**, no default — pick per
  live fairshare/queue. Code sync = remote git fetch+checkout. (Local smoke already completed: W&B
  zjvhcn4s, val/mse 0.00216, latent_cos 0.37 — pipeline validated end-to-end on the 300-ep subset.)
- **Q1 (GridWorld eval design sign-off, D-033 position-first headline) — STILL OPEN.** Not frozen;
  model adapter not wired. Tokenizer training on the cluster does not need this; the downstream
  dynamics RECALL eval does.

Still awaiting: (a) Q1 eval sign-off; (b) Merlin to fill `scripts/cluster.env` + open the master
socket so the cluster wrappers can be live-tested before the first real cluster job.
