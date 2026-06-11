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

Resolution: (pending)
