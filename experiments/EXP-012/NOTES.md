# EXP-012 — budget-matched vanilla dynamics baseline (D-016)

Decision: D-016. Spawned by ESC-007 (FF7-loss vs training-budget confound; baselines not
training-matched). Vanilla (--ff7 0 --fresh) at the EXACT EXP-010 budget. Config + launch:
experiments/EXP-012/{config.yaml,run.sh}.

## Purpose / what it controls
my_dynamics (old H2 baseline) is weak at motion (EXP-011: 1-step 4.5px > copy-last 3.2px), but
its training provenance is older/approximate, so EXP-009↔EXP-010 are not training-matched. This
run is the clean control: identical architecture + data + 100ep budget as the FF7 arms, FF7 loss
OFF. Lets us attribute the EXP-011 dynamics gain and re-anchor H2/H3.

## Pre-registered expectations (D-016, before results)
- COLOR cliff: reproduces EXP-009 (post-window color recall → chance; architectural). H2 stands.
- MOTION (1-step teacher-forced pos_err): expect << my_dynamics 4.5px (much of that was
  undertraining); the open question is vanilla-budget-matched vs FF7 ~1.0px.
  - vanilla ≈ FF7 (~1px) ⇒ dynamics gain was training budget, not the FF7 loss.
  - vanilla still > FF7 ⇒ FF7 loss genuinely improves dynamics.
- Tripwire (D-016): vanilla showing beyond-window color recall < T-004 bar ⇒ EXP-010 color win
  was not the register relay ⇒ halt + rethink.

## Plan
1. Train (run.sh first stage) → vanilla_s0.pt; 2. frozen probe → results.json + sheet;
3. rerun EXP-011 diagnostic on vanilla_s0.pt (open-loop/teacher-forced/copy-last) for the
   apples-to-apples motion comparison vs FF7 and my_dynamics.

## Provenance
- Code: master @ <fill>. Local 4070. W&B exp012-vanilla-s0 (project transformer-D-dynamics).

## Observed
<fill after run>

## Reconciliation
<fill — Expected / Observed / Surprise / H2+attribution / Tripwires / Next>
