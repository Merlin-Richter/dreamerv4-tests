# ORIENT.md

Rewritten: 2026-06-14 (AUTONOMOUS motion-prediction session, D-026. EXP-018 diagnosis DONE →
building C1. Branch feat/motion-prediction.)

## What we are doing and why
- **Operating mode (Merlin, this session):** work autonomously for several hours on MOTION
  PREDICTION, "dont block yourself" (= do NOT halt-and-wait at experiment gates this session),
  "dont break anything" (additive/config-gated changes only, all gates green, frozen tokenizer/
  probe untouched, branch `feat/motion-prediction`). Use the idea agent for inspiration. He's away.
- **The problem (his redirect):** predict ball MOTION/position over multiple steps, even WITHOUT
  occlusion. Prior check (EXP-011/12/13/14): tokenizer encodes position fine (R²0.96); vanilla can't
  even predict 1 step (4.5px > copy-last); FF7/FF9 aux loss → good 1-step (~1px); but NO model holds
  a trajectory past ~8-12 in-context steps.
- **EXP-018 diagnosis (DONE, decisive):** teacher-forced pos_err is FLAT in horizon for ALL models
  (ff7/ff9 ~1px h1→h24); open-loop compounds to chance. ⇒ the deficit is **autoregressive error
  COMPOUNDING / exposure bias**, not a depth-degrading map. τ-sweep flat ⇒ C0 ruled out.
- **Method (method-architect T-016 + verifier T-017):** **C1** = config-gated, identity-when-off
  time-axis multi-step prediction loss (sibling of `_ff9_loss`; τ=0 successor target predicted from
  the model's OWN detached self-generated context; TBPTT-1). Directly trains compounding-robustness.

## In flight (2 background jobs)
1. **EXP-018 canonical artifact run** (brvtvbec1): full P1+lightP2, 32 eps → diagnosis.json + .png.
2. **critical-claim-verifier on C1** (T-017): scrutinizing the design before I implement. **MUST
   read its verdict before coding C1** (§4).
GPU otherwise idle. No cluster (scripts/ deferred).

## NEXT ACTIONS (autonomous, no waiting on Merlin)
1. Read C1 verdict (T-017-C1-verdict.md) → implement C1 in dynamics_model.py (revise per verdict),
   add smoke tests incl. multistep_h=0 identity guard; keep FF7/FF9/KV/stream gates green.
2. Launch budget-matched A/B on occluded subset (--max-episodes): vanilla CONTROL vs C1 treatment,
   same subset/seed/epochs. Eval curtain-up open-loop with probe_multistep tooling.
3. Reconcile + build views + write up; record EXP-019/020. Iterate (C2 scheduled-sampling if late
   horizons still drift). Do NOT start op-3/ESC-014 work.

## Open escalation (parked this session)
- **ESC-014 (OPEN):** op-3 relay gradient design (DYNAMIC-state occluded memory). Parked — Merlin
  redirected to motion. Resume after this session / on his return.

## Recently done
- **EXP-018 motion diagnosis — DONE 2026-06-14.** Compounding confirmed; C0 ruled out. Tooling:
  experiments/EXP-018/{probe_multistep.py, make_views.py, NOTES.md}. method-architect proposal
  (tasks/T-016-architect-proposal.md), C1 design (tasks/T-017-C1-design.md). Added safe
  --max-episodes train flag.
- Viewer FF9 support (D-025/T-015); EXP-017 FF9 v2 accepted (ESC-015 resolved).

## Current worries
1. C1 cost (~2.3× vanilla/batch) on the 4070 — using a 250-ep subset + ≤30 ep for the A/B. Absolute
   numbers will be below the 100-ep references; the A/B DELTA (same budget) is the signal.
2. Off-manifold drift of self-generated latents over h steps could cap any motion loss (verifier +
   a decode-validity check should flag it). Watch clean val/diffusion for single-frame regression.
3. Single seed, short budget — a positive delta will need a confirmation run before any strong claim.
