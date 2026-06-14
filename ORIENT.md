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

## Done since: EXP-018 diagnosis + C1 built & verified
- C1 implemented (D-027): config-gated `multistep_h` loss `_multistep_loss` in dynamics_model.py +
  train flags --multistep/--lambda-multistep/--multistep-warmup. Verified V-T017-C1 (identity-when-off
  proven, detach safe, mechanism=DAgger). Smokes 6/6; FF9/FF7/KV/stream gates all green. Committed.

## In flight (1 background job)
- **EXP-019 vanilla CONTROL training** (job b0j8a4uiw): occluded 250-ep subset, 40 ep, bs32, seed0
  → experiments/EXP-019/vanilla_ctrl_s0.pt + train.log. ~19 min. code @ a07fdee.
- A/B eval harness ready: experiments/EXP-020/ab_eval.py (open-loop + TF curves + displacement
  collapse monitor on curtain-up episodes).

## NEXT ACTIONS (autonomous, no waiting on Merlin)
1. When CONTROL done → launch **EXP-020 C1 treatment** (same budget + --multistep 4 --lambda-multistep
   1.0 --multistep-warmup 10) → experiments/EXP-020/c1_h4_s0.pt.
2. When both done → `python experiments/EXP-020/ab_eval.py` → compare open-loop curves + val/diffusion
   (tripwire: C1 val_diffusion not materially > control's) + per-j multistep monitor (prior-emission).
3. Reconcile (EXP-019/020), build view, write up. Iterate: if open-loop unchanged → re-diagnose;
   if late horizons still drift → C2 scheduled-sampling. Do NOT start op-3/ESC-014 work.

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
