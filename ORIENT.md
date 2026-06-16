# ORIENT.md

Rewritten: 2026-06-16 ~06:00 (AUTONOMOUS overnight session, D-028/D-029. Merlin asleep; continuous
work, NO escalation gates this session, additive/non-breaking. Returns in the morning.)

## What we are doing and why
- **Mode:** work continuously+autonomously on the C1/motion idea; don't block on gates; produce
  valuable info; don't break anything; clean up the repo during training downtime.
- **Problem:** dynamics model can't hold a ball trajectory over multi-step OPEN-LOOP rollout (even
  without occlusion). EXP-018 diagnosed it as autoregressive COMPOUNDING / exposure bias.
- **C1 (D-027):** config-gated multi-step DAgger loss `_multistep_loss` — trains on the model's OWN
  detached self-generated context held at context_signal, supervising off-manifold rollout states.

## In flight (1 GPU training job)
- **EXP-021 — full-data C1 TRAINING (background bb2st8ox6).** C1 on ALL 1000 occluded episodes (vs
  EXP-020's 250-ep subset), same C1 flags, 40-ep budget (will be interrupted by morning ~ep8-10; 327
  batches/ep, ~24min/ep; per-epoch ckpt → experiments/EXP-021/c1_full_s0.pt). Log: EXP-021/train.log.
  Purpose (D-029): CONFOUND-FREE compounding test — compare C1-full vs the COMPETENT reference set
  (vanilla_s0/ff7_k3/ff9v2, all full-data) at matched TF-map quality. Resolves EXP-020's weak-control
  confound.

## Headline results this session (all committed)
1. **EXP-020 (C1 vs vanilla, 250-ep subset) — C1 SUPPORTED at full budget.** Open-loop pos_err stays
   BELOW chance through h24 (17px@h24; crossChance h=25) vs control at chance from h1; TF flat ~2px vs
   control ~20px; val 0.00305 < control 0.0082; collapse monitor coherent (predDisp 4.4). View:
   experiments/EXP-020/headline.png. CAVEAT: 250-ep control is a WEAK motion model (chance TF) →
   confounds "C1 learns a map at all on tiny data" with "C1 fixes compounding on a competent map" →
   EXP-021 is the clean test.
2. **EXP-022 (context_signal inference sweep) — BIG finding, answers Merlin's noise question.** Lowering
   the rollout's context_signal (telling the model its self-generated context is LESS reliable) nearly
   HALVES ff7_k3's open-loop compounding (h16 18.6→10.5px), no retraining, per-step map untouched. C1 is
   already optimal at the default 0.9 (it INTERNALIZED that robustness via its DAgger loss). vanilla is
   flat (bad map). → the inference-trust lever is real+large on competent models; all prior open-loop
   evals at 0.9 were SUBOPTIMAL for ff7-type models. View: experiments/EXP-022/sweep.png.

## NEXT ACTIONS (autonomous; morning)
1. **When EXP-021 reaches a usable epoch (or completes):** probe the latest c1_full_s0.pt with the
   open-loop+TF curves (src/eval/motion via probe_multistep or a small driver) AND **sweep context_signal
   per model** (EXP-022 made the s-sweep a standard eval axis). Compare C1-full-at-best-s vs
   ff7_k3-at-best-s vs ff9v2: does C1's TRAINED robustness beat ff7 + the tuned inference knob? Reconcile
   + view + index + commit.
2. Candidate follow-ons (not started): per-frame/decaying trust schedule s(j) (needs a small generate()
   shim); 2nd seed of EXP-020 for stability; fold EXP-014/015 reusable code into src/eval if time.

## Repo cleanup done (D-028)
- `src/eval/` toolbox created: motion curves (`motion.py`) + reusable A/B view (`ab_view.py`) extracted
  verbatim from experiments; EXP-018/020/022 drivers import it; CPU smokes reproduce known numbers.
- Hygiene: removed scratch test.py + stale pycache. Docs: REPO_MAP.md (concept→location map),
  src/eval/README, CLAUDE.md synced. Deeper src/ reorg staged as tasks/T-019 (DEFERRED, needs approval).

## Current worries
1. EXP-021 may only reach ~ep8-10 by morning — partial, but informative (EXP-020 partial ep17 was). If
   C1-full's TF map isn't competent yet at the morning checkpoint, the compounding comparison is
   inconclusive → let it train longer.
2. The EXP-022 finding reframes C1: part of C1's value may be reproducible by tuning inference
   context_signal. EXP-021's fair test (C1-full-best-s vs ff7-best-s) settles whether C1 adds more.
3. Single seed throughout. Positive deltas need a seed/confirmation before any strong standalone claim.

## Parked
- **ESC-014 (OPEN):** op-3 relay gradient design (DYNAMIC-state occluded memory). Parked since Merlin
  redirected to motion. Resume after the motion thread.
