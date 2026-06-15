# ORIENT.md

Rewritten: 2026-06-16 (AUTONOMOUS overnight session, D-028. Merlin asleep; continuous work, NO
escalation gates this session, additive/non-breaking, + repo cleanup during training downtime.)

## What we are doing and why
- **Operating mode (Merlin, this session):** work continuously+autonomously on the C1/motion idea
  overnight; do NOT block on escalations/present-then-stop gates; just produce valuable information;
  don't break anything; work within the folders. Use training downtime to (a) extract reused
  code/evals out of experiments/ and (b) clean up the repo. He returns in the morning.
- **The problem:** the dynamics model can't hold a ball trajectory over multi-step OPEN-LOOP rollout,
  even WITHOUT occlusion. EXP-018 diagnosed this as autoregressive error COMPOUNDING / exposure bias
  (teacher-forced per-step map is fine for ff7/ff9 ~1px flat; open-loop compounds to chance ~h12).
- **C1 (D-027):** config-gated multi-step DAgger loss `_multistep_loss` — trains on the model's OWN
  detached self-generated context held at context_signal, supervising the off-manifold states open-loop
  visits. Built + verifier-vetted (V-T017-C1: mechanism = on-policy distribution correction, helps iff
  the deficit is off-manifold accuracy, which EXP-018 confirms).

## In flight (1 GPU training job)
- **EXP-020 FULL C1 run — TRAINING (background b76z65vb7).** 40 ep, 250-ep subset, bs32 lr3e-4 seed0,
  --multistep 4 --lambda-multistep 1.0 --multistep-warmup 10 → experiments/EXP-020/c1_h4_s0.pt.
  Log: experiments/EXP-020/train_full.log. ~5-6h (C1 ~2.3× vanilla/batch). Provenance: run.sh added.
  Started ~epoch1 at 00:1x. When done → run ab_eval + build view + reconcile (NOTES) + record.

## Done this session
- **Preliminary EXP-020 readout (partial ep17 C1 vs full control):** C1 wins big — TF flat 2.6px vs
  control ~23px (at chance); OL crosses chance h~20 vs h=1; collapse monitor PASSES (predDisp 4.6 vs
  gt 3.2). Caveats: budget mismatch + weak control. Artifacts: experiments/EXP-020/{NOTES.md,ab.json}.
- **Control sanity-check:** the 250-ep subset CRIPPLES vanilla motion (TF ~20px=chance); full-data
  vanilla_s0 is ~4.5px. So EXP-020's A/B confounds "C1 learns a map at all on tiny data" with "C1 fixes
  compounding on a good map" → motivates a larger-data A/B vs the existing vanilla_s0 (see NEXT).
- **Rollout-noise question (Merlin asked):** self-generated frames are stored as clean 100%-signal
  latents and fed back at the SAME fixed context_signal=0.9 as real frames — NO provenance marker, NO
  per-frame confidence channel. That's the exposure-bias mechanism; C1 fixes the distribution half, not
  the confidence-representation half. (Lever idea → IDEAS.md, see below.)
- **Refactor (D-028, safe + test-verified):** new `src/eval/` toolbox; moved motion curves
  (open_loop/teacher_forced/τ-sweep + A/B helpers) verbatim from experiments → `src/eval/motion.py`;
  EXP-018/EXP-020 drivers now import it (CPU smokes reproduce known numbers). Hygiene: removed scratch
  test.py + stale pycache. Docs: REPO_MAP.md (concept→location map), src/eval/README, CLAUDE.md synced.
  Deeper src/ reorg (models/training/envs split) DEFERRED as tasks/T-019 (needs Merlin approval).

## NEXT ACTIONS (autonomous; no waiting on Merlin)
1. **On EXP-020 train completion:** `python -u experiments/EXP-020/ab_eval.py --episodes 48 --horizon 24`
   → build headline view → reconcile in NOTES (vs D-027 expectation + tripwires) → update EXPERIMENTS
   index → commit. (Present-then-stop is RELAXED this session; record the read, keep going.)
2. **Then (next decision, GPU free): EXP-021 larger-data A/B** — C1 on a bigger subset/full occluded
   data vs the EXISTING full-data vanilla_s0 (which has a good ~4.5px TF map), to isolate the
   compounding fix from the tiny-data confound. Draft run.sh + decision during downtime; launch after.
3. **Downtime (CPU, outcome-independent):** EXP-020 view generator; write the context_signal-as-
   confidence-channel method idea into IDEAS.md (Merlin's noise question → a concrete C-variant).

## Current worries
1. EXP-020 (250-ep) confound (above) — the clean test is the larger-data A/B (#2). Don't overclaim
   from EXP-020 alone.
2. C1 capacity tension (V-T017-C1 Part2): the multistep term can steal capacity from the 1-step map at
   long h. Watch clean val/diffusion (tripwire ≤ ~0.003) + TF flatness in the full run.
3. Single seed, short budget — a positive delta needs a confirmation/seed run before any strong claim.

## Parked
- **ESC-014 (OPEN):** op-3 relay gradient design (DYNAMIC-state occluded memory). Parked since Merlin
  redirected to motion. Resume after the motion thread.
