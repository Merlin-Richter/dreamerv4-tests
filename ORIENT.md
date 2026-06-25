# ORIENT.md

Rewritten: 2026-06-25 (overnight autonomous memory-training campaign, D-048).

## What we're doing right now and why
**Autonomous overnight campaign (Merlin asleep, explicit instruction to run multiple memory-training
ideas without the present-then-stop gate; D-048).** Frontier = FF9 memory tokens contain dynamic
hidden state in-window (EXP-028: pos 0.94 in-window, decays to chance ~k28) but the memory->memory
relay (op-3) is UNTRAINED. Tonight: implemented FF9 **rollout-training** (trains the relay on the
gradient path, TBPTT-k) and launched a budget-matched A/B on the GridWorld bench.

## P1 gating probe DONE (EXP-029) — the result that shaped the design
Dynamic-secret relay credit probe (continuous 1-D bounce, integrate each hop). **For DYNAMIC state
there is NO free extrapolation: recall horizon == training rollout depth.** Within train depth,
tbptt-k carries to ~2k hops (tbptt16 ~= full BPTT to depth 31); beyond train depth ALL modes drift to
chance or worse (BPTT overshoots to 6.75 > 2x chance @d199). DICTATES: to recall to k~D, train the
rollout to depth >=D. CAVEAT: continuous-position probe is PESSIMISTIC for GridWorld's discrete/
bounded/periodic state — the GridWorld A/B is the real test (-> discrete-memory/VQ if it still drifts).

## IN FLIGHT — 3 ferranti jobs (branch feat/ff9-rollout-training; training code == 1a00ba6, synced a4588b9)
- **EXP-030 job 409752** — FF9 rollout MODERATE (window16, clip28, h24, tbptt12, tail, +ff9 3,
  warmup20, 80ep). RUNNING (ep9 ~03:00, train0.10/val0.08, ETA ~06:40). Monitor: bg wait btxr1rma7.
- **EXP-032 job 409753** — VANILLA window-32 control. RUNNING (~03:00, ETA ~04:30). Monitor: bg wait bkmep1943.
- **EXP-031 job 409754** — DEEP (clip48, h44, tbptt16). PENDING (Resources) — may not finish by morning (bonus).
Implementation + verifier (all 4 claims SUPPORTED) + eval tooling all DONE & committed. When a job lands:
follow experiments/EXP-030/EVAL_RUNBOOK.md (pull -> recall_relay.py relay+windowed -> plot_rollout_compare).

## BASELINE DONE this session (the "before" for the A/B)
experiments/EXP-030/recall_env_ff9_norollout_relay.json — FF9 v2 (no rollout) under the UPDATING-memory
relay inference: untrained B2 relay COLLAPSES by k~3 (pos 0.67->0.17->chance@k4; color->chance@k8).
The bar FF9+rollout must clear under the same relay inference. See EXP-030/NOTES.md for inference
semantics (relay = 2-frame pure-memory; windowed = EXP-028 sliding-window; don't conflate them).

## NEXT ACTIONS (in order)
1. **Implement UPDATING-memory inference** (essential for eval — the trained relay is exercised ONLY
   by an updating memory carry, not plain sliding-window or the frozen snapshot). Mirror
   full_state_rollout_step but UPDATE mem_carry each step from the written memory. Add a generate
   dispatch + an eval adapter mode.
2. **critical-claim-verifier** on _ff9_rollout_loss (correctness of the relay gradient path; runs in
   parallel with training).
3. When checkpoints land: env-direct recall A/B vs vanilla(EXP-027) + FF9(EXP-028), under BOTH
   updating-memory and plain inference. Build comparison views. Present-then-stop deferred to a single
   consolidated MORNING BRIEF (Merlin asked for autonomous overnight work).
4. If the relay still drifts on GridWorld -> the discrete-memory (VQ) idea (see DECISIONS D-048 notes
   / the morning escalation).

## Implementation state (all on feat/ff9-rollout-training, committed)
- `_ff9_rollout_loss` (dynamics_model.py): differentiable memory chain, TBPTT-k, hide_mode {tail,iid},
  seed-write + h hops of 2-frame [source|new] windows; carry = written memory (op-3 relay). loss()
  windows main terms to max_temporal_length, feeds full clip to rollout term. Identity-when-off.
- train_dynamics flags: --ff9-rollout H --ff9-rollout-tbptt K --ff9-rollout-phide P
  --ff9-rollout-hide-mode {tail,iid} --ff9-rollout-warmup E --rollout-clip-len N.
- encode_frames chunks T>16 clips into 16-frame tokenizer windows (frozen tokenizer is window-16).
- Tests: src/tests/test_ff9_rollout.py (6/6, incl. relay-Jacobian + TBPTT-depth + byte-identical-off).
  FF7/KV/stream gates green. Local smoke (clip28/h24) trains+saves clean.

## Open escalations / worries (for the morning brief)
- ESC-020/021 still OPEN (FF9 corrected-inference 2nd-seed/verifier; rollout-training design sign-off).
  Tonight's runs partly cover ESC-021 (built + running C1). Morning brief will consolidate.
- The eval inference choice (updating-memory vs plain vs frozen-snapshot) is subtle and result-defining
  — must be applied identically to baselines and documented. Verify before claiming any A/B.
- P1's "no extrapolation for dynamic state" is the key risk: if GridWorld's discrete state behaves like
  the continuous probe, rollout-training only extends recall to ~train depth (h), not arbitrarily.

## Parked (pre-pivot; resume only if Merlin redirects)
- C1/motion (EXP-021), occluded-line H3 (FF7/FF9 static-color). See BOARD-archive / prior ORIENT.
