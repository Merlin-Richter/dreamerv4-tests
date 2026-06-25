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

## CAMPAIGN COMPLETE — AWAITING MERLIN (ESC-022 morning brief is the present-then-stop)
EXP-030 (h24), EXP-031 (h44 deep), EXP-032 (vanilla w32) all TRAINED + EVALUATED (env-direct recall
A/B). EXP-033 (M=16 capacity test) STILL TRAINING (job 409760, ~ep4 ~07:50, ETA ~12:30) — append its
relay curve when done (EVAL_RUNBOOK). Headline figure: experiments/EXP-030/compare_rollout.png.
**RESULT (see ESC-022 for the full decisive read):** rollout-training is a verified working credit fix
-> a persistent bounded memory that carries STATIC hidden state FLAT beyond the window (color 0.8@k32)
and, with DEEP training (h44>h24), sustains residual DYNAMIC position FAR past the window (k>=20) where
the latent-window inference has decayed to chance. Depth is a real lever. BUT dynamic precision is
modest + saturates short of the training depth (continuous-memory cap), and the relay TRADES OFF
near-window windowed dead-reckoning. -> recommend DISCRETE/VQ memory next (needs Merlin sign-off).

## AUTH_DEAD RESOLVED (Merlin re-auth'd ~12:18). EXP-033 (M=16) COMPLETED rc=0 (val 0.081 == M=4).
M=16 relay eval RUNNING locally (capacity-vs-stability: does wider memory lift dynamic precision?).
Append its curve to ESC-022 + NOTES when done, then truly idle on Merlin's ESC-022 verdict.

## NEXT (awaiting Merlin's ESC-022 verdict)
1. [in progress] EXP-033 (M=16) relay eval -> capacity vs representation. If M=16 doesn't lift dynamic
   precision over M=4 -> the cap is REPRESENTATION (continuous drift) -> firms discrete/VQ.
2. On his call: build DISCRETE/VQ memory (my lean; architecture change) OR consolidate the static-memory
   relay win + the honest dynamic result. Do NOT build VQ without sign-off.

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
