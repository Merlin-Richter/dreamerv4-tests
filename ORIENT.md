# ORIENT.md

Rewritten: 2026-06-13 ~09:00 (T-010 done; EXP-012 budget-matched baseline TRAINING)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75; T-004 H3 bar = color ΔRGB < ~63 at n_occ {12,16,24}.
- **H3 — FF7 v1 supports COLOR (EXP-010).** Position reframed by EXP-011: position is encoded in
  the tokenizer (probe R²=0.96, deficit is in D not C); FF7 tracks motion well (1-step ~1px,
  open-loop ~12 steps); my_dynamics is a weak motion model (1-step 4.5px). Occluded-pos-at-chance
  = dead-reckoning chaos, not a base-capability wall. Merlin agreed (ESC-007).
- **EXP-011 surfaced a confound:** FF7-better-dynamics vs my_dynamics-undertrained is unidentified;
  EXP-009/010 are not training-matched. → EXP-012 fixes it.

## In flight
**EXP-012 — budget-matched VANILLA baseline TRAINING** (D-016, background task b6ox435tv, local
4070 via venv/Scripts/python.exe). Exact EXP-010 budget, --ff7 0 --fresh, 100ep+probe (~2.6h).
run.sh chains: train → frozen probe → (then I rerun the EXP-011 diagnostic on it). Logs:
experiments/EXP-012/{train.log,probe.log}. W&B exp012-vanilla-s0.
**Confirm liveness:** train.log should show "Epoch 1" within ~2 min at ~90s/epoch (GPU). If it
stalls or runs CPU-slow → check venv/CUDA (HOWTO/gpu_venv.md), do NOT silently relaunch.

## NEXT ACTION when EXP-012 finishes
Reconcile in experiments/EXP-012/NOTES.md (pre-registered there): (1) does it reproduce the
EXP-009 post-window COLOR cliff? (H2 should stand — architectural.) (2) Rerun the EXP-011
diagnostic (`venv python experiments/EXP-011/diagnose.py` won't pick it up — add vanilla_s0.pt to
MODELS or a one-off) → vanilla 1-step pos_err: ≈FF7 ⇒ gain was budget; >FF7 ⇒ FF7 loss helps
dynamics. (3) D-016 tripwire: vanilla beyond-window color < bar ⇒ EXP-010 color win wasn't the
relay ⇒ halt. Build comparison view (vanilla vs FF7 vs my_dynamics), decisive read, ESC-008,
present-then-stop. After: Q3 path = closed-loop/distributional position metric.

## Recently done
- **T-008 (D) — KV cache** (D-017, 2026-06-13, Merlin-directed independent work while EXP-012
  trains). Absolute-position RoPE + `generate_cached` (intra-frame substep K/V reuse), bit-for-bit
  == `generate`, ~2× faster, validated past the cos/sin table. Gate green: test_kv_cache.py 5/5,
  FF7 smokes 5/5. Training/default forward unchanged. Cross-frame eviction cache left as optional
  follow-up (BOARD). Does NOT touch EXP-012.
- **T-010** — play_dynamics_checkpoint carries FF7 registers (refactor → memory_rollout_init/step;
  verified 9.9 vs 64.4 dRGB). Merlin can re-test interactively on ff7_k3.

## Current worries
1. EXP-012 liveness/GPU — verify it's training on GPU (not CPU/stalled) before trusting the ~2.6h ETA.
2. batch-size=32 is inferred (EXP-010 had no saved config; 327 iters/epoch ⇒ 32). If the
   comparison hinges on a tiny gap, the budget-match isn't perfect — note when reconciling.
3. The EXP-011 diagnostic rerun on vanilla needs wiring (its MODELS dict is hardcoded) — small task
   at reconciliation time.
