# ORIENT.md

Rewritten: 2026-06-13 ~14:00 (EXP-012 DONE → ESC-008 present-then-stop; awaiting Merlin)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75. H3 bar (T-004) = color ΔRGB < 63 at n_occ {12,16,24}.
- **H3 — FF7 v1 supports COLOR (EXP-010), and EXP-012 just CLEARED the confound:** a budget-matched
  vanilla baseline (val loss = FF7's) confirms FF7's wins are the **method, not training budget**,
  on both axes — color (vanilla → chance beyond window; FF7 holds below bar) and motion (both
  vanillas ~4.5px 1-step ≫ FF7 ~1.0px). my_dynamics retired; vanilla_s0 is the H2/H3 baseline.
  EXP-009/010 conclusions retroactively trustworthy.
- **Position memory still open** (the ESC-006/007 question): position is encoded in C (probe R²=0.96)
  and FF7 tracks motion 1-step ~1px, but occluded position = dead-reckoning chaos under the current
  open-loop metric. Needs a closed-loop/distributional position metric to judge honestly.

## In flight
**Nothing running.** 4070 idle. **Blocked on Merlin's ESC-008 verdict** (present-then-stop, §5) —
NOT starting the next decision until he answers.

## NEXT ACTION (after Merlin's ESC-008 verdict)
Recommended path (proposed in ESC-008, agreed Q3 direction from ESC-007): (1) design the
**closed-loop / distributional position metric** (cheap; unblocks honest position claims); then
(2) the **sequential stop-grad register-relay training** idea (IDEAS.md, worked out with Merlin
2026-06-13) as the leading H3 *position* method — TBPTT-1 relay so context carries real relayed
memory tokens. If he redirects (relay-first, or a 2nd vanilla seed to firm the motion claim), follow that.

## Recently done
- **EXP-012 — budget-matched vanilla baseline (D-016)** done; confound resolved (see above).
  Views: experiments/EXP-012/{headline_color.png, headline_motion.png, sheet.png}. → ESC-008.
- **T-008 (D) — KV cache (D-017)**, Merlin-directed independent work during the EXP-012 wait.
  Absolute-RoPE + generate_cached, bit-for-bit == generate, ~2× faster. test_kv_cache.py 5/5,
  FF7 smokes 5/5. Cross-frame eviction cache = optional follow-up (BOARD).
- **T-010** — play_dynamics_checkpoint carries FF7 registers (memory_rollout_init/step).

## Open threads / parked
- **Sequential register-relay training** (IDEAS.md): the train/inference mismatch — main loss
  trains on learned-init registers, never deeply-relayed ones. Merlin's idea: sequential stop-grad
  relay (TBPTT-1) so context has real memory tokens. Parked on his ESC-008 direction.
- **Cross-frame KV eviction cache** + tokenizer-C cache (optional, BOARD).

## Current worries
1. The motion surprise: vanilla ≈ my_dynamics (both ~4.5px) refuted my undertraining prediction.
   Read as "weakness intrinsic," not a bug (two independent vanillas agree) — but single-seed; a
   2nd vanilla seed would firm it if Merlin wants. Flagged in ESC-008.
2. FF7's base-dynamics improvement (4.6×) conflates loss vs relay-inference — not yet disentangled.
