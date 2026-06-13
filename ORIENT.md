# ORIENT.md

Rewritten: 2026-06-13 (ESC-008 RESOLVED → designing the position-memory consistency metric, D-018)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75. H3 color bar (T-004) = ΔRGB < 63 at n_occ {12,16,24}.
- **H3 — color retention is the FF7 method, confound resolved (EXP-012).** Budget-matched vanilla
  (val 0.0066 = FF7's) reproduces the architectural COLOR cliff → chance beyond the N=8 window but
  NOT FF7's sub-bar retention; on MOTION both vanillas ~4.5px 1-step ≫ FF7 ~1.0px. `my_dynamics`
  retired; `vanilla_s0` is the H2/H3 baseline. EXP-009/010 retroactively trustworthy (Merlin agreed).
- **Position memory — now the active front, with a NEW metric direction.** Merlin rejected the
  open-loop GT-matched position metric (ESC-008): it penalizes both the non-tracker ("ball is center")
  AND the accurate-but-butterfly-desynced tracker. New direction (D-018): measure whether the model's
  believed (x,y) AND velocity stay **self-consistent / physically coherent** over the occluded steps
  ("what would it predict if revealed now", compared across steps), not exact GT match.

## In flight
**Nothing running.** 4070 idle. **Unblocked** (ESC-008 answered). Now in INSTRUMENT-DESIGN, not
experiment — designing the position-memory consistency metric (D-018).

## NEXT ACTION
Converge the D-018 metric design WITH Merlin (proposal presented this session: onset-fidelity +
self-physics-consistency + report-only GT-tracking-horizon; state-probe readout; ceiling/chance/
copy-last/matched-horizon controls). Open question to him: do we credit a self-consistent belief that
has diverged from GT (butterfly) as "memory"? Then → run the design past the `critical-claim-verifier`
agent (measurement-validity audit) → build + FREEZE the metric BEFORE any H3 position method (§8 spine).
Leading H3 position METHOD to try once position is measurable: sequential stop-grad register-relay
training (IDEAS.md, worked out with Merlin 2026-06-13).

## Recently done
- **EXP-012 (D-016)** done; confound resolved; ESC-008 RESOLVED (1)(2) agreed, (3) metric redirect.
- **T-008 (D) — KV cache (D-017)** — absolute-RoPE + generate_cached, bit-for-bit, ~2× faster. Done.
- **T-010** — play_dynamics_checkpoint carries FF7 registers (memory_rollout_init/step).

## Tooling
- **`critical-claim-verifier` agent (committed by Merlin)** — independent unbiased fact/validity check
  for difficult code or difficult architecture/objective ideas. Use it for the D-018 metric validity
  audit before freeze. Not for routine work.

## Open threads / parked
- **Sequential register-relay training** (IDEAS.md): TBPTT-1 stop-grad relay so training context
  carries real relayed memory tokens (fixes the learned-init mismatch). Leading H3 *position* method;
  starts after the metric is frozen.
- **Cross-frame KV eviction cache** + tokenizer-C cache (optional, BOARD).

## Current worries
1. **State-probe transfer (the D-018 validity risk):** a readout trained on VISIBLE hidden states may
   not transfer to OCCLUDED ones, or may decode the curtain rather than a belief. Must validate
   (held-out accuracy + a "not just the curtain" check) — this is the core verifier-audit target.
2. The motion claim is single-seed (vanilla_s0 ≈ my_dynamics ~4.5px). A 2nd vanilla seed would firm
   it if Merlin wants; he didn't ask for it.
3. FF7's base-dynamics improvement (4.6×) still conflates loss vs relay-inference — not disentangled.
