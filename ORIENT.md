# ORIENT.md

Rewritten: 2026-06-13 (EXP-013 DONE → position memory near-absent; blocked on ESC-009. Parallel orch: EXP-014/D-019)

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
**Nothing of mine running.** 4070 free (modulo a PARALLEL orchestrator running EXP-014 — the
loss-vs-relay-vs-window 1-step disentangle, D-019; I'm hands-off it, path-scoped commits only).
EXP-013 DONE → **blocked on Merlin's ESC-009 verdict** (present-then-stop).

## NEXT ACTION
**Awaiting Merlin's ESC-009 verdict.** EXP-013 built+validated+applied the position-memory metric.
Decisive read: **blind position memory is near-absent** — vanilla_s0 ≈ copy-last (freezes ball, zero
motion propagation); FF7 retains only marginally more (ff7_k1 <copy-last throughout; ff7_k3 best 1st
blind step then ≈copy-last); EXP-011's "~12 step tracking" was curtain-UP (visual feedback), blind
horizon ~1–4 steps. Register relay carries static COLOR not dynamic POSITION. Metric validated
(calibration==V-T011; readout faithful FF7 k=1=1.9px) — weak result is the models, not the instrument.
Questions to Merlin (ESC-009): (1) agree with read; (2) freeze metric with coherence-horizon headline
(not the muddy single billiard residual); (3) proceed to the **sequential stop-grad register-relay
training** (IDEAS.md) as the H3 position METHOD — the natural fix since FF7's single-timestep loss
teaches color-carry not motion-carry — or wait for EXP-014's loss-vs-relay verdict first.

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
1. **Readout feasibility (model-side, UNVERIFIED — verifier C4):** the whole reveal-decode readout
   rests on the model rendering a clean detectable ball on the 1-frame counterfactual reveal when it
   HAS the info. Env-side is proven (curtain action absolute/Markov; generate takes arbitrary actions),
   but the ceiling control must be run on vanilla_s0 first; if it fails, fall back to a state-probe.
   (The earlier state-probe-transfer worry is moot — reveal-decode readout chosen instead.)
2. **Confabulation limit (verifier C2):** the metric credits anchored-physical-coherence, not correct
   dead-reckoning, past the bounce-ambiguity horizon. Equals the F2-forgiveness Merlin wanted, but the
   claim language must accompany every position number. Pending his framing lock.
3. The motion claim is single-seed (vanilla_s0 ≈ my_dynamics ~4.5px). A 2nd vanilla seed would firm
   it if Merlin wants; he didn't ask for it.
4. FF7's base-dynamics improvement (4.6×) still conflates loss vs relay-inference — not disentangled.
