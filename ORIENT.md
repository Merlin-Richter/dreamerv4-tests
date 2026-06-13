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
**Nothing running.** 4070 idle. **Unblocked** (ESC-008 answered). INSTRUMENT-DESIGN phase:
position-memory consistency metric (D-018, spec tasks/T-011.md). Verifier audit **DONE** (V-T011)
and folded into the spec. **Blocked on Merlin's one framing lock** before build (see below).

## NEXT ACTION
**Awaiting Merlin's framing lock** (T-011 "Open question", sharpened by the audit): accept that the
metric measures *anchored physical coherence* (credits F2 = late GT-divergence; the verifier proved
the confabulation limit == the F2-forgiveness Merlin asked for), with GT-tracking-horizon as the only
near-GT constraint — OR a stricter near-GT bar (which re-introduces the butterfly penalty he rejected).
My lean: accept it. Once locked → IMPLEMENT with the 5 verifier fixes (n_occ≥8 headline floor;
non-degeneracy gate; speed fixed at env S; lengthen onset anchor; **run ceiling control FIRST on
vanilla_s0** — readout feasibility is the one thing synthetic tests couldn't establish) → GT-floor +
forgetting-surrogate calibration → FREEZE → run on vanilla_s0 + FF7 k1/k3, present-then-stop.
Readout DECIDED (Merlin, ESC-008): counterfactual reveal-decode (NOT state-probe) — sidesteps the
probe-transfer risk. Leading H3 position METHOD after freeze: sequential stop-grad register-relay
(IDEAS.md).

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
