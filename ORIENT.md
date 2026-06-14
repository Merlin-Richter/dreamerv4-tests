# ORIENT.md

Rewritten: 2026-06-14 (EXP-017 FF9 v2 full eval DONE → ESC-015 present-then-stop, awaiting Merlin)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75; H2 anchored on budget-matched `vanilla_s0`.
- **H3 — memory-token line. STATIC hidden state (color) now SOLVED cleanly; DYNAMIC (position) still open.**
  - FF7 (registers, single-timestep loss): carries static color beyond window but decays; loss is also a
    1-step dynamics regularizer (EXP-010/012/014).
  - **FF9 v2 (distinct MEMORY tokens + leak-free memory-only sufficiency loss), EXP-017 — JUST EVALUATED:**
    perfectly retains static COLOR FLAT at ceiling through n_occ=48 (6× the window), strictly beating FF7
    (decays) and vanilla (cliff). The chosen architectural baseline over-delivered on static state.
  - **Position/motion is NOT retained by ANY method yet** (posErr ~chance through occlusion) — the frozen
    snapshot can't integrate motion. That is op-3 / the sequential relay's job (T-014, ESC-014).
- **Big-picture plan (Merlin):** experiment with architectures + objectives to find persistent memory;
  keep what sticks. Memory-token line is now validated for static state; dynamic state is the next frontier.

## In flight
**Nothing running. 4070 idle.** EXP-017 fully done (train + eval). No cluster (scripts/ deferred).

## NEXT ACTION
**WAIT for Merlin on ESC-015 (present-then-stop for EXP-017) — blocking per §5.** Do not start the relay
build, ESC-014 probes, or any follow-up until he weighs in.

EXP-017 decisive read (full detail in ESC-015 + experiments/EXP-017/NOTES.md):
- Beyond-window color ΔRGB FLAT 12–14 across n_occ 2→48 (ceiling ~13, chance ~105, T-004 bar 63);
  occluded ≈ drift (occlusion adds 0 color loss). FF7 17→85; vanilla cliff→chance@8.
- PRIMARY memory sufficiency: L(mem) 0.018–0.033 vs L(no-mem) 0.27 (88–93% gap, captures motion).
- No regression: val diffusion 0.00172 vs vanilla 0.0066 (~3.8× sharper).
- Position NOT retained (the honest caveat) → needs op-3.
- Inference design (`generate_full_state_memory` A1+B1) settled by critical-claim-verifier first (V-T013-eval).
- All 3 D-024 tripwires clear; HIGH favorable surprise (D-024 predicted ≈FF7; got strictly better).
- Views: experiments/EXP-017/{headline_color.png, memory_sufficiency.png, sheet_ff9.png}. Code @ 0f02f18.

## TWO open escalations for Merlin
- **ESC-015 (NEW, blocking):** the EXP-017 present-then-stop above. Q: agree with the read / call the
  memory-token baseline a success / direction for the dynamic-state extension.
- **ESC-014 (still OPEN):** the op-3 relay gradient design (the dynamic-state method). Verifier V-T014
  REFUTED pure detached carry (drifts past trained depth). Options P-a (tbptt-k sweep) [lean] / P-c
  (dynamic-state probe) / P-b (train-to-depth detach). My lean P-a+P-c before the Mode B build. This is
  the natural next step once ESC-015 is acknowledged — it is what carries DYNAMIC state.

## Recently done
- **EXP-017 eval (D-024) — DONE 2026-06-14.** generate_full_state_memory (A1+B1, verifier-vetted) +
  dispatch in all 4 generate entry points; FF9 smokes 9/9 (+2 new) + FF7/KV/stream gates green. Primary
  readouts (eval_primary.py) + frozen color sweep (frozen_color.py, frozen instrument reused unmodified,
  grid extended past 24) + views. Reproduces vanilla/ff7 at overlapping n_occ (harness validated).
- T-012/D-020 cross-frame KV eviction cache (the rollout substrate); EXP-015/016 perf (efficiency
  subobjective closed, ESC-011/012). EXP-013 position metric (built, uncertain strength, not a gate).

## Open threads / parked
- **op-3 / sequential relay (dynamic-state memory)** — the H3 position frontier; blocked on ESC-014.
- EXP-013 position metric: built, uncertain strength, parked (ESC-009). Revisit if a position method needs it.
- 2nd FF9 seed to firm the flat-color claim (optional, on promise — Merlin relaxed the 2-seed order).
- Tokenizer-C KV cache (optional, BOARD).

## Current worries
1. **FF9's flat color is partly the static-snapshot inference (B1) matching a static attribute** — it
   cannot drift, so color is trivially perfect. This is fair (each model uses its faithful inference) and
   genuinely strong for static state, but it is NOT evidence the approach carries DYNAMIC state. Position
   confirms it doesn't yet. Framed honestly in ESC-015; the real test is op-3 on position.
2. Single seed. The result is large and clean (flat at ceiling, occluded≈drift) so seed-fragility is
   unlikely, but a 2nd seed would firm it if Merlin wants it before building on top.
