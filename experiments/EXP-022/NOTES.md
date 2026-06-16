# EXP-022 — inference-only context_signal sweep on open-loop motion (D-029)

NO TRAINING. Tests the IDEAS "uncertainty-aware rollout" lever: rollout feeds ALL context (real prefix +
self-generated) at one global signal `context_signal` (default 0.9). Sweep it at INFERENCE, measure
open-loop pos_err. Reuses src/eval/motion.open_loop_curve (overrides dyn.config.context_signal).
Provenance: existing ckpts vanilla_s0 (EXP-012), ff7_k3 (EXP-010), c1_h4_s0 (EXP-020); probe env/detector
5503e75; tokenizer trained_autoencoder.pt; 32 ep, H16, curtain-up, seed 20000+. Artifacts: sweep.json,
sweep.png, sweep.log.

## Result (decisive read) — the inference-trust lever is REAL and LARGE on competent models
Open-loop pos_err @ horizon (px), by context_signal s:
```
ff7_k3      h8    h12   h16    crossChance
 s=0.50     6.2    7.9  10.5     h=17        <- LOW trust: compounding nearly HALVED vs default
 s=0.70     6.7   10.2  13.9     h=17
 s=0.90    10.5   16.1  18.6     h=15        <- DEFAULT (what all prior open-loop evals used)
 s=0.99     9.8   14.9  19.6     h=15
c1_h4_s0
 s=0.50    13.0   17.3  20.8     h=13        <- LOW trust HURTS C1
 s=0.90     8.9   11.8  13.5     h=17        <- DEFAULT is ~OPTIMAL for C1
 s=0.99    10.3   15.3  16.7     h=17
vanilla_s0: ~flat across s (h16 ~25-29 all s) -> knob inert on a model with no good per-step map.
```

**Three findings:**
1. **ff7_k3 over-trusts its self-generated context at the default 0.9.** Lowering context_signal to
   0.5-0.7 (telling the model the context is less reliable) DRAMATICALLY reduces open-loop compounding
   (h16: 18.6 -> 10.5, ~halved) while leaving the per-step map (h1 ~1px) untouched. A free, inference-
   only, no-retrain robustness gain.
2. **C1 is already best at 0.9; the knob does NOT help it (lowering s hurts).** Expected: C1's DAgger
   loss (`_multistep_loss`) trains on its OWN context held at context_signal=0.9, so C1 has INTERNALIZED
   the robustness ff7 only gets via the inference hack. C1 and the knob are two routes to the same place.
3. **vanilla is flat in s** — the lever helps models that have a good per-step map but compound (ff7),
   not models that can't predict a step (vanilla). Confirms it's a TRUST/compounding lever, not a map fix.

Surprise: HIGH-favorable on magnitude (Merlin asked the question; the effect is bigger than I expected).

## Methodological flag
ALL prior open-loop numbers (EXP-011/018/020) used context_signal=0.9, which is SUBOPTIMAL for ff7-type
models — so the "open-loop compounding" we diagnosed is partly an inference-trust artifact that's cheaply
tunable. Relative comparisons (all at 0.9) still stand, but the magnitude of ff7's compounding was
inflated by a sub-optimal inference setting.

## Implication for the C1 story (sharpens EXP-021)
The fair competent comparison is now: **C1 (full data) at its best s  vs  ff7_k3 at its best s.** If
C1-full only matches ff7+tuned-knob, C1 is "just" learning what the knob does (less interesting). If
C1-full beats ff7+knob, C1's training adds robustness beyond trust-tuning. -> EXP-021 eval MUST sweep
context_signal for every model, not evaluate at 0.9 only. (This sweep is now a standard eval axis.)

Status: lever validated as a flat inference knob. Next: (a) per-frame / decaying-trust schedule s(j)
(needs a small generate() shim) could beat the flat knob; (b) fold the s-sweep into EXP-021's eval.
