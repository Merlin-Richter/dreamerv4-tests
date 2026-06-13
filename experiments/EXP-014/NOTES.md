# EXP-014 — Disentangle FF7's base-dynamics gain: LOSS vs RELAY-INFERENCE (D-019)

**Type:** analysis-only, no training. Existing checkpoints + frozen probe env/detector (5503e75).
**Provenance:** code @ master (this session, pre-commit of D-019); tokenizer `trained_autoencoder.pt`;
dynamics ckpts `experiments/EXP-012/vanilla_s0.pt`, `experiments/EXP-010/k{1,3}/ff7_k{1,3}_s0.pt`.
Script: `disentangle.py`. Episodes seeded 20000+i (same as EXP-011/012 teacher-forced eval), 32 eps,
horizon 24, N=8, P=3, K=4. Device: GPU (venv).

## Purpose
ORIENT worry #4 / EXP-012 "bonus finding": FF7 sharpens 1-step teacher-forced pos_err ~4.6×
(vanilla_s0 4.66 ≫ ff7 ~1.0px), but that ~1px was produced through the **register-relay** inference
path (`generate()` dispatches `use_register_memory=True` → `generate_memory()`, dynamics_model.py:528),
a **window-1** relay, NOT the ≤7-frame windowed attention the vanilla_s0 number used. The ~1px conflated
(i) FF7-loss weights, (ii) the relay, (iii) window size. This separates them by running each model's
1-step teacher-forced prediction through BOTH inference paths on the IDENTICAL GT window.

## Results (32 eps, horizon 24)

| model | vanilla path (windowed, NO relay) | relay path (window-1 + carried reg) |
|---|---|---|
| vanilla_s0 | **4.73 px** (med 4.67) | 5.34 px (med 5.63) |
| ff7_k1 | **1.63 px** (med 1.37) | 1.00 px (med 0.90) |
| ff7_k3 | **1.04 px** (med 0.94) | 0.99 px (med 0.89) |

copy-last (freeze ball) = GT 1-step displacement = 3.19 px. ball_lost_rate = 0.00 everywhere.

Sanity anchors (harness validated): ff7_k3 relay 0.99 ≈ EXP-012 0.96; ff7_k1 relay 1.00 ≈ EXP-012 1.02;
vanilla_s0 windowed 4.73 ≈ EXP-012 4.66. The relay-path column reproduces EXP-011/012's FF7 numbers,
and the vanilla path reproduces vanilla_s0 — so cross-path differences are real, not harness drift.

## Reconciliation
**Expected (D-019):** FF7-vanilla-path much better than vanilla_s0's 4.66, i.e. the bulk of the 4.6× is
the loss; a residual relay advantage from window-1 register sufficiency.
**Observed:** Confirmed, and the loss share is even larger than I hedged.
- **FF7 LOSS is the dominant factor.** FF7 weights through the plain windowed path with NO relay
  (learned-init scratch registers, same ≤7-frame window as vanilla_s0): k3 **1.04 px**, k1 1.63 px,
  vs vanilla_s0 **4.73 px**. That is a 4.5× (k3) / 2.9× (k1) improvement in *windowed* 1-step dynamics
  attributable to the FF7 objective alone — zero relay involved. This is exactly the forward the FF7
  model's own main diffusion loss uses (no `register_in`), so it is a fair "what did the loss buy" read.
- **The relay is a SECONDARY, arm-dependent contributor.** k3: 1.04→0.99 (negligible, ~5%); k1:
  1.63→1.00 (the relay closes k1's residual gap). Interpretation: k1's shorter supervised lookahead
  left weaker windowed weights, which the window-1 register-sufficiency inference compensates; the
  better-trained k3 already gets there from the weights, so the relay adds almost nothing at 1-step.
- **The relay needs the FF7 loss.** vanilla_s0 forced through the relay path = 5.34 px, *worse* than its
  own windowed 4.73 — the window-1 + carried-register inference hurts weights not trained for it. So the
  relay does not rescue non-FF7 weights; the FF7 objective is required.

**Surprise:** mild-favorable. The loss share is bigger than expected (k3 relay barely beats k3 vanilla).
**Hypothesis impact:** Resolves worry #4 and *confirms+sharpens* the EXP-012 bonus claim — "FF7 sharpens
base 1-step dynamics" is TRUE and is primarily the **single-timestep-sufficiency objective acting as a
dynamics regularizer on the windowed weights**, not an inference-path artifact. Relevant to H3: the FF7
loss has value beyond memory-relay (a cleaner base dynamics model), and a follow-on memory method
(sequential relay) should expect most 1-step accuracy from the objective, with the relay buying retention
(color beyond-window, EXP-010), not raw 1-step accuracy.
**Tripwires checked (D-019):** (1) FF7-vanilla ≈ vanilla_s0 → did NOT fire (1.04 ≪ 4.73), no retraction.
(2) relay-path FF7 fails to reproduce EXP-012 → did NOT fire (anchors match). (3) vanilla_s0-relay ≈ 1px
→ did NOT fire (5.34px; relay needs FF7 loss). All clear.
**Caveats:** single-seed weights per arm (EXP-010/012 are seed 0); this is 1-step teacher-forced only
(does not speak to open-loop compounding or occluded retention — those are EXP-011/EXP-010's domain and
the D-018 metric's). Conclusion is about *1-step dynamics quality attribution*, nothing more.
**Next:** present-then-stop → ESC-010. No further action presumed pending Merlin's read.
**Complementary to EXP-013 (landed same day, other orchestrator):** EXP-013's blind-occlusion
position-memory metric found vanilla ≈ copy-last (no motion propagation) and FF7 retaining only
marginally more. Together with this: the FF7 loss buys a clean *1-step* dynamics model (≈1px, this
EXP), but that 1-step quality does NOT translate into dynamic *position* memory through true occlusion
(EXP-013) — it relays static color, not motion. The two findings are consistent and reinforce each other.

## Access points
- Headline chart: `experiments/EXP-014/headline.png` (grouped bars, vanilla vs relay, with copy-last +
  vanilla_s0 reference lines).
- Numbers: `experiments/EXP-014/results.json`. Script: `disentangle.py`. Chart: `make_png.py`.
