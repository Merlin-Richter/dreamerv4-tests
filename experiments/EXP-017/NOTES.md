# EXP-017 — FF9 v2 memory-token baseline (overnight)

Decision: D-024 | launched per Merlin's overnight-run request 2026-06-14 (late-night).
Provenance: master @ d1a38d1; tokenizer trained_autoencoder.pt (frozen); data occluded.npy +
occluded_actions.npy; 4070 (venv). Config: `config.yaml`. Launch: `run.sh`.

## Purpose
The architectural baseline for the memory-token line: distinct MEMORY tokens (registers → pure scratch)
trained with the FF9 v2 memory-only-sufficiency loss (ops 1&2: write-mem←latents, read-mem→latents),
at the EXACT EXP-010(FF7)/EXP-012(vanilla) budget. NOT expected to beat FF7 on beyond-window depth
(V-T013: within-window sufficiency only; the cross-window relay = op-3 = T-014 is built on top of THIS
next). This run gives us the trained memory-token model + a clean FF9-v2 reference.

## Why this and not the relay tonight
Merlin asked for a relevant overnight run. The full detached relay (Mode B, op-3) is exactly where the
verifier (V-T014) flagged bug-prone surface (cache-under-grad graph blowup, carry-norm, anchored per-step
loss). Rushing it at midnight risked a silently-broken run wasting the night. FF9 v2 is built + smoke-green
and is the needed foundation, so it's the high-value low-risk overnight choice. Relay built carefully next.

## Status / what's deferred
- TRAIN ONLY. Checkpoint `ff9v2_s0.pt` saved every epoch (resumable / usable at any stopping point).
- **Eval DEFERRED to tomorrow** — needs `generate_full_state_memory` (the memory-carry inference path,
  analog of generate_memory). Without it, `generate()` would NOT carry memory across rollout steps →
  a probe tonight would mis-measure the model as ~vanilla. So no probe tonight.
- W&B skipped (unattended robustness); per-epoch train/val loss + the diffusion/ff9 split are in
  `train.log`.

## TODO tomorrow (on completion)
1. Build `generate_full_state_memory` + dispatch (`use_full_state_memory`) — memory-carry rollout
   (reuse memory_rollout_init/step machinery on the memory slot) + smokes.
2. Memory-sufficiency probe (PRIMARY readout: L(memory) ≪ L(no-memory) within window).
3. Frozen-probe color at n_occ {12,16,24,32,48} vs vanilla_s0 (EXP-012) + ff7_k3 (EXP-010); expect ≈FF7.
4. Reconcile (Expected/Observed/Surprise/tripwires per D-024) → present-then-stop.
5. ESC-014 (relay gradient design: tbptt-k sweep / dynamic-state probe / train-to-depth) still OPEN —
   Merlin's call before the op-3/Mode-B build.

## Reconciliation (training complete; eval still pending)
Expected (D-024): clean training; memory-sufficiency within window; no base-dynamics regression;
frozen-probe color ≈ FF7.
Observed (training): **completed all 100 epochs** in 4h18m (~155s/epoch), stable. Final total
train 0.0587 / val 0.0767 (total = diffusion + ff9; NOT comparable to vanilla's diffusion-only 0.0066).
**Loss split on the trained ckpt** (avg over 5×B32 batches of occluded; NOT a clean val split — mixes
train eps, so diffusion is optimistic): **diffusion 0.00158** (vanilla_s0 val ref ~0.0066 → base dynamics
healthy, NOT regressed; possibly FF9-sharpened à la FF7/EXP-014) | **ff9 0.046** (down from ~0.85 at init,
~18× → within-window memory IS load-bearing: memory-only prediction far beats chance).
Surprise: mild-favorable (diffusion low; memory sufficiency clearly learned). No tripwire fired so far.
Still TODO for the real verdict: (a) clean val-split diffusion vs vanilla; (b) explicit L(mem)≪L(no-mem)
gap (ablate injected memory); (c) generate_full_state_memory → frozen-probe color n_occ {12,16,24,32,48}
vs vanilla_s0 + ff7_k3. → present-then-stop after (c).

---

## EVAL COMPLETE — reconciliation (2026-06-14) — code @ 0f02f18, probe frozen 5503e75

Inference design (`generate_full_state_memory`) settled by `critical-claim-verifier` BEFORE building
(tasks/T-013-eval-inference.md; EXPERIMENTS row V-T013-eval; verdict SUPPORTED = **A1+B1**): write a
full-state memory snapshot ONCE from the observed prefix (near-clean), inject it STATIC at a τ=0
(pure-noise) source frame each rollout step — the only ops FF9 v2 was actually trained on. The
re-extract relay (B2) is the untrained op-3 and drifts (the verifier reproduced V-T014's failure);
A2 (near-clean source) gave identical recall. So A1+B1 is the FF9-faithful, fair inference.

**Expected (D-024):** clean training; within-window memory sufficiency; no base-dynamics regression;
frozen color ≈ FF7 ("flatter = bonus; worse = investigate").

**Observed:**
- **(a) No regression — IMPROVED.** Clean held-out val split (Generator(0), 5%, chunk_len 16):
  FF9 v2 diffusion **0.00172** (±0.00006) vs vanilla_s0 ~0.0066 → ~3.8× sharper (FF9-as-dynamics-
  regularizer, à la FF7/EXP-014). `primary.json`.
- **(b) Memory sufficiency (PRIMARY) — strongly load-bearing.** At τ_term=0 (whole path pure noise →
  prediction must come from memory alone): L(mem) 0.018/0.025/0.033 vs L(no-mem) 0.269/0.272/0.273
  (j=1/2/3), chance(var) 0.41, copy-last 0.38/0.63/0.69. Memory closes 88–93% of the no-mem gap, sits
  ~20× below chance, and ≪ copy-last (copy-last rises to 0.69 as the ball moves while L(mem) stays
  ~0.03) → memory is a genuine full-state object that captures MOTION within the window, not a static
  frame copy. At τ_term=0.9 the gap →~10% (target's own latent makes memory redundant) — expected shape.
  `memory_sufficiency.png`.
- **(c) Frozen color — FAR BETTER than FF7 (the "bonus" branch), not ≈FF7.** color ΔRGB vs n_occ
  (ceiling ~13, chance ~105, T-004 bar 63):
  ```
  n_occ      2    6    8   12   16   24   32   48
  FF9 v2   13.1 13.5 14.3 13.0 12.5 12.2 12.9 12.2   <- FLAT at ceiling through 48 (6x the window)
  ff7_k3   16.9 25.1 28.4 40.0 46.5 67.6 74.2 85.5   <- decays; crosses bar 63 by n_occ~22
  vanilla  14.1 14.3 93.7 102.8 108 98.9 95.3 104.3  <- cliff to chance at the window edge (n_occ 8)
  ```
  FF9 v2 occluded ≈ its matched-horizon drift at EVERY point (e.g. 12.2 vs 12.3 @48) → occlusion adds
  ZERO color loss. `frozen_color.json`, `headline_color.png`, `sheet_ff9.png` (predicted reveal ball
  matches GT color at every n_occ incl. 48).
- **Position NOT retained (as expected).** posErr ~20–30px at all n_occ for ALL three models (FF9 no
  better than vanilla/FF7); latent-MSE near chance (position-dominated, the T-004 confound). The static
  snapshot can't integrate motion — dynamic state needs op-3 (the relay, T-014).
- **Harness cross-check:** the runner reproduces published numbers at overlapping points — vanilla
  102.8/108/98.9 @12/16/24 (EXP-012: chance 105–110); ff7_k3 40.0/67.6 @12/24 (EXP-010: 40/65). The
  frozen instrument is faithfully reused (no edit; only the n_occ grid extended past 24).

**Surprise: HIGH (favorable).** D-024 predicted ≈FF7; FF9 v2 is flat at ceiling — strictly dominant on
static-color retention. Why: A1+B1 carries a written-once snapshot that CANNOT drift, so a static
attribute (color) is held perfectly forever; FF7 re-extracts its register each step (one-hop relay) and
accumulates drift → decays. Each model uses its own faithful inference, so the comparison is fair; the
deeper cause of FF9's flatness is the static-snapshot inference matching a static attribute.

**Tripwires checked (D-024 "would change my mind"):** (1) memory not load-bearing → NOT fired (88–93%
gap). (2) base regression → NOT fired (improved). (3) color materially worse than FF7 → NOT fired
(materially BETTER). No HALT condition; the surprise is favorable.

**Hypothesis impact:** H3 — the memory-token architecture cleanly retains STATIC hidden state (color)
arbitrarily far beyond the window with no drift, beating both the vanilla cliff and FF7's decay. This is
the architectural baseline D-024 set out to build, and it over-delivered on static state. It does NOT yet
solve DYNAMIC (position/motion) memory — the frozen snapshot can't update — which is exactly op-3 / the
sequential relay (T-014/ESC-014), now clearly motivated and de-risked by a working write+read substrate.

**Next:** ESCALATE (ESC-015) present-then-stop. Then Merlin's call on ESC-014 (relay gradient design)
for the dynamic-state extension.
