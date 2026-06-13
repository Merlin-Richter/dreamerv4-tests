# EXP-013 — Position-memory consistency metric (T-011 / D-018): build, validate, apply

Date: 2026-06-13. Local 4070 (venv). Frozen probe env/detector 5503e75. Tokenizer
trained_autoencoder.pt. Models: vanilla_s0 (EXP-012, H3 baseline), ff7_k1_s0, ff7_k3_s0 (EXP-010).
Metric: `src/probe/position_consistency.py`. Present-then-stop → ESC-009.

## What this is
The position-memory metric Merlin redirected us to in ESC-008 (open-loop GT-matched error was
rejected). Reads the model's belief via the **counterfactual reveal** (inject curtain-UP for one
frame, decode, gated `detect_ball`), swept over occlusion horizons k=1..K so consecutive reveals
are the belief at consecutive blind timesteps. Scores (a) onset GT-anchor, (b) best-fit
constant-speed billiard residual (GT-free self-consistency), (c) report-only GT-tracking horizon.
Framing LOCKED (Merlin): anchored-physical-coherence, F2 forgiven. Audited by
critical-claim-verifier (V-T011) before build; 5 fixes folded in.

## Instrument validation (passed — the weak result below is the MODELS, not the metric)
- **Synthetic calibration reproduces V-T011 exactly** (`prod_calibrate.json`): billiard residual
  GT 0.77 (floor), F2-bounce 0.79 (passes), hallucinated 0.78 but onset 22px (caught by anchor),
  frozen 6.6 / shuffled 10.8 / smooth-drift 4.9 (forgetting ≫ floor). Speed-fixed is load-bearing.
- **Readout faithfulness confirmed:** FF7_k3's belief at the FIRST blind step = **1.9px** (seed
  4000) / 4.0px (20-seed mean) — reproduces its known ~1px short-horizon skill. So the
  reveal-decode readout is faithful; `found_rate=1.00` (a ball is always rendered & detected).

## The result (decisive read) — position memory through TRUE (blind) occlusion is near-absent
Per-k belief-vs-GT error, mean / 20 seeds (`per_k_curve.json`, `headline_position.png`):

```
            k=1  k=2  k=3  k=4  k=5   k=8   k=12
copy-last   5.7  8.5 11.1 13.8 16.3  23.4  30.7   (freeze ball @ last-seen position)
vanilla_s0  5.7  8.6 12.3 14.6 18.5  21.8  20.6
ff7_k1      5.3  7.3 10.8 12.0 15.8  18.9  21.3
ff7_k3      4.0  7.9 11.9 14.5 19.4  23.8  24.5
```
Billiard self-consistency residual (mean, trustworthy n_occ≥8): vanilla 14.85, ff7_k1 10.32,
ff7_k3 12.64 (floor 0.77; forgetting surrogates 4.9–10.8). Coherence horizon (<6px): all ~1 step.

1. **vanilla_s0 ≈ copy-last** at k=1–4 — it freezes the ball at its last-seen position, i.e. ZERO
   useful motion propagation through blind occlusion. (Residual 14.85 > even the frozen surrogate
   6.6 — a weak open-loop model actively *wanders* off any physical path, worse than freezing.)
2. **FF7 retains only marginally more than freezing.** ff7_k1 sits consistently ~1–3px BELOW
   copy-last at every horizon (small but real position retention beyond freezing). ff7_k3 has the
   best single blind step (4.0 vs 5.7px) then decays to ≈copy-last. Both lose coherence (belief
   teleports, jumps ≫ speed 2.92) by k≈5.
3. **This corrects the EXP-011 optimism.** EXP-011's "FF7 tracks ~12 steps open-loop" was
   **curtain-UP** rollout — the model sees its own generated ball each step (visual feedback).
   Under genuine occlusion (no feedback), the blind dead-reckoning horizon is ~1–4 steps (lucky
   seeds) / ~1 step (mean). The metric measures the right thing (blind memory), and it is weak.
4. **H3 story, sharpened:** the FF7 register relay carries STATIC color indefinitely (EXP-010) but
   does NOT carry usable position+velocity through blind occlusion. The marginal FF7>copy-last
   position signal (esp. k1) is real but tiny — the relay propagates a little motion, not a
   sustained trajectory.

## Honest caveats / methodology
- **Single billiard residual is a poor headline** — it averages the coherent-early window with the
  incoherent-late tail into one muddy number. The per-k curve + copy-last reference + coherence
  horizon is the right summary (and the residual still ranks correctly: ff7_k1<ff7_k3<vanilla).
  Proposing coherence-horizon as the frozen headline (pending Merlin, ESC-009).
- **Late-tail artifact (Merlin's flag, observed):** beyond k≈5 vanilla dips BELOW copy-last
  (k=12: 20.6 vs 30.7) because a wandering/desynced prediction coincidentally re-approaches GT as
  copy-last drifts monotonically away (bounded box + bounce-back). So ONLY k≤~5 is trustworthy for
  all series; the coherence horizon (k≤~1–2) is the clean regime.
- ff7_k1 vs ff7_k3: k1 is the steadier position retainer here (lowest residual, below copy-last
  throughout); k3 wins only the first blind step. (k3 won COLOR in EXP-010.)

## Files
- `headline_position.png` (the curve), `per_k_curve.json` (curves incl. copy-last),
  `vanilla_s0_posmem.json` / `ff7_k1_posmem.json` / `ff7_k3_posmem.json` (full metric per model),
  `../verify-T011-scorer/prod_calibrate.json` (instrument validation).
- Metric: `src/probe/position_consistency.py`. Spec: `tasks/T-011.md`.
