# EXP-018 — Motion-prediction diagnosis (D-026 / T-016 probes P1+P2)

Date: 2026-06-14. Local 4070 (venv). NO TRAINING. Frozen probe env/detector (5503e75),
tokenizer trained_autoencoder.pt. Curtain-UP (k=0) = the no-occlusion motion regime Merlin
asked about. Models: vanilla_s0 (EXP-012), ff7_k3 (EXP-010), ff9v2_s0 (EXP-017).
Script: `probe_multistep.py` (reuses EXP-011 diagnose fns). Provenance: branch
feat/motion-prediction. Purpose: CONFIRM the method-architect diagnosis (T-016) before building.

## What was measured
- **P1 (decisive):** per-horizon position error, TEACHER-FORCED (predict each frame from the
  GROUND-TRUTH context window) vs OPEN-LOOP (predict from the model's own generated context).
- **P2:** teacher-forced 1-step error while sweeping the signal level context frames are fed at
  (rollout pins this at context_signal=0.9; training saw random per-frame τ).

## Results (24-ep quick pass; 32-ep canonical run -> diagnosis.json/png, same conclusion)
Position error (px) vs horizon h. copy-last(h) = freeze at last-seen (accumulates ball drift);
chance ≈ 20px.

```
            h:     1     2     4     8    12    16    24
vanilla  TF:     5.2   4.6   4.9   4.2   4.8   4.1   4.8     (flat ~4.5 — bad map, but stationary)
         OL:     4.8   6.8  10.8  17.6  24.9  28.6  23.2     (compounds to chance)
ff7_k3   TF:     1.0   1.0   0.8   0.9   0.9   1.2   1.2     (flat ~1.0 — excellent, stationary)
         OL:     0.8   1.8   4.9  10.1  15.7  20.4  19.7     (compounds to chance by ~h16)
ff9v2    TF:     1.2   1.4   1.0   1.1   0.8   1.0   1.3     (flat ~1.1 — excellent, stationary)
         OL:     1.1   3.1   9.2  18.8  26.4  31.7  31.9     (compounds fastest)
         copylast 3.2   6.1  12.0  21.4  28.1  32.6  31.3
```
P2 τ-context sweep (vanilla, TF 1-step err): 0.50:4.7 0.70:4.7 0.80:4.7 0.90:4.5 0.95:4.8 — FLAT,
no cliff at the rollout's 0.9.

## Reconciliation (§5)
Expected (D-026): a short diagnostic confirming compounding as the dominant multi-step deficit.
Observed:
- **Teacher-forced error is FLAT in horizon for EVERY model** — given true context the per-step map
  does not degrade with depth. ff7/ff9 hold ~1px out to h=24; vanilla holds ~4.5px.
- **Open-loop error compounds from the per-step floor up to chance**, purely from feeding the model
  its own slightly-wrong outputs. Even the 1px-per-step models (ff7/ff9) hit chance by ~h12-16.
- **The entire multi-step deficit = the gap between flat-TF and rising-OL = autoregressive error
  compounding / exposure bias.** Confirmed for the good-map models.
- **P2 flat → no τ-context cliff → link-4b ruled out → C0 NOT needed.**
- Secondary: vanilla's TF floor (~4.5px, worse than copy-last) is the link-3 single-step deficit the
  FF7/FF9 aux loss already fixes (EXP-014); compounding is universal and orthogonal to it.

Surprise: none (confirms the architect's link-4 hypothesis cleanly and rules out C0).
Hypothesis impact: motion deficit is diagnosed = autoregressive compounding; the fix must expose the
model to its OWN context during training (multi-step / scheduled-sampling loss). Directly motivates C1.
Tripwires checked (D-026): architect did NOT refute compounding (it confirmed it) — no re-aim needed.
Next: verify C1 design (critical-claim-verifier, T-017) -> implement C1 -> budget-matched A/B
(vanilla control vs C1) on occluded subset, eval curtain-up open-loop. (Autonomous session: present
artifacts, do NOT halt-and-wait per Merlin's steer.)
```
