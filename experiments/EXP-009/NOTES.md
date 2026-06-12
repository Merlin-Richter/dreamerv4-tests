# EXP-009 — H2 baseline on the frozen revisit-consistency probe

Decision: D-011. Probe frozen at commit f1cf860 (`src/probe/`, see `src/probe/README.md`).
Run: `python src/probe/revisit_probe.py --out experiments/EXP-009/results.json`
Models: tokenizer `trained_autoencoder.pt` (frozen), dynamics `my_dynamics.pt`.
Config: N=8, P=3, R=1, K=4, n_occ grid {2,4,6,7,8,9,12,16,24}, 64 episodes/n_occ.

## Expected (written before reading results, from D-011 + the agreed metric framing)
- Color recall (ball ΔRGB) near the ceiling for n_occ < ~7 (prefix still in window),
  then a sharp rise toward the chance floor for n_occ >= ~8 (prefix scrolled out) —
  the H2 cliff at the window size. Differenced against the matched drift curve, the
  effect should be specific to occlusion (drift color ΔRGB stays low throughout).
- latent-MSE rises above the matched drift curve once n_occ crosses the window.
- Position is expected to be drift-confounded (occluded ~ drift), so reported as a
  secondary, not the headline.
- Detector gate must pass (else numbers untrustworthy).
This is the H1-baseline (vanilla sliding-window) model, which by construction has NO
state beyond its window — so the cliff is expected and would *support* H2's premise and
calibrate the instrument for later H3 method comparisons.

## Observed (results.json @ f1cf860, 64 eps/n_occ, N=8, P=3)
Controls: ceiling color ΔRGB **15.9** / latent-MSE 0.266 ; chance color ΔRGB **109.9** /
latent-MSE 0.875. (ceiling = full-visible rollout; chance = curtain-only context.)

Color ΔRGB (headline) vs matched-horizon drift, by n_occ:
| n_occ | 2 | 4 | 6 | **7** | 8 | 9 | 12 | 16 | 24 |
| occluded | 15.8 | 15.8 | 16.8 | **94.4** | 116.0 | 113.9 | 108.4 | 100.5 | 120.3 |
| drift    | 17.1 | 17.9 | 19.1 | 22.9 | 22.0 | 23.6 | 24.4 | 30.5 | 39.6 |

- **Sharp cliff between n_occ=6 and n_occ=7.** For n_occ<=6 occluded ΔRGB sits at the
  ceiling (~16, at or below the drift curve → no recall deficit). At n_occ>=7 it jumps to
  the chance floor (~110) and stays there. The drift curve rises only gently (17→40), so
  the effect is occlusion-specific, not ordinary autoregressive drift.
- **The cliff location is exactly the sliding-window geometry.** Window N=8 ending at the
  reveal index covers [i-7, i]; the last prefix (color-carrying) frame is index 2; it is
  in-window iff reveal_index = P+n_occ = 3+n_occ <= 9, i.e. n_occ <= 6. At n_occ=7 the
  prefix scrolls out and recall is gone. Predicted boundary == observed boundary.
- latent-MSE mirrors color: 0.41–0.57 (n_occ<=6) → 0.79–0.92 (n_occ>=7, ≈ chance 0.875).
- Position is drift-confounded as predicted: occluded 10–25px vs drift 9–29px (overlapping).
- ball_lost_rate = 0 everywhere → it always renders a ball, just the wrong color (a clean
  hallucination, not a detector failure).
- Detector gate PASS: pos p99 0.65px, ΔRGB p99 0.0, miss 0.0 → instrument trustworthy.
- Metric validation: Pearson(latentMSE, colorΔRGB)=**0.952**, (latentMSE, posErr)=0.847.
- Visual: experiments/EXP-009/sheet.png (GT top / prediction bottom). n_occ=2,6 colors
  match; n_occ=8 GT magenta → predicted green (the cliff). n_occ=12,24 single samples
  coincidentally hallucinate near-GT colors — the 64-ep aggregate above is the evidence,
  not the sheet.

Note on exit code: the background pipeline reported exit 1, but that was `tee` failing to
open a (then-missing) log file; `main()` completed and wrote both artifacts. No probe bug.

## Reconciliation
Expected: color recall at ceiling for n_occ < ~7, sharp rise to chance for n_occ >= ~8;
latent-MSE above drift past the window; position drift-confounded; detector gate passes.
Observed: exactly that, with the cliff at n_occ=7 matching the N=8/P=3 geometry to the frame.
Surprise: none (textbook confirmation of the corrected sliding-window mechanism, D-011).
Hypothesis impact: H2 (a sliding-window WM cannot recall hidden state once evidence leaves
the window) — **supported** by this baseline; the instrument is calibrated for H3 method
comparisons. Not declaring the verdict unilaterally — present-then-stop gate (§5).
Tripwires checked: D-011 — (1) recall collapses beyond the window: YES (cliff at n_occ=7);
(2) latent-MSE tracks color: YES (r=0.952) → latent-MSE confirmed valid as the headline,
no escalation needed before pre-registering.
Next: present-then-stop (§5); pre-register T-004 criteria with Merlin (color-recall headline,
position confounded secondary, latent-MSE validated) before declaring the H2 verdict and
before any H3 method run.
