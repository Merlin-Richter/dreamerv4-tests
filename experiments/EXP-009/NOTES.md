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

## Observed
(pending run completion — fill from results.json)

## Reconciliation
Expected: <above>
Observed: <headline numbers>
Surprise: <none|mild|high>
Hypothesis impact: <H2 ...>
Tripwires checked: D-011 — (1) recall must collapse beyond the window; (2) latent-MSE
must track color (metric_validation pearson) else escalate before pre-registering.
Next: present-then-stop (§5); pre-register T-004 criteria with Merlin before declaring
the H2 verdict.
