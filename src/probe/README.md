# Revisit-consistency probe suite (T-002, the H2/H3 spine)

Measures how well a world model recalls hidden ball **color** and **position** after the
revealing frames have left its sliding context window, as a function of occlusion length
`n_occ`, at window `N` and visible prefix `P`. See `tasks/T-002.md` for the full design
and `DECISIONS.md` D-011 for the rationale.

## Run

```bash
python src/probe/revisit_probe.py                 # full sweep, 64 episodes/n_occ
python src/probe/revisit_probe.py --dry-run       # fast smoke (tiny grid, 4 eps)
python src/probe/revisit_probe.py --out experiments/EXP-NNN/results.json
```

Outputs `results.json` (per-`n_occ` metrics + matched drift curve + controls +
metric-validation r) and `sheet.png` (GT top / prediction bottom, columns = time,
red line = context|generation boundary, yellow box = measured reveal frame) next to it.

## What it does (frozen contract)

- **Episode:** `[P up | n_occ down | 1 reveal up]`, exact GT from `OccludedBouncingEnv`.
- **Pure generation:** context = encoded visible prefix only; the model generates the
  occluded + reveal frames. The sliding window drops the prefix once the sequence
  exceeds `N`, so for `n_occ >= N-1` recall requires carrying the info forward.
- **Window N at inference:** set via `config.max_temporal_length` on the loaded model.
  RoPE is relative, so M<N needs **no retrain** (D-011). Default N=8 (trained at 16).
- **Primary metric:** latent-token MSE (predicted reveal latent vs frozen-tokenizer GT
  latent). Decomposition: ball color ΔRGB (headline) + position px (drift-confounded
  secondary). `ball_lost_rate` = no detectable ball (its own failure mode).
- **Controls:** ceiling (`n_occ=0`), chance (curtain-only context), and a
  **matched-horizon drift curve** (all-visible rollout at the same length/index) to
  difference out ordinary autoregressive drift.
- **Detector gate:** the ball detector is validated on GT frames first
  (`detector_gate.pass`); if it fails, recall numbers are NOT trustworthy.
- **Channel order:** dataset-native (BGR) end-to-end, no swap (see `probe_env.py`).

## Frozen version

Frozen at commit **f1cf860** for the H2 baseline (EXP-009). Any later change to the
probe is a logged decision (protocol §8) because it silently redefines prior results.
Defaults: `N=8, P=3, R=1, K=4 (cfg), n_occ grid {2,4,6,7,8,9,12,16,24}, 64 eps/n_occ`.
