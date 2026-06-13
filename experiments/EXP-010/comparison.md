# EXP-010 — FF7 v1 vs baseline (frozen probe 5503e75)

T-004 bar: color dRGB < 63 at n_occ in (12, 16, 24) (ceiling 15.9, chance 109.9).

## Color dRGB at the reveal frame (HEADLINE)

| series | 2 | 4 | 6 | 7 | 8 | 9 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| baseline (EXP-009) occluded | 15.8 | 15.8 | 16.8 | 94.4 | 116.0 | 113.9 | **108.4** | **100.5** | **120.3** |
| baseline (EXP-009) drift-ctrl | 17.1 | 17.9 | 19.1 | 22.9 | 22.0 | 23.6 | 24.4 | 30.5 | 39.6 |
| FF7 k=1 occluded | 17.2 | 21.2 | 27.1 | 31.7 | 37.9 | 39.0 | **52.1** | **59.0** | **79.8** |
| FF7 k=1 drift-ctrl | 14.2 | 16.5 | 20.1 | 23.0 | 24.5 | 24.6 | 27.9 | 28.2 | 32.2 |
| FF7 k=3 occluded | 17.8 | 20.1 | 24.3 | 27.7 | 31.6 | 32.2 | **39.8** | **55.1** | **65.1** |
| FF7 k=3 drift-ctrl | 13.7 | 16.8 | 20.1 | 23.3 | 25.4 | 24.7 | 24.5 | 31.8 | 36.5 |

## latent-token MSE (secondary)

| series | 2 | 4 | 6 | 7 | 8 | 9 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| baseline (EXP-009) occluded | 0.4 | 0.5 | 0.6 | 0.9 | 0.8 | 0.9 | 0.8 | 0.9 | 0.9 |
| baseline (EXP-009) drift-ctrl | 0.4 | 0.5 | 0.5 | 0.6 | 0.6 | 0.6 | 0.7 | 0.8 | 0.8 |
| FF7 k=1 occluded | 0.4 | 0.6 | 0.6 | 0.7 | 0.7 | 0.7 | 0.8 | 0.7 | 0.8 |
| FF7 k=1 drift-ctrl | 0.3 | 0.4 | 0.5 | 0.6 | 0.6 | 0.6 | 0.6 | 0.7 | 0.7 |
| FF7 k=3 occluded | 0.4 | 0.6 | 0.7 | 0.7 | 0.7 | 0.8 | 0.8 | 0.8 | 0.8 |
| FF7 k=3 drift-ctrl | 0.2 | 0.3 | 0.4 | 0.5 | 0.5 | 0.5 | 0.6 | 0.6 | 0.7 |

## Controls per series (own rollout path — checks base-quality tripwire D-014)

| series | ceiling dRGB | chance dRGB | ceiling MSE | chance MSE | ball_lost max | detector gate |
|---|---|---|---|---|---|---|
| baseline (EXP-009) | 15.9 | 109.9 | 0.27 | 0.88 | 0.00 | PASS |
| FF7 k=1 | 9.3 | 100.6 | 0.07 | 0.90 | 0.00 | PASS |
| FF7 k=3 | 9.0 | 106.7 | 0.06 | 0.91 | 0.00 | PASS |