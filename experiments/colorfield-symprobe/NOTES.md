# colorfield-symprobe — SYM tier 10-minute H100 budget probe

Launched 2026-07-08 (Merlin: pixel at 10-min H100 is "barely in reach" — is the simpler
sym tier better suited?).

## Context / priors
- Pixel H100 (colorfield-pixcurve, job 416906): 0.124 s/step at 1.32M/bs128 => 10 min
  ~ 4800 steps; pixel appearance knee ~3k, revisit-acc plateau ~10k, memory absent to 80k.
- Sym local (sym20 probe, 4070): 237 steps in 20 min (5.08 s/step, dataloader-bound —
  per-clip on-the-fly grid render + encode), yet quality ≈ pixel @ 2000-3750 steps.
  Sym wins on task difficulty per step; its H100 pace is the open number.
- Local sym is NOT viable at 10 min (~118 steps). The H100 question decides the backend.

## Design
- Same model/config as sym20 (1.31M = 128/6/8, bs128, fixed n_ctx, seed 0) => comparable.
- Datagen on cluster, gated on byte-identity with the local sidecars (sha256 of
  actions.npy, both splits — recorded in editable/BUILD_NOTES.md).
- Pace probes first (60s, workers 0 vs 8) — sym is CPU-bound, so DataLoader workers may
  matter more than the H100 itself; job submitted with --cpus 16 to give workers room.
- Real run: --budget-s 600 (train_sym BUDGET_STOP owns termination), --sched-steps from
  measured pace ×0.95 (the driver's intended sizing mechanism), snapshots
  100/**237 (= sym20 final step, same-step anchor)**/500/1000/2000/4000/8000/16000.
- sheets_sym rendered in-job (non-fatal if cv2 missing); reduced frozen_sym eval runs
  locally after pull.

## Pre-registered readouts
1. Sym H100 steps/min at workers 0 and 8 -> the backend calibration table.
2. Steps reached in 10 min; where that lands on the sym quality curve (sheets_sym acc
   at snapshots vs the sym20 checkpoint at 237 steps).
3. Verdict input: does a 10-min H100 sym run reach the "interesting" regime (crisp
   frames + scroll logic + partial band tracking, i.e. >= sym20-at-237 quality, ideally
   well past it) with headroom for the loop to search? Compare against pixel-at-10-min.

## Ops
- Job: ferranti H100, name `colorfield-symprobe`, --hours 2 (self-stops ~20-25 min).
- Pull: `scripts/pull_results.sh --cluster ferranti colorfield-symprobe --what all`.
- SHA + JOB_ID in agent/EXPERIMENTS.md (row `colorfield-symprobe-h100`).

## RESULTS (2026-07-08, job 417029 COMPLETED rc=0)

- **Datagen determinism: byte-identical on cluster** (sha256 gate passed both splits).
- **Pace: workers0=305 / workers8=329 steps/min** (0.19 s/step) — the data path is NOT
  a bottleneck (8% delta); the local "dataloader-bound" note in BUILD_NOTES is REFUTED
  (measured per-clip __getitem__ = 0.47 ms; batch-128 assembly ~60 ms).
- **10-min budget = 3109 steps** (13x the entire local 20-min sym20 probe). Sched 3125
  sized from measured pace landed exactly on the cosine tail (final lr 1.5e-6).
  Flow 0.171-region -> 0.0106, still descending at stop.
- **Sheet curve (viewport cell acc, seeds 5+6, chance ~0.17)**:
  step 100 mean 0.18 (garbage) | 237 first 0.52-0.60 mean 0.21 | 1000 first 0.48-0.60 |
  2000 first 0.48-0.68 mean 0.20-0.31 | **3109 first 0.68 / mean 0.30-0.34 / last 0.12**.
  At the 237 anchor the H100 run ≈ the local sym20 probe (first 0.44-0.6) — consistent.
- **VERDICT INPUT: sym at 10 H100 minutes is squarely in the interesting regime** —
  well past garbage, loss still descending, far from ceiling (mean 0.3 vs 1.0), i.e.
  HEADROOM + room for the loop to find optimizations. Contrast pixel-at-10-min
  (~4800 steps): right at the start of its revisit plateau with long-age memory absent.
  Next per the task file: reference arms (vanilla --n-memory 0) + seed-noise sigma at
  this budget -> keep-rule; then the frozen_sym comeback eval numbers.
