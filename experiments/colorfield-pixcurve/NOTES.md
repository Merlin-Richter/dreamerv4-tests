# colorfield-pixcurve — pixel-tier steps-vs-quality curve, H100 extension

Launched 2026-07-08 (Merlin's ask: "a useful ~4h cluster job on the new envs").

## Why
- The local pixel calcurve (autoresearch/runs/calcurve/, 1.32M = 128/6/8, bs128, fixed
  n_ctx, seed 0, sched 7000) was OOM-killed at step ~3750. Partial finding: appearance
  knee ~3k steps, **memory absent at 3750** — pixel tier confirmed as promotion gate.
  Open question: does memory EVER emerge at the pixel tier, and at what step count?
- The harness calibration go/no-go still needs the **H100 s/step** number for the
  4070-vs-H100 backend decision (local anchor: 3.92 s/step at this exact config).

## Design
- Same config + seed as the local calcurve, so snapshots are directly comparable;
  snapshot **3750** is the cross-backend anchor against the local kill point.
- `--budget-s 14400` (4h) — train.py's own BUDGET_STOP ends the run; SLURM walltime
  (6h) is only the outer margin, so the job cannot be walltime-killed mid-write.
- **MEASURED H100 PACE (from the first submission 416895, cancelled at ~2 min):
  0.124 s/step at this config (step 250 @ 35s incl setup, step 500 @ 66s) = ~31x the
  local 3.92 s/step — far beyond the 4-6x planning guess. Calibration number in hand.**
- Resize: `--sched-steps 110000` (~3h48m of schedule; warmup = min(200, 10%) = 200,
  identical to the local run, so early snapshots stay LR-matched on the 3e-4 flat) +
  `--epochs 200` (125k-step cap > expected ~115k) so the 4h budget is the binding stop.
  The original sched 14000 + default epochs 50 would have exited EPOCHS_DONE at ~65 min.
- Snapshots: 250,500,1000,2000,3750,6000,8000,10000,14000,20000,28000,40000,56000,80000,110000
  -> runs/colorfield-pixcurve/dynamics_step{N}.pt on the cluster (~5-6 MB each).

## Pre-registered readouts (when pulled)
1. ~~H100 s/step~~ DONE early: **0.124 s/step** (31x local) from the cancelled first
   submission — already actionable for the backend decision (a 10-min H100 budget
   ≈ 4800 steps ≈ well past the local appearance knee).
2. Same-step sanity anchor: step-2000/3750 snapshots vs the local ones (same seed,
   same data order by construction — loss prints should track closely).
3. Morning-after pass: driver/sheets.py + reduced frozen eval per snapshot -> extend
   the pixel curve past 3750; look for ANY memory signal (comeback bins > chance) out
   to 110k steps. If memory never appears even at 110k (~85x the local partial), that
   decisively quantifies the pixel promotion-gate cost and strengthens the
   sym-tier-first strategy; if it DOES appear, the step count locates the memory knee.

## Ops
- Job: ferranti H100, name `colorfield-pixcurve`, 1 GPU, --hours 6.
- Pull: `scripts/pull_results.sh --cluster ferranti colorfield-pixcurve --what all`
  (checkpoints are small: 1.32M params ~ 5-6 MB each, 11 snapshots + final).
- SHA + JOB_ID recorded in agent/EXPERIMENTS.md (row `colorfield-pixcurve-h100`).
