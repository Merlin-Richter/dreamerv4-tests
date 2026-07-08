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

## RESULTS (2026-07-08 ~15:15, mid-run @ step ~81k — snapshots 250..80000 analyzed)

Figure: `curve_pixcurve.png` (plot_curve.py; loss CSVs from both logs + sheets_sweep.log).

1. **Cross-backend anchor VALIDATED**: H100 and 4070 loss curves coincide through the
   local kill point (val @625: 0.00564 vs 0.00559; the flow curves overlay exactly on
   the log-log plot). Same seed => same data order; silicon is irrelevant. Snapshot
   3750 is a faithful continuation point.
2. **Loss knee**: flow 0.0122 @3750 -> 0.0075 @10k -> 0.0060 @28k -> 0.0056 @80k —
   plateau, on the 3e-4 flat (NOT annealing). val(normal) bottoms ~0.0032 @3k then
   drifts up to ~0.0055 (never-trained normal loss; expected for rollout-only).
3. **Revisit acc (driver/sheets.py, seeds 5+6, chance 0.2): WINDOW-SHAPED, NOT
   MEMORY-SHAPED.** Mean: ~chance to 2k, lifts at 3750 (0.29), plateaus 0.32-0.35
   from 10k -> FLAT to 80k. Per-frame: FIRST revisit frame (age ~60, nearest the
   window) climbs 0.33 -> 0.80-0.87 by 10k; LAST frame (age ~190) stays at chance
   (0.07-0.27) at EVERY snapshot incl 80k. The lift is all near-age frames — the
   exact "window-shaped gains" signature the leaderboard legibility backstop
   (autoresearch-harness task, WINDOW PIN §3) was designed to catch.
4. **VERDICT (pre-registered Q3): pixel-tier memory does NOT emerge by 80k steps**
   (~21x the local partial; loss+acc both flat 10k->80k, so the remaining ~30k steps
   will not change it). Pixel tier confirmed as promotion gate at this scale — the
   sym-tier-first strategy stands. Caveat: sheet numbers are 1 scripted episode x 2
   map seeds (illustrative); an eval-grade claim wants the frozen comeback eval
   (chance-corrected age bins) on ~3 snapshots (e.g. 3750/20k/final), reduced config.

## Ops
- Job: ferranti H100, name `colorfield-pixcurve`, 1 GPU, --hours 6.
- Pull: `scripts/pull_results.sh --cluster ferranti colorfield-pixcurve --what all`
  (checkpoints are small: 1.32M params ~ 5-6 MB each, 11 snapshots + final).
- SHA + JOB_ID recorded in agent/EXPERIMENTS.md (row `colorfield-pixcurve-h100`).

- 2026-07-08 15:17: job 416906 CANCELLED by Merlin at 3h04m (~step 82k) — verdict already unambiguous from the 10k-80k plateau; snapshots 250..80000 + dynamics.pt (~81k) are the final artifacts (pulled local, .pt gitignored).
