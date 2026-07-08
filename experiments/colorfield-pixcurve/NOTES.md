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
- `--sched-steps 14000`: sized to the H100 estimate (~4x local => ~1 s/step => ~14k
  steps in 4h). If the node is slower the run dies mid-flat like the local one did —
  the snapshot curve is the deliverable, not the final ckpt.
- Snapshots: 250,500,1000,2000,3750,4000,6000,8000,10000,12000,14000
  -> runs/colorfield-pixcurve/dynamics_step{N}.pt on the cluster.

## Pre-registered readouts (when pulled)
1. H100 s/step at bs128 / 1.32M (from snapshot elapsed prints) -> calibration log.
2. Same-step sanity anchor: step-2000/3750 snapshots vs the local ones (same seed,
   same data order by construction — loss prints should track closely).
3. Morning-after pass: driver/sheets.py + reduced frozen eval per snapshot -> extend
   the pixel curve past 3750; look for ANY memory signal (comeback bins > chance) at
   6k-14k steps. If memory never appears even at 14k, that quantifies the pixel
   promotion-gate cost and strengthens the sym-tier-first strategy.

## Ops
- Job: ferranti H100, name `colorfield-pixcurve`, 1 GPU, --hours 6.
- Pull: `scripts/pull_results.sh --cluster ferranti colorfield-pixcurve --what all`
  (checkpoints are small: 1.32M params ~ 5-6 MB each, 11 snapshots + final).
- SHA + JOB_ID recorded in agent/EXPERIMENTS.md (row `colorfield-pixcurve-h100`).
