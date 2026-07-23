# Memory Maze no-FF9 → K=4 shortcut continuation

## Question

Can the completed dense-memory/no-FF9 Memory Maze checkpoint learn the shortcut ladder required for
practical K=4 rollout without losing its d_min dynamics or memory relay?

The source checkpoint is `checkpoints/memmaze/dynamics_mem2mem_noff9.pt` (ferranti 415143, 50 epochs,
31h34m). It was trained with `--no-bootstrap`, which pinned `n_d_unlocked=1`: only
`d_min=1/128` received targets. This continuation keeps the architecture, data, 100% mem→mem rollout,
50/50 clean/noise modes, and no-FF9 loss unchanged. Only shortcut bootstrap targets are introduced.

## Pre-registered schedule

12 active optimizer-step hours on one ferranti H100, fresh AdamW state, peak LR `1e-4`, global
grad-clip 1.0:

| Active time | Finest-first targets available |
|---:|---|
| 0:00–1:00 | 1/128 only |
| 1:00–2:15 | through 1/64 |
| 2:15–3:30 | through 1/32 |
| 3:30–4:45 | through 1/16 |
| 4:45–6:00 | through 1/8 |
| 6:00–12:00 | through 1/4 |

Sampling is uniform over unlocked targets. The first hour linearly warms the fresh optimizer from
`1e-6` to `1e-4`; the final 20% cosine-cools back to `1e-6`. K=1 and K=2 (`d=1`, `d=1/2`) are
excluded globally and from this run. The destination is
`checkpoints/memmaze/dynamics_mem2mem_noff9_k4.pt`; the source checkpoint is never overwritten.

The trainer checkpoints hourly, at each unlock boundary, and at epoch end. The SLURM allocation is
13 hours so the requested 12 active training hours plus validation/checkpoint overhead can exit cleanly.

## Success / watch-outs

- K=4 rollout quality improves relative to the d_min-only source.
- d_min held-out behavior and the carried memory relay do not regress.
- No loss/gradient instability at unlock boundaries; especially watch the first coarse unlock and
  the final 1/4 unlock.
- Aggregate `val(normal)` is only a monitor; final judgment needs held-out per-d losses and matched
  Memory Maze K=4 rollout sheets.

## Provenance

Pending submission.
