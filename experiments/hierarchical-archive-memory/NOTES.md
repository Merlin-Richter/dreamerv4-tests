# Hierarchical archive memory — implementation/run record

Design/task: `tasks/in-progress/hierarchical-archive-memory.md`.

## Hypothesis

The per-frame fast-memory relay is useful for immediate state but repeatedly rewrites old observations.
A tiny detached-source compressor can turn each 16-frame, per-slot fast-memory segment into one durable
archive token per slot. Grouped archive readers then give current fast memory a direct, cheap attention
path to views from hundreds of frames ago.

The experiment continues `checkpoints/memmaze/dynamics_mem2mem_noff9.pt`; it does not edit `src/`.

## Version-one implementation

- `model.py`: one-block slot-wise compressor; one grouped GQA-capable archive reader per temporal block;
  separate absolute-RoPE archive K/V cache; archive-aware prefill/commit/read-only rollout.
- `rollout.py`: fixed W=32 absolute-position rollout, 50/50 latent-present/memory-only modes, optional
  fast-memory hiding and mixed archive ablation, dense blockwise backward, detached archive sources,
  leaf proxies, deferred compressor VJP.
- `train_archive.py`: checkpoint-validated warm start, 512-frame clips, fixed-window trainer, W&B/checkpoint
  output.
- `sheets.py`: archive-checkpoint Memory Maze sheets with a `--zero-archive` same-checkpoint ablation.
- `smoke.py`: deterministic CPU architecture/cache/autograd gates.
- `calibrate.py`: production-checkpoint synthetic CUDA memory/time probe.

## Local verification (2026-07-12, RTX 4070 Laptop)

All archive smoke gates pass:

1. compressor shape and per-slot source isolation;
2. grouped reader isolation, exact eligibility zero, raw-vs-cached equivalence under GQA;
3. gate-zero archive model exactly equals the base model;
4. boundary writes and `commit=False` state immutability;
5. deferred compressor VJP equals direct one-shot gradients;
6. blockwise long rollout with archive-only fast-memory/latent hiding gives nonzero reader and compressor
   gradients.

Existing `src/tests/test_dynamics.py` and `src/tests/test_dynamics_cache.py` both pass unchanged.

Production-shape synthetic calibration, batch 1, fast-memory no-FF9 checkpoint:

| Frames | Slides | Archives / used | Peak allocated | Time |
|---:|---:|---:|---:|---:|
| 48 | 1 | 3 / 2 | 1.63 GiB | 1.36 s |
| 128 | 6 | 8 / 7 | 3.66 GiB | 1.45 s |
| 512 | 30 | 32 / 31 | 3.66 GiB | 3.44 s |

The identical 128/512 peak is the critical result: dense activation memory is bounded by TBPTT while
only detached segment sources/proxies grow with clip length. The 512-frame calibration used
`--fast-memory-hide-frac 0.25` and exercised both hiding variants.

Controlled archive-required overfit (`overfit.py`, frozen unconditional backbone, 400 steps):

```text
initial loss:                  0.11688
trained archive-on loss:      0.04851
same weights, archive zeroed: 0.11242
```

The first unconstrained probe found the generic-mean shortcut and correctly failed its ablation gate.
Freezing the non-archive backbone removed that loophole: the passing result demonstrates that the
compressor/readers can carry sample-specific latent identity through the archive-only route.

## Run

```bash
bash experiments/hierarchical-archive-memory/train.sh 50 BS
```

Calibrate `BS` on the target cluster before the long run. Add archive-forcing only after the base path
and evaluation instrumentation are verified, for example:

```bash
bash experiments/hierarchical-archive-memory/train.sh 50 BS \
  --fast-memory-hide-frac 0.25 --hide-latents-frac 0.5
```

## Decision metric

The mechanism wins only if same-checkpoint archive-on versus archive-zeroed evaluation improves
Memory Maze corner/revisit predictions beyond the dense TBPTT/local-window horizon without regressing
short-horizon rollout quality. Training loss is not the decision metric.
