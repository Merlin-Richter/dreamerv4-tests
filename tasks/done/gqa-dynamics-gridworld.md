# GQA dynamics experiment: 4× smaller KV cache, GridWorld test

Requested by Merlin 2026-07-04: "in an experiment try to implement grouped query attention to cut
the KV footprint by 4x. train on gridworld as a test."

## Done means
GQA (16Q/4KV heads) dynamics model implemented as `--model-module` experiment (no spec change),
correctness-verified (causality + cache equivalence + measured 4.00× cache cut — DONE pre-launch,
see `experiments/gqa-dynamics/smoke.py`), trained on GridWorld with the τ0-anchor objective, and
compared head-to-head vs `dynamics_vanilla_tau0.pt` (same objective, MHA): val/loss, teacher-forced
probe, recall w8. Verdict + EXPERIMENTS.md line.

## Provenance
- ferranti **job 415214** @ SHA `7ae5d72` (50ep bs256 seed0, 5x data, --hours 4), submitted 2026-07-04 19:51. -> `checkpoints/gridworld/dynamics_gqa_tau0.pt`, W&B `gw-dyn-gqa-tau0`.

## Result (2026-07-04)
DONE — PARITY at 4.00x smaller cache. Job 415214 @ 7ae5d72 (17 min). GQA (16Q/4KV) matches
MHA-tau0 on everything (val 0.001058 vs 0.001032; teacher-forced 1.0 at t>=4; free-run 1.0 flat;
recall w8 identical honest-baseline shape) with the rollout KV cache measured at exactly 4.00x
smaller (230 vs 922 KB) and 11% fewer params. Full table: experiments/gqa-dynamics/NOTES.md.
