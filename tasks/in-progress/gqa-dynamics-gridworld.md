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
