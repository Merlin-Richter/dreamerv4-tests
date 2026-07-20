# dreamer4-community-baseline

Integration glue for training the independent community Dreamer 4
([nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4)) on Memory Maze as an independent
vanilla baseline. The task is `tasks/backlog/memmaze-community-dreamer4-baseline.md`. This directory is
experimental integration code and does not follow `src/`'s one-spec-per-file discipline. The community
repo is not vendored here: `setup_upstream.sh` clones the pinned commit into a gitignored run directory,
applies the committed patch, and records the exact upstream diff and environment.

- `memmaze_to_dreamer4.py` converts raw Memory Maze `.npz` trajectories into the community repo's paired
  frame-shard and demo trees. The native six-way one-hot action occupies the first six of the upstream
  stack's 16 action dimensions. No time shift is applied: raw `action[t]` produced raw `image[t]`.
- `validate_integration.py` synthesizes Memory Maze data, runs the converter, loads it through the real
  upstream `ShardedFrameDataset` and `WMDataset`, then exercises 64x64 tokenizer and dynamics forwards.
- `upstream-memmaze.patch` is the narrow adaptation against upstream `b8abafbf`: it selects `memmaze`,
  accepts native 64px dynamics shards, adds resumable elapsed-time stop/cosine LR, selectable W&B mode,
  guaranteed final checkpoints, and a matched-noise action-shuffle diagnostic. It does not change the
  model architecture or training loss.
- `setup_upstream.sh` and `requirements-community.txt` reproducibly create the external checkout and
  isolated pinned environment, then rerun the integration regression.
- `validate_converted.py` checks all converted shard/demo tensors and uses trajectory content fingerprints
  to prove train/eval disjointness.
- `phase1_smoke.sh` is the ferranti Phase 0/1 driver: setup, resumable public-data download, full train
  part 0 plus eval conversion, H100 batch calibration, tokenizer smoke, held-out reconstruction sheet,
  action-conditioned dynamics smoke, checkpoints, and GPU telemetry.
- `make_recon_sheet.py` and `summarize_checkpoint.py` provide held-out visual acceptance and stable
  checkpoint throughput/provenance summaries.

The cluster-free integration test passed locally against upstream `b8abafbf` on 2026-07-20.

Submit the smoke only through the academic-cluster wrapper:

```bash
bash scripts/submit_job.sh --cluster ferranti --name memmaze-d4-phase1-smoke --hours 8 -- \
  bash experiments/dreamer4-community-baseline/phase1_smoke.sh
```
