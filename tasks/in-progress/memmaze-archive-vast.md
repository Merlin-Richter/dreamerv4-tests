# Train hierarchical archive memory on Memory Maze via vast.ai (5090)

Requested by Merlin 2026-07-12. Continue `checkpoints/memmaze/dynamics_mem2mem_noff9.pt` with the
hierarchical archive mechanism (`experiments/hierarchical-archive-memory/`, commit `eebb4b6` — all
correctness gates passed locally, see `tasks/done/hierarchical-archive-memory.md` §Result) so it
learns archive tokens. ferranti is DOWN → this runs on the vast box (RTX 5090, replacement instance
2026-07-12, proxy endpoint ssh8.vast.ai:13617 in cluster.env), which starts empty (no dataset/venv).

## Constraints discovered (why the pipeline differs from ferranti)
- **32 GB container disk** on the vast box → the ferranti path (unzip → convert_memmaze.py →
  **35.7 GB** memmaze9x9.npy → ensure_latent_cache) cannot fit. Training only reads the frames npy
  for its SHAPE (`train_archive.py:136`; cache-HIT check is `shape[:2]`); it trains off the fp16
  latent cache (~3 GB).
- **New prep path**: `experiments/hierarchical-archive-memory/prep_vast.{sh,py}` — download ONE
  shard, stream npz → latent cache + `_actions.npy` + a **SPARSE PLACEHOLDER** `data/memmaze9x9.npy`
  (valid npy header, zero data blocks; marker file `data/memmaze9x9.npy.SPARSE-PLACEHOLDER.txt`).
  NEVER pull/rsync that placeholder (it materializes 35.7 GB of zeros); never read pixels from it.
- **Shard**: Merlin's linked file id `1KmVoAofGWnwBJ0EqClYqWNBzENMA8riE` = **train-part8** (9.6 GB
  zip, ~10%) — a DIFFERENT 10% than ferranti's train-part0 (fresh data for the continuation; same
  distribution, episode order self-consistent within the shard).
- **Checkpoints go up via the new `scripts/push_file.sh`** (inverse of pull_file.sh; rsync over the
  master socket): `checkpoints/memmaze/tokenizer.pt` (329 MB) + `dynamics_mem2mem_noff9.pt` (164 MB).
- W&B key recovered from `~/.netrc` into `scripts/cluster.env` (was blank since the 06-18 recon).

## Plan
1. Commit prep/push scripts → `sync_code.sh --cluster vast autoresearch/jul11`.
2. `push_file.sh` the two checkpoints (before the prep job — it needs the tokenizer).
3. `vast_run.sh --name memmaze-archive-prep -- bash experiments/hierarchical-archive-memory/prep_vast.sh`
   (also builds the venv: unpinned torch must resolve a Blackwell-capable cu12.8+ build — prep does a
   GPU smoke matmul first and hard-fails if not).
4. Short calibrate job (`calibrate.py --frames 512 --dense-tbptt-frames 64`, bs sweep) → pick BS.
5. `vast_run.sh --name memmaze-archive -- bash experiments/hierarchical-archive-memory/train.sh
   EPOCHS BS --fast-memory-hide-frac 0.25 --hide-latents-frac 0.5` (archive-forcing per NOTES.md —
   without it the dense relay can ignore the archive, §17 of the design; flag for Merlin).
6. Monitor first epochs (vast_status.sh / W&B transformer-archive-memory), record ETA; pull
   `checkpoints/memmaze/dynamics_archive.pt` via pull_file.sh when done/interim.

## Done means
Training running healthily on the 5090 (archive stats nonzero in the epoch lines: n_archives ~31,
n_archives_used > 0), W&B live, provenance recorded in EXPERIMENTS.md + this file. Memory CLAIMS
wait for same-checkpoint archive-on/zeroed evals (design §16) — training loss is not the metric.

## Provenance
- (fill in: sync SHA, run names, W&B run id)
