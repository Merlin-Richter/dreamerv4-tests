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
- 2026-07-12: code synced @ `cf2ce02` (prep+push_file+calib commits acbcd8e/77d2870/cf2ce02 on
  autoresearch/jul11). Checkpoints pushed via push_file.sh (tokenizer.pt 329 MB +
  dynamics_mem2mem_noff9.pt 164 MB, ~0.7 MB/s uplink). Prep run `memmaze-archive-prep`
  (vast_run.sh, train-part8) launched ~04:40 — venv build (torch 2.13.0) + shard download +
  streamed latent build.
- ~04:55–05:30: HOST NETWORK OUTAGE (~35 min; not a reboot — host uptime 5d18h, container
  intact). The box's own DNS/outbound died too: the prep job burned its 3 pip retries on
  NameResolutionError and exited cleanly (pidfile freed). Both endpoints refused during the
  window; recovered on their own. NOTE this host's network has now dropped twice today
  (01:58 restart, 04:55 blackout) — flag for Merlin if it recurs mid-training.
- 05:36: prep relaunched (pid 2171), venv resumed via markerless-dir logic. Checkpoints
  verified on the box (byte sizes match local).
- venv COMPLETE: torch 2.13.0 + CUDA-13 wheel stack (Blackwell OK). ~06:25 another blackout
  stalled gdown dead at 7.52/10.1 GB (hung socket, no timeout) → cancelled; added gdown
  resume=True (download_memmaze.py) + fixed a set-e/pipefail death in prep_vast.sh's
  skip-check (first `memmaze-archive` launch died instantly on it, rc=1).
- FINAL FORM: single autonomous job `memmaze-archive` = run_all_vast.sh 50 train-part8
  (prep [idempotent] -> bs auto-calibration [largest of 8/6/4/2/1 with peak<26 GiB] ->
  train.sh 50 BS --fast-memory-hide-frac 0.25 --hide-latents-frac 0.5). Launched @ 7fb4007,
  pid 4713; gdown RESUMED from the 7.5 GB partial (78% within 2 min). No orchestrator
  round-trips needed anymore; local persistent watcher samples significant log lines.
- W&B: transformer-archive-memory / memmaze-archive-n16-r1 (key restored in cluster.env).
- Perf: bs=6 ran at 54% avg util (sequential-slide rollout is latency-bound) → restarted at
  bs=8 (calib peak 27.4 GiB, card-usable 31.36; + expandable_segments): **68% avg util,
  28.2 GiB, 379 W** → ~+25% throughput, ETA ~13-14 h for 50 epochs. bs≥9 does not fit;
  further gains need code changes (fuse slide loop / torch.compile) — future work.
- GOTCHA fixed on the way: vast_cancel killed only job.sh; the orphaned python trainer kept
  the GPU (22 GiB) and OOM'd the next launch → vast_cancel now kills the process group.
- First bs=6 epoch (for reference before restart): val(local) 0.00222, archive rollout
  0.00707, archives 32.0/used 31.0, hide 0.251/0.128 — archive path active and healthy.
- To pull the result: scripts/pull_file.sh --cluster vast checkpoints/memmaze/dynamics_archive.pt
