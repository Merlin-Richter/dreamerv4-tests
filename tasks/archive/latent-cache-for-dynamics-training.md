# Latent disk cache for dynamics training (skip the tokenizer at train time)

Requested by Merlin 2026-07-03. **Spec edits delegated to the agent for this campaign** (Merlin: "You
can edit specs"). Design agreed in-session:

## Problem
`train_dynamics.py` (and `experiments/mem2mem/train_mem2mem.py`) encode pixels -> latents through the
frozen tokenizer **per batch, every epoch**: tokenizer sits in VRAM all run, same frames re-encoded
once per epoch, and the DataLoader streams the full pixel npy (memmaze: 35.7GB) each epoch.

## Design (agreed)
- **On-disk cache keyed by (tokenizer, dataset-file):** `<frames_stem>.latents-<tokhash>.npy` (+ `.json`
  meta) next to the frames file; tokhash = sha256 of the tokenizer checkpoint bytes (12 hex).
- First run with a (tokenizer, frames) combo: encode the whole dataset once (no-grad, eval-mode
  encoder, non-overlapping windows of the tokenizer's `max_temporal_length`, trailing partial window
  encoded as-is), store **fp16** `(N, T, n_latents, bottleneck_dim)` via open_memmap, atomic rename.
  Then free the tokenizer and train from the mmapped latents. Later runs: cache hit -> tokenizer never
  loaded onto the GPU for training.
- Dynamics chunking (incl. the per-epoch random clip offset) happens on the cached latents at ANY
  offset. Merlin's call: window position doesn't matter materially ("latents only depend on frame t");
  supporting evidence: the GridWorld mem2mem winner was already trained on boundary-crossing latents
  (train_mem2mem.py block-encode) and hit 0.99 recall. Still MEASURE window-invariance (probe: same
  frames encoded at two window offsets; report latent cos/MSE + decoded-recon delta) and record numbers.
- `--encode-online` escape hatch keeps the old per-batch path (debug/AB). `--build-latent-cache-only`
  builds the cache and exits (for prep jobs, avoids two parallel jobs racing to build).
- `--test-checkpoint` viz still uses the tokenizer (decode) — unchanged.
- Update spec `specs/training/train_dynamics.md` accordingly; adapt `experiments/mem2mem/train_mem2mem.py`
  to the same cache helper (experiment-local, no spec).

## Done means
- Spec + `src/training/train_dynamics.py` updated; gate tests green; GridWorld local (4070): cache
  builds, cache-hit training runs WITHOUT the tokenizer in VRAM, loss curve ~matches online path.
- Window-invariance probe numbers recorded (GridWorld local; memmaze in the prep task).
- mem2mem trainer consumes the cache.

## RESULT (2026-07-03)
DONE. Cache implemented in train_dynamics.py (+spec updated under delegated authority): fp16 <frames>.latents-<sha12>.npy + json meta, atomic build, --encode-online/--build-latent-cache-only/--cache-batch. GridWorld 4070 verification: cache builds (40ep/5s), cache-vs-online 2-epoch losses match (val 0.0347 vs 0.0336), gate tests green. Window-invariance probe (experiments/memmaze-dynamics/probe_window_invariance.py): latent cos 0.9975, window-delta recon MSE 1.3e-6 << recon-vs-GT 8.2e-6 -> Merlin claim CONFIRMED on GridWorld. mem2mem trainer ported to cache (rollout-only smoke green).
