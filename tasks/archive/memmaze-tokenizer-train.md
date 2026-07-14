# Train a tokenizer on the Memory Maze 9x9 offline dataset

> DRAFT for Merlin to review/edit, then move to backlog. Decisions marked **[LOCKED]** were chosen by
> Merlin in the kickoff Q&A; items marked **[OPEN]** still need his call before execution.

## Why / goal
Extend the memory-research pipeline from GridWorld to a real 3D memory env (Memory Maze, Pasukonis/Hafner).
Step 1 of the pipeline is a **frozen tokenizer** that compresses 64x64 first-person frames to latents; a
Memory Maze dynamics + memory model later trains on those latents and is scored on a Memory-Maze recall/
probe eval. This task produces only the tokenizer (the latent space); dynamics is a follow-up.

## Decisions [LOCKED by Merlin]
- **Data:** download the **FULL** Memory Maze **9x9** offline dataset (~100GB, 30k trajectories x 1001
  frames), **once**, **on the cluster** (ferranti). Source: the repo's Google Drive folder
  (`external/memory-maze/README.md` -> "Offline Dataset").
- **Compute:** train on **ferranti (H100)**. Data lives on the cluster (downloaded there; NOT synced via
  our code path).
- **Tokenizer config:** `n_latents=32`, `bottleneck_dim=16` (=> 512-d bottleneck/frame), `embedding_dim=512`,
  `n_heads=16`, `depth=12` (4x[spatial,temporal,spatial]; temporal at layers 1,4,7,10), context window
  `max_temporal_length=64`, **LPIPS on**.
- **Data feeding:** a **single `.npy`** mmapped on disk (the existing `ChunkClipDataset` already does this —
  never loads the whole set into RAM).
- **Loss:** plain recon MSE + LPIPS. **fg-weight OFF** (the GridWorld foreground-mask loss keys off
  "pixels that deviate from a static scene" — under egomotion nearly every pixel moves, so it would mark the
  whole frame; it does not transfer to Memory Maze).
- **Resolution:** 64x64x3 (the offline `image` is already 64x64x3 — matches `img_input`).
- **Channel order:** **keep RGB as-is — do NOT touch channels.** Memory Maze is a separate env from
  GridWorld's BGR pipeline; the tokenizer just autoencodes, so channel order is irrelevant to the model
  (it learns whatever channels it's given). Only cosmetic caveat: a later cv2-based Memory Maze viewer would
  show swapped colors unless told the data is RGB — not the model's concern.

## Pipeline (steps to execute)
1. **Download** 10% (one folder) of the full 9x9 offline dataset onto ferranti (~10GB) into `data/memmaze9x9_raw/` (gitignored).
   - The source is a Google Drive folder. Headless download needs `gdown`/`rclone`; GDrive folder pulls of
     this size can hit quota throttling.
   - The dataset ships pre-split: 29k train + 1k eval trajectories. Try this download using gdown or similar and if we hit quota, just train on what we got. how val is split is irrelevant, we can use our normal slit
2. **Convert** the per-trajectory `.npz` files -> a single mmappable `.npy` on the cluster. (we are not sure what the extracted zip actually contains)
   - (If we need to convert): New script (experiment-local, e.g. `experiments/memmaze-tokenizer/convert_memmaze.py`): iterate the
     `.npz` files, take rollouts of `images` (1001,64,64,3) uint8 **as-is (RGB, channels untouched)**, write incrementally
     into a preallocated `np.memmap` of shape `(N, 1001, 64, 64, 3)` uint8 so the converter stays within RAM.
   - Output: `data/memmaze9x9.npy`
     gitignored, on /weka (546T free, fine). The trainer self-split via `--val-fraction`
3. **Expose the config.** `train_tokenizer.py` currently only CLI-exposes `img_input_H/W` and
   `--context-length` (=`max_temporal_length`); `n_latents/bottleneck_dim/embedding_dim/depth/n_heads` are
   hardcoded to defaults. To run the LOCKED config we must parameterize them. Add CLI args to `src/training/train_tokenizer.py` (+ update its spec `specs/training/...`) — touches
     a spec-backed file
   Recommendation to consider: (a) is cleaner long-term (the dims are legitimately env-dependent), (b) keeps
   the spec-driven `src/` pristine until this graduates.
4. **Train** on ferranti H100 with the LOCKED config + `--lpips`, `--context-length 64`.
   - epochs, batch size, LR. The model is much bigger than the GridWorld tokenizer (512-d, 12
     layers, 64-frame temporal window, 96 spatial tokens/frame = 64 patches + 32 latents) AND clips are
     64 frames, so activation memory is large — batch size needs searching on the H100 (first make a process that searches what the highest batch size effective for training with this config is).
     keep MAE mask range (`mae_min/max_mask` 0.0/0.9)
5. **Eval / validity** (no GridWorld closed-form readout exists for Memory Maze):
   - reconstruction sheets (`--save-recon`) eyeballed for sharpness, val recon MSE + LPIPS, and a
     latent-collapse check (latent activation variance / no mean-image collapse — the QK-norm temperature
     issue from `[[qk-norm-attention-temperature]]`). Pull the recon sheet back to inspect.
   - **Frozen** after training; checkpoint -> `checkpoints/memmaze/tokenizer.pt` (pulled to local).

## Risks / things to watch
- **Download logistics** (100GB GDrive folder, headless) is the most likely blocker — see step 1 [OPEN].
- **357GB single `.npy` on /weka**: mmap random-access over a network FS may bottleneck throughput; consider
  frame subsampling (step 2 [OPEN]) or sequential-ish sampling. Conversion is a one-time ~hours job.
- **Capacity vs the env:** 32x16 = 512-d/frame is 2x the GridWorld bottleneck; whether it reconstructs 3D
  views sharply is exactly what the recon sheets will show. Downstream: the Memory Maze **dynamics model
  must match `n_latents=32, bottleneck_dim=16`**.
- The tokenizer ignores actions; the rich label set in the npz (`agent_pos`, `maze_layout`, `target_*`) is
  not needed now but is the raw material for a later Memory-Maze recall/probe eval (the analogue of
  GridWorld's measurement-only `hidden_state()`).

## What "done" means
A frozen Memory Maze 9x9 tokenizer checkpoint that reconstructs held-out frames cleanly (sharp recon sheets,
low val recon/LPIPS, no latent collapse), trained on ferranti from the converted single-`.npy` dataset, with
provenance (branch + SHA + job id + config) recorded in `experiments/memmaze-tokenizer/NOTES.md` and a one-
line entry in `agent/EXPERIMENTS.md`.

## Provenance (to fill at execution)
- Branch / SHA: `<fill>`. Cluster ferranti. Job: `<fill>` -> `checkpoints/memmaze/tokenizer.pt`.
- Dataset: `data/memmaze9x9.npy` converted from the GDrive 9x9 offline set.

## RESULT (2026-06-30)
DONE — ferranti 412635 @ be1258e (15ep, 10h16m, rc=0): val MSE 0.000074, latent_cos 0.235 (no collapse), pred_std 0.16; recon sheet faithful (only high-freq texture smoothing); ckpt pulled+verified -> checkpoints/memmaze/tokenizer.pt, FROZEN. Details: experiments/memmaze-tokenizer/NOTES.md
