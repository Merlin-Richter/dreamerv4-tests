# Memory Maze 9x9 tokenizer — experiment notes

Task: `tasks/in-progress/memmaze-tokenizer-train.md` (LOCKED decisions by Merlin).
Goal: a **frozen** tokenizer that compresses Memory Maze 64x64x3 first-person frames to latents
(`n_latents=32, bottleneck_dim=16` => 512-d/frame), trained on ferranti (H100). Dynamics is a follow-up.

## Status
- Branch: `exp/mem2mem-rollout-only`. Cluster: ferranti (galvani socket DOWN, not needed).
- [done] Config exposure: `train_tokenizer.py` now CLI-exposes `--embedding-dim/--depth/--n-heads/
  --n-latents/--bottleneck-dim` (unset = GridWorld dataclass defaults). Spec updated.
- [done] Scripts: `download_memmaze.py`, `convert_memmaze.py`, `bs_search.py` (this dir).
- [pending] cluster jobs: bs-search (batch size) + download/convert (data) — see Provenance.

## Dataset download — RESOLVED structure (2026-06-29)
Drive folder `1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB` -> `memory-maze-9x9` (id `1-tMdUzshBEEIo5EjBnECn7IFK1lJLFsQ`)
is **11 single zip files**, NOT 30k loose files:
- `eval.zip` (1k eval trajectories) + `train-part0.zip` .. `train-part9.zip` (10 shards, ~2.9k traj each = 29k train).
- So the task's "one folder ~= 10%" = **one `train-partN.zip` ~= 2,900 trajectories (~10 GB)**.
- We pull whole SINGLE public files via `gdown <id>` => gdown's 50-files-per-folder limit does not apply.
  File IDs are baked into `download_memmaze.py`.
- Decision (flagged to Merlin): start with **train-part0** (~10%, ~2.9k traj x 1001 frames ~= 2.9M frames —
  already far larger than GridWorld). Reversible: `convert_memmaze.py` handles N trajectories, so we can
  pull more parts and re-convert if the tokenizer wants more data. (Task LOCKED text says "FULL ~100GB"
  but the pipeline step says "10% / train on what we got" — taking the latter, conservative-first.)

## LOCKED config (from the task)
`n_latents=32, bottleneck_dim=16, embedding_dim=512, n_heads=16, depth=12` (4x[spatial,temporal,spatial];
temporal at layers 1,4,7,10), `max_temporal_length=64`, LPIPS ON, fg-weight OFF, MAE mask 0.0/0.9,
img 64x64x3, RGB kept as-is. `embedding_dim/n_heads => head_dim 32` (divisible, ok).

## Eval / "done"
No closed-form readout for Memory Maze. Validity = recon sheets (sharpness), val recon MSE + LPIPS,
latent-collapse health (latent_cos < 0.7 escaped, pred_std > 0.04 content — the QK-norm-temperature
failure mode). Frozen -> `checkpoints/memmaze/tokenizer.pt` (pulled local).

## Provenance (fill at execution)
- Config-exposure commit SHA: `db412935` (branch `exp/mem2mem-rollout-only`).
- Prep job (bs-search + download train-part0 + convert): ferranti **job 412622** @ SHA `db412935`
  (`cluster_prep.sh train-part0`, --hours 3 --cpus 8). DONE rc=0, ~13 min total.
  - **bs-search (H100, LOCKED cfg, LPIPS on):** bs1 10.9GB/12.4clips/s · bs4 38.5GB/16.2 · bs8 75.2GB/17.4 ·
    bs12 OOM. **MAX batch = 8**; throughput plateaus past bs4 (LPIPS-VGG-bound). Using **bs=6** for the run
    (~56GB, headroom; no speed cost since plateaued).
  - **data:** train-part0 = **2900 trajectories**, each `image` (1001,64,64,3) uint8 -> `data/memmaze9x9.npy`
    (2900,1001,64,64,3) **35.7 GB** on /weka. Convert ~3.5 min @ ~14 traj/s. README format confirmed (key 'image').
- Validation run (real-data smoke, 600 eps x 3 ep, bs6, full LOCKED cfg, LPIPS, grad-spike 5.0): ferranti
  **job 412625** @ SHA `db412935` -> `checkpoints/memmaze/tokenizer_val.pt`. Purpose: real /weka epoch time +
  stability + initial recon/latent-collapse health, before sizing the full run.
- **Validation result (412625, 600 eps x 3, bs6):** val MSE 0.00496 -> 0.00154 -> 0.00078 (halving/epoch,
  NOT plateaued); latent_cos 0.327/0.341/0.382 (<0.7 => NO collapse); pred_std ~0.15 (>0.04 => real content);
  1/1425 steps skipped (stable). Real throughput 2.74 it/s @bs6 = 16.4 clips/s (compute-bound on LPIPS, NOT
  /weka-IO-bound) => ~40 min/epoch on full train-part0. GREEN — proceed to full run.
- **Full train job:** ferranti **job 412635** @ SHA `be1258e`
  (`train_and_recon.sh 15` = 15 epochs bs6 LOCKED cfg + LPIPS + grad-spike 5.0 + W&B, then a recon sheet),
  --hours 13 --cpus 8. Merlin chose 15 epochs on the 10% shard. -> `checkpoints/memmaze/tokenizer.pt` +
  `experiments/memmaze-tokenizer/_recon_memmaze.png`. W&B: transformer-C-tokenizer / memmaze-tok-full.

NOTE (flagged to Merlin): the prep job holds an H100 while doing CPU/IO-only download+convert (the
wrappers only expose the GPU partition). One-time; acceptable. bs-search runs first so the GPU isn't
idle the whole time.
