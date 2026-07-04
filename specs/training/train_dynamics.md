# train_dynamics.py — train the dynamics model on the frozen tokenizer's latents.

Trains a `DynamicsModel` (vanilla or FF9-memory) on GridWorld frames encoded by the FROZEN tokenizer.
Loss = shortcut forcing (+ FF9 sufficiency when `--ff9>0`). The dynamics model never touches pixels.

## Interface
- CLI: `--frames <.npy>`, `--tokenizer <frozen .pt>`, and `--checkpoint <out.pt>` are **required** (no
  defaults; env-specific). `--checkpoint` is the SAVE destination (written each epoch; also the
  checkpoint loaded by `--test-checkpoint`). `--resume <in.pt>` (optional) loads weights + config to start
  training FROM (default: random init; e.g. warm-start FF9 from a vanilla checkpoint).
  `--epochs --batch-size --lr --seed --context-length`; memory: `--ff9 K --n-memory M`;
  model dims (unset = GridWorld dataclass defaults; env-dependent, mirrors train_tokenizer):
  `--embedding-dim --depth --n-heads --gqa-groups --n-registers`;
  `--wandb*`; `--test-checkpoint` (interactive rollout viz instead of training).
  Latent cache: `--encode-online` (legacy per-batch encoding), `--build-latent-cache-only` (build the
  cache, then exit — for prep jobs so parallel training jobs don't race to build), `--cache-batch N`
  (episodes per encode batch during the one-time build; default 4).
- Produces `<checkpoint>.pt` = `{config, model_state_dict}`.

## Behavior
- Load frames (memmap) + actions; split train/val (fixed seed). `n_actions` auto-detected from actions.
- **Latent disk cache (default):** latents are read from `<frames_stem>.latents-<tokhash12>.npy`
  (+ `.json` meta) next to the frames, keyed by the sha256 of the tokenizer checkpoint bytes. On a
  cache MISS, encode the full dataset ONCE — frozen eval-mode encoder (MAE off, deterministic),
  non-overlapping windows of the tokenizer's `max_temporal_length` (trailing partial window as-is),
  fp16 `(N, T, n_latents, bottleneck_dim)`, written via `open_memmap` + atomic rename — then free the
  tokenizer. On a HIT the tokenizer is **never loaded**: no tokenizer VRAM, no per-batch encode, and
  the DataLoader streams latents (~12x smaller than pixels for memmaze). Dynamics dims
  (`n_latents/bottleneck_dim`) come from the cache's trailing dims (cache path) or the tokenizer
  modules (`--encode-online` path).
- Cached latents are sliced at ARBITRARY clip offsets (incl. the per-epoch random start offset): the
  causal encoder gives a frame's latent its window-start dependence, but per-frame-reconstruction
  training leaves latents ~window-invariant (Merlin 2026-07-03; measured by
  `experiments/memmaze-dynamics/probe_window_invariance.py`, and the GridWorld mem2mem winner already
  trained on boundary-crossing block-encoded latents).
- `ChunkClipDataset`: yields `context-length`-frame clips (+ per-frame action ids) from EITHER uint8
  pixel frames (scaled to [0,1]) OR fp16 cached latents (passed through as fp32). Each batch:
  `z1 = clip` (cache path) or `z1 = encode(frames)` (online path), then `model.loss(z1, actions)`.
- Optimiser AdamW; **gradient clipping at max_norm=1.0** (prevents the tokenizer-style blow-up). Per-epoch
  val loss; checkpoint each epoch; W&B optional (loss parts).
- **LR schedule (per-step, mirrors the tokenizer):** linear warmup over the first ~5% of steps
  (min 200) → flat at peak LR → cosine cooldown only over the **final 20%** (decay starts at 80% of
  total steps), down to ~1e-6. (Not a cosine-from-step-0 — the model should train at full LR for the
  bulk of the run.)

## Invariants
- Tokenizer is frozen (eval, no grad) and its dims drive the dynamics config — they must agree
  (cache path: dims read from the cache shape, which the tokenizer produced).
- Latents come from the disk cache by default (tokenizer runs once per (frames, tokenizer) combo,
  never during training); `--encode-online` is the only per-batch encoding path. Dynamics trains only
  the transition, never the tokenizer.
- `--test-checkpoint` (viz) still loads the tokenizer (encode context / decode predictions) — pixels
  in, pixels out; the cache is a training-path optimization only.
- `--ff9 K` ⇒ memory model (`n_memory=M>0`, FF9 sufficiency loss on); else vanilla. grad-clip stays on.
- `context-length` = the model's temporal window. Run with `-u`.
