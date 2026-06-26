# train_dynamics.py — train the dynamics model on the frozen tokenizer's latents.

Trains a `DynamicsModel` (vanilla or FF9-memory) on GridWorld frames encoded by the FROZEN tokenizer.
Loss = shortcut forcing (+ FF9 sufficiency when `--ff9>0`). The dynamics model never touches pixels.

## Interface
- CLI: `--frames <.npy>`, `--tokenizer <frozen .pt>`, and `--checkpoint <out.pt>` are **required** (no
  defaults; env-specific). `--checkpoint` is the SAVE destination (written each epoch; also the
  checkpoint loaded by `--test-checkpoint`). `--resume <in.pt>` (optional) loads weights + config to start
  training FROM (default: random init; e.g. warm-start FF9 from a vanilla checkpoint).
  `--epochs --batch-size --lr --seed --context-length`; memory: `--ff9 K --n-memory M`;
  `--wandb*`; `--test-checkpoint` (interactive rollout viz instead of training).
- Produces `<checkpoint>.pt` = `{config, model_state_dict}`.

## Behavior
- Load frames (memmap) + actions; split train/val (fixed seed). Load + FREEZE the tokenizer; read its
  `n_latents/bottleneck_dim` so the dynamics config matches. `n_actions` auto-detected from the actions.
- `ChunkClipDataset`: yields `context-length`-frame clips (+ per-frame action ids). Each batch:
  encode frames → latents with the frozen tokenizer (no grad), then `model.loss(z1, actions)`.
- Optimiser AdamW; **gradient clipping at max_norm=1.0** (prevents the tokenizer-style blow-up). Per-epoch
  val loss; checkpoint each epoch; W&B optional (loss parts).
- **LR schedule (per-step, mirrors the tokenizer):** linear warmup over the first ~5% of steps
  (min 200) → flat at peak LR → cosine cooldown only over the **final 20%** (decay starts at 80% of
  total steps), down to ~1e-6. (Not a cosine-from-step-0 — the model should train at full LR for the
  bulk of the run.)

## Invariants
- Tokenizer is frozen (eval, no grad) and its dims drive the dynamics config — they must agree.
- Latents encoded per batch on the fly; dynamics trains only the transition, never the tokenizer.
- `--ff9 K` ⇒ memory model (`n_memory=M>0`, FF9 sufficiency loss on); else vanilla. grad-clip stays on.
- `context-length` = the model's temporal window. Run with `-u`.
