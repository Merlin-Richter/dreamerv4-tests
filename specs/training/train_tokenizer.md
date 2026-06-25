# train_tokenizer.py — train the temporal autoencoder (the latent backbone), then freeze it.

Trains `AutoEncoder` to reconstruct GridWorld frames; the result is frozen and reused everywhere.
Reconstruction loss (MSE, optional LPIPS) over T-frame clips.

## Interface
- CLI: `--frames data/gridworld.npy --checkpoint <out.pt> --epochs --batch-size --lr --seed`;
  `--lpips` (perceptual term); `--wandb*`; `--test-checkpoint` (show recon strips instead of training).
- Produces `<checkpoint>.pt = {config, model_state_dict}` + recon-strip images.

## Behavior
- Load frames (memmap), T-frame clips, train/val split. Forward = `decoder(encoder(frames))`; loss =
  MSE (+ LPIPS if enabled) in [0,1]. MAE patch-dropout is active in train mode (in the encoder).
- AdamW; per-epoch val; save the BEST checkpoint by a foreground/recon metric (not just the last); emit
  reconstruction strips (GT vs recon) so the square is visibly encoded, not a background-only cheat.

## Invariants
- **Stability (load-bearing, EXP-024→025):** AdamW `beta2≈0.95` + skip optimizer steps on grad-norm
  spikes + keep the BEST checkpoint. Without these the loss can hit a low then explode and the single
  overwrite discards the good model. Log per-step grad-norm so an intra-epoch explosion is visible.
- The recon VISUAL is the real success check (a low aggregate MSE can hide a dropped square — EXP-024).
- Output is FROZEN after training. BGR, [0,1]. Run with `-u`.
