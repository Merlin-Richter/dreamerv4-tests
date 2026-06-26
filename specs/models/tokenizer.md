# tokenizer.py — temporal video autoencoder (the frozen latent backbone)

Compresses each video frame into a small set of continuous latent tokens, and reconstructs frames from
them. The dynamics model lives in this latent space. Trained separately (LPIPS+MSE), then FROZEN — the
dynamics model and all evals build on a fixed latent space. Same "efficient transformer" architecture as
the dynamics model: 2-D (space/time) attention, RMSNorm, RoPE on time, QK-norm, soft-cap, SwiGLU.

## Interface
- `AutoEncoderConfig`: `embedding_dim, depth, n_heads, mlp_ratio`; `patch_size, img_input_H/W`;
  `n_latents, bottleneck_dim`; `max_temporal_length`; `mae_min_mask, mae_max_mask`; stability/soft-cap.
- `AutoEncoder(cfg)` = `Encoder` + `Decoder`; `forward(frames) -> recon`.
  - `Encoder(frames (B,T,H,W,3)) -> latents (B,T,n_latents,bottleneck_dim)`.
  - `Decoder(latents) -> frames (B,T,H,W,3)`.

## Behavior
- **Patchify**: split each frame into `patch_size`×`patch_size` patches → patch tokens (linear proj) +
  learned position embedding. `n_patches = (H/patch)·(W/patch)`.
- **Encoder**: append `n_latents` learned latent tokens to the patch tokens; run the transformer; keep
  the latent tokens; project to `bottleneck_dim` + Tanh → the latents. **Restricted spatial attention =
  the information bottleneck**: in the encoder the latent tokens attend to patches but **patches do NOT
  attend to latents**; so the latents must summarise the frame.
- **Decoder**: from the bottleneck latents + `n_patches` learned patch tokens, run the transformer;
  **patch tokens attend to the latents but the latents do NOT attend back**; keep the patch tokens →
  project to pixels → **sigmoid (bounds output to [0,1], the range of the /255 targets)** → un-patchify
  to the frame.
- **Temporal layers** (the middle of each `[spatial, temporal, spatial]` triple — **every 3rd block**,
  `i%3==1`, so `depth=9` is three groups) are causal across time with RoPE — frames see the past, giving
  temporally-consistent latents.
- **MAE patch-dropout** (train only): per image, drop a fraction `~U[mae_min,mae_max]` of patch tokens
  (replaced by a learned token) before encoding. Forces the decoder to use the latents instead of
  copying patches / collapsing to the mean image.

## Invariants
- **Learnable per-head attention temperature** (`logit_scale`, init log(4), clamped to log(100)). With
  QK-norm, the textbook 1/√d scale makes attention near-uniform → latents collapse to the mean image;
  the sharper learnable scale escapes that basin. Do not drop it.
- The restricted cross-attention masks (encoder: patches can't see latents; decoder: latents can't see
  patches) ARE the bottleneck — keep them exactly.
- FROZEN after training; the dynamics model + evals depend on a fixed latent space. BGR in/out, [0,1].
- Latents are `(B,T,n_latents,bottleneck_dim)`; `n_latents=4`, `bottleneck_dim=64`, `patch_size=8`.
