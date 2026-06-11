# EXP-006 — Tokenizer A/B: VGG LPIPS vs MSE-only, long cluster runs (backfilled)

Decision: D-007 | Hypothesis: H1 (tokenizer quality) | 2026-06-10, cluster
Data: `occluded.npy`, 100 epochs each, config in W&B.

## Provenance

| arm | W&B run | commit | host | batch | runtime |
|---|---|---|---|---|---|
| A: no LPIPS | [1lzegsxt "Long non-LPIPS run"](https://wandb.ai/models-eberhard-karls-universit-t-t-bingen/transformer-C-tokenizer/runs/1lzegsxt) | 58ebfde | galvani-cn109 | 20 | 10.1 h |
| B: VGG LPIPS (weight 0.2, normalized) | [rc01geau "Long LPIPS (vgg) run"](https://wandb.ai/models-eberhard-karls-universit-t-t-bingen/transformer-C-tokenizer/runs/rc01geau) | 3205e8e | mlcbm014 | 24 | 1.7 h |

Shared config: embedding_dim 256, depth 8, n_heads 16, n_latents 4,
bottleneck_dim 64, patch_size 8, mae_mask 0–0.9, max_temporal_length 16, lr 3e-4,
bf16.

## Reconciliation

Expected (from D-007): LPIPS arm beats non-LPIPS on val/mse and visual sharpness.
Observed:
- A (no LPIPS): val/mse **3.23e-4**, latent_cos 0.038, pred_std 0.160
- B (VGG LPIPS): val/mse **1.41e-4**, latent_cos 0.044, pred_std 0.163
  (train/lpips 0.0115, lpips_scale 0.018)
LPIPS arm ~2.3× better on val/mse and visually sharper.
Surprise: none.
Hypothesis impact: H1 tokenizer component supported; recon quality "very happy"
(Merlin).
Confounds, acknowledged: arms differ in commit (B includes the LPIPS-normalization
commit beyond A's), batch size (20 vs 24), and host/GPU (10.1 h vs 1.7 h walltime,
30 vs 179 samples/s). Not a clean ablation; direction and magnitude judged
sufficient for the adoption decision regardless.
Next: proceed per plan → freeze B's checkpoint as `trained_autoencoder.pt`
(file timestamp 2026-06-10 18:48 local ≈ rc01geau finish 16:43 UTC) and train
dynamics on it (D-008 / EXP-007).
