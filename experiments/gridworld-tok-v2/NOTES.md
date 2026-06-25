# EXP-024 — GridWorld tokenizer v2 — FAILED (ball dropped); reconciliation below is CORRECTED

> **CORRECTION (2026-06-18, after Merlin's visual review):** my first reconciliation (kept below for the
> record) wrongly called this a clean success off the aggregate metrics. It is a **FAT FAIL.** The recon
> **drops the ball entirely** — background+grid reconstructed perfectly, the blue ball is ABSENT in every
> reconstruction frame across all bg colors (evidence: `_block0.png` red bg, `_block2.png` purple bg; GT
> top row has the ball, recon bottom row does not). Mechanism: the ball is ~1/36 cells (~1% of pixels), so
> per-pixel MSE + LPIPS have near-zero gradient for it → the tokenizer quickly settled into the trivial
> local optimum of reconstructing only the static background. **The aggregate val MSE (0.00364) is BLIND to
> this** — it is dominated by the background+grid, which is why it looked healthy. latent_cos/pred_std being
> "healthy" only means the latents vary (across bg/grid), NOT that they encode the ball. Lesson: for a sparse
> foreground object, aggregate recon MSE is not a validity-bearing metric — need a ball-region/foreground
> metric. The latents carry no ball position/color, so this tokenizer is NOT a usable backbone. Do NOT freeze.
> Fix: foreground-weighted reconstruction loss + a ball-region recon metric in W&B; retrain. (Surprise: HIGH.)

---
## ORIGINAL (incorrect) reconciliation — kept for the audit trail
# EXP-024 — GridWorld tokenizer v2 (frozen backbone) — reconciliation

Run: ferranti job 405629 `gridworld-tok-v2`, COMPLETED 0:0, elapsed 01:23:05 (~1.4h, as D-041 predicted).
Provenance: feat/motion-prediction @ d5cef58. Decision: D-039 (+ D-041 perf fix). W&B run `gridworld-tok-v2`
in project `transformer-C-tokenizer`. Config: 6×6 anti-overfit env (D-038), cluster datagen 3000×200,
tokenizer LPIPS(vgg), bs64, lr3e-4, 30 epochs, --cpus 8, bf16+TF32, --save-recon.
Artifacts here: `tokenizer.pt` (55MB, final ep30), `recon.png` (6 input/recon strips), `slurm-405629.out`.

## Reconciliation (§5)
- **Expected** (from D-039): faithful gridworld reconstruction (crisp grid + colored square, low val MSE),
  NO latent collapse (latent_cos well below 0.7, pred_std high). Frozen tokenizer ready for vanilla dynamics.
- **Observed** (epoch 30/30):
  - val MSE **0.00364** (train 0.00379), monotone-converging, no plateau-then-rise. recon-strip MSE 0.0037.
  - latent_cos **0.217** (threshold <0.7 = collapse escaped) — healthy, well clear.
  - pred_std **0.362** (>0.04 = real content, not mean-image collapse).
  - LPIPS 0.013.
  - recon.png: crisp grid lines, saturated distinct backgrounds (red/purple/blue), ball square localized
    and color-correct in the reconstruction row; no gray-mush / black-collapse.
- **Surprise: none.** Clean expected result. (Note: not directly comparable to the ESC-016 local smoke
  val 0.00216 — that was a 300-ep subset on the PRE-D-038 geometry; this is full data on the new 6×6/10px
  anti-overfit geometry, a harder recon target. latent_cos 0.217 here is BETTER than the smoke's 0.37.)
- **Mid-training instability blip (~epoch 11):** W&B sparklines show val/mse spiking to max with latent_cos
  and pred_std dipping together at ~ep11, then FULL recovery and clean convergence. The tokenizer's grad
  clipping (max_norm=1.0, train_tokenizer.py:600) let it self-correct. Flagged as motivation: the DYNAMICS
  model is currently UNclipped (train_dynamics.py:466–468) and has a riskier loss — clip it before its run.
- **Hypothesis impact:** infrastructure/backbone milestone (H-gridworld enabler), not a hypothesis test.
  This is the frozen tokenizer the GridWorld dynamics (H2/H3 memory work on the clean discrete env) builds on.
- **Tripwires checked:** D-039 tripwires all clear — no latent collapse (cos 0.217, pstd 0.362), recon keeps
  the square + colors (the 10px-vs-8px patch change did NOT hurt recon), val MSE converged by 30ep. None fired.
- **Next:** present-then-stop for Merlin. On bless → freeze as `checkpoints/gridworld/tokenizer.pt` and
  proceed to the vanilla GridWorld dynamics run (record a decision; apply grad-clip fix first; re-profile batch).
