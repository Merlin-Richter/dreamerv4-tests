# EXP-002 — Single-image autoencoder on BouncingBall (backfilled)

Decision: D-003 | Hypothesis: H1 | ~mid May 2026, local
Provenance: pre-git; code at `src/B_single_image_auto_encoder/`. Data:
`bouncing.npy` (2026-05-17). No metrics archived.

Purpose: baseline frame-only AE — image → patches → attention → 4×bottleneck →
decoder — before adding the temporal dimension.

Observed: reconstructions acceptable (qualitative, interactive OpenCV viewer).

Reconciliation (retroactive): expected faithful recon, got it. Surprise: none.
Hypothesis impact: H1 component validated. Next: temporal AE (EXP-003).
