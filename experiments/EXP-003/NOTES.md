# EXP-003 — Temporal autoencoder on BouncingBall: latent collapse (backfilled)

Decision: D-003 | Hypothesis: H1 | late May 2026, local
Provenance: approximate; commit 60a4b67 "still has MAE latent collapse"
(2026-05-31) marks the diagnosed state. Data: `bouncing.npy`.

Purpose: temporal AE (C) with MAE dropout, alternating spatial/temporal layers.

Expected: faithful recon through the bottleneck across frames.
Observed: **latent collapse.** The black background + small ball makes all-black
prediction a deep MSE optimum; the decoder output collapsed toward black and
latents of different frames had pairwise cosine similarity near 1 (verified with an
ad-hoc cos-sim inspection — predecessor of the `latent_cos` metric).
Surprise: high (at the time).
Hypothesis impact: H1 blocked — negative result, environment-induced.
Tripwires checked: n/a (pre-protocol).
Next: new decision needed → D-004 (CurtainsEnv with dense backgrounds).

This is a first-class negative result: reconstruction objectives follow the data's
loss landscape, not our intent.
