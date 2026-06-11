# EXP-004 — Temporal autoencoder on CurtainsEnv: collapse resolved (backfilled)

Decision: D-004 | Hypothesis: H1 | 2026-05-31 .. 2026-06-10, local (ZaubererPC)
Provenance: spans commits 9019835..7cb30c1; iteration runs from 2026-06-09 logged
to W&B project `transformer-C-tokenizer`:

| run | commit | state | note |
|---|---|---|---|
| vxhqzyli (glorious-dew-1) | 7cb30c1 | finished | val/mse 7.95e-4, latent_cos 0.195 |
| 4jlvep00 (feasible-fire-2) | 7cb30c1 | failed (73s) | iteration debris |
| o5pw6f75 (young-oath-3) | 7cb30c1 | killed | latent_cos 0.146 |
| g4o7kf4h (fanciful-pine-4) | 7cb30c1 | killed (40s) | iteration debris |
| kdfzo1es (fearless-grass-5) | 7cb30c1 | crashed @ epoch 39 | latent_cos **0.031**, val/mse 7.89e-4 |
| arnp5mis (astral-cherry-6) | 29ecca7 | failed (19s) | iteration debris |

Pre-2026-06-09 training (the actual collapse fix) was not logged; details are from
memory ("a couple other adjustments I no longer remember" — acknowledged provenance
gap, see D-004).

Expected: no collapse on dense gradient backgrounds.
Observed: collapse resolved. latent_cos drops from 0.195 to 0.031 across
iterations; reconstructions retain ball and background (qualitative). val/mse
plateaued ~7.9e-4 locally — broken later by longer cluster training + D-007
changes (see EXP-006).
Surprise: none (after D-004's reasoning).
Hypothesis impact: H1 tokenizer component unblocked.
Next: proceed → dynamics (EXP-005), tokenizer upgrade (EXP-006).
