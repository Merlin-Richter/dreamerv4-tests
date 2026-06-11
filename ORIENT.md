# ORIENT.md

Rewritten: 2026-06-11

## What we are doing and why

We reproduce a DreamerV4-style pipeline (frozen video tokenizer + shortcut-forcing
dynamics model) on CurtainsEnv to then test whether reconstruction-only encoding
objectives fail to retain occluded state (H2) and whether a retention-forcing
objective fixes it (H3). See GOAL.md. The tokenizer is good (EXP-006: VGG-LPIPS,
val/mse 1.41e-4, frozen as `trained_autoencoder.pt`). The dynamics model is the
problem: EXP-007 trained to healthy val/loss (1.93e-3) but rollouts randomize ball
color and position immediately, even with fully visible context. Merlin's verdict
(ESC-001): diagnose this before building the §8 probe suite.

## In flight

Nothing. No cluster jobs running (wrappers don't exist yet — T-003; all cluster
access manual via Merlin so far). No workers.

## Next action

Spawn T-001 (diagnosis of the EXP-007 failure) per D-009: latent-geometry
measurements on the frozen tokenizer, shortcut-forcing implementation audit,
context ablations on `my_dynamics.pt`. Local (4070). Before that, commit the
uncommitted working-tree changes (T-005) so the diagnosis has clean provenance.

## Current worries

1. **val/loss is a poor proxy** — EXP-007 proves the training loss can look fine
   while rollouts are useless. Until the probe suite (T-002) exists we have no
   trustworthy quantitative signal on dynamics quality.
2. **Provenance debt**: pre-2026-06-09 history is approximate (no logging); EXP-007
   rollout images were never archived; EXP-006's A/B was confounded (commit, batch,
   host). Backfill marks these honestly; going forward the protocol applies.
3. **EXP-005's "plausible rollouts" were only eyeballed** — the unconditional
   baseline may have had the same defect, undetected.
