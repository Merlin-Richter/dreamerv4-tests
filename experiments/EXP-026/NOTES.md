# EXP-026 — GridWorld tokenizer-roundtrip recall CEILING

Purpose: D-044 tripwire. Before training any GridWorld dynamics model, check the FROZEN tokenizer's
latent can represent the square's exact cell + colour at reveal frames. The roundtrip (encode→decode
of the TRUE frames) is the upper bound on what any dynamics model on this latent can recall — the
dynamics model predicts latents that get decoded by the same decoder.

## Setup
- Tokenizer: checkpoints/gridworld/tokenizer.pt (EXP-025, D-044; W&B 70k76148 @ f1e3d6c). window L=16.
- Data: gridworld.npy, first 500 of 3000 episodes (×200 frames). BGR (verified — the train_tokenizer
  "RGB" comments are mislabeled; dataset read by read_square directly with no channel flip).
- Method: each episode encode→decode in consecutive L=16 windows (tail clamped to [T-L,T]); recon
  scored through the FROZEN recall core (D-045: read_square + score_episode + aggregate). Compared
  per-k to oracle (true frames) and copy-last (no-memory). ~13k reveal events (k=1: n=6781 ... k=42: n=1).

## Reconciliation
Expected (D-044): tokenizer faithfully autoencodes a visible square → ceiling near oracle; if it
smeared the square the discrete readout would drop below 1.0 at some k.
Observed: tokenizer-roundtrip == oracle == **1.0000 at EVERY k** for position_score, position_acc
(exact 1/36), color_acc, bg_acc. Zero (k, metric) cases where tok < oracle. Copy-last behaves exactly
as the periodicity finding predicts: position_acc spikes to 1.0 at k≡9 (mod 10) (k=9,19,29,39, n large)
and sits at ~chance otherwise; color/bg copy-last = 1.0 at all k (static attributes).
Surprise: none (favourable — clean confirmation).
Hypothesis impact: D-044 tripwire CLEARED — the frozen latent is NOT the bottleneck. Any downstream
recall failure will be the dynamics model's memory, not the tokenizer's representational capacity.
Tripwires checked: D-044 "latents don't preserve enough to read out at reveal frames" → NOT triggered
(perfect readout). No new tripwire triggered.

## Implication for the eval (worth stating)
In GridWorld, **position is the ONLY memory metric.** The square's colour never changes, so copy-last
(freeze last-seen) already scores color_acc = bg_acc = 1.0 at every k — colour cannot discriminate
memory here (unlike the occluded line, where colour was the hidden static attribute). This vindicates
D-033's position-first headline: the dynamic-memory signal lives entirely in position vs k, judged
against copy-last per-k (D-045). Colour/bg stay as cheap identity/confound checks.

## Caveat (honest)
The ceiling tests that the decoder faithfully renders a square the encoder SAW (the square is in the
input at reveal frames). It does not test whether the latent manifold is smooth/predictable enough for
a dynamics model to HIT the right latent — that is a dynamics-model question, out of scope for a
representational ceiling. What it does prove: decode(z) is faithful for the true z, so the decode step
adds no error; the open question is purely whether dynamics can predict z.

## Next
Tokenizer + eval are both green and the latent is uncapped. Path: train the vanilla GridWorld dynamics
baseline (cluster), then wire the dynamics-rollout frame source into adapter.py and run the real recall
curves vs k against this oracle/copy-last reference.
