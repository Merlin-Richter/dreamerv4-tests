# EXP-027 recall eval — dynamics-rollout frame source (design, to verify before trusting numbers)

Goal: score the trained dynamics model's recall of the hidden square at reveal frames, vs occlusion
length k, through the FROZEN recall core (D-045), comparable to the EXP-026 oracle/copy-last reference.
Run on the 150 HELD-OUT val episodes (perm seed-0 split, train_dynamics.py:353).

## Faithful inference protocol (per reveal event) — "look away for k, look back"
A reveal event = (last_visible_t = lv [last curtain-UP frame], k occluded frames, reveal_t = lv+k+1).
To predict the reveal frame the model must retain the square through k hidden steps.

1. Encode the TRUE frames once per episode: z1 = tokenizer.encoder(frames)  (frozen, no grad).
2. Context = true latents ending at lv: ctx = z1[:, a:lv+1], a = max(0, lv - (max_temporal_length-1)).
   (generate() windows to the last max_ctx=15 internally; occluded context frames carry only the
   curtain — that IS what the model observes, so feeding their true latents is faithful, not a leak.)
3. n_generate = reveal_t - lv  (k occluded + 1 reveal). Roll out:
     gen = model.generate_cached(ctx, n_generate, action_idx = curtain_ids[a : reveal_t+1])
   action_idx is the TRUE curtain channel (0=up/1=down) aligned to [context...generated] — the model
   is told each frame's curtain (legit: the agent chooses the curtain), it must remember the POSITION.
   generate_cached is bit-identical to generate, ~Kx faster. context_signal = config default 0.9
   (faithful default; the EXP-022 lever is a later ablation, not the baseline number).
4. Predicted reveal latent = gen[:, -1] (absolute frame reveal_t).
5. Decode in a temporal window (decoder is temporal — never decode a latent in isolation):
     full = concat(ctx, gen); win = full[:, -max_temporal_length:]; frames = tokenizer.decoder(win)[0]
     reveal_frame = frames[-1]   # uint8 BGR, NO channel swap (decoder output matches BGR data)
6. score: read_square(reveal_frame) vs states[reveal_t]/colors → score_episode-style record with k.

Aggregate per-k through the frozen aggregate(); compare to oracle (==1.0) and copy-last.

## Why this matches the baselines already in the frozen eval
copy-last freezes the square at its last curtain-UP cell (= lv) → its belief origin is exactly lv, the
same origin as this rollout's last true context frame. So "model beats copy-last per-k" is apples-to-
apples. For k >= window (15), lv scrolls out of the model's 15-frame window → vanilla CANNOT see the
last-observed frame when predicting the reveal → expected cliff toward copy-last/chance. That cliff is
the baseline result; memory methods must push it right.

## Matched-horizon open-rollout control (separates "can't track motion" from "memory lost")
Same rollout but with the curtain held UP for the same horizon (model runs free in the clear for k
steps). If the model tracks motion well in the clear but fails under occlusion → memory loss (the point).
If it fails even in the clear → it never learned the dynamics (a capacity/budget problem, not memory).

## Risks the verifier should check
- action_idx alignment vs generate()'s internal new_idx indexing (off-by-one would feed wrong curtain).
- Decoding window: is reading frames[-1] of the last-16 window the correct reveal frame? (vs decoding
  the full sequence; the temporal decoder is causal so the last frame attends only to earlier — OK.)
- Is feeding TRUE latents for occluded context frames a leak? (Argued no: curtain frames carry no
  square; but confirm the occluded frame's latent doesn't encode the hidden square — EXP-024 showed
  the tokenizer drops the ball when occluded; here occluded frames are pure curtain anyway.)
- Channel order (BGR, no swap) — EXP-026 confirmed.
