# recall.py — the env-based GridWorld memory eval: given a model, perform a rollout through occlusion, score retention.

A scorer to answer the question: how well and long can the model accurately model the environment even though it cant see the state.
We start with `n_ctx` revealed env-generated ground truth frames and then let the model perform a long rollout with only action=1=occlusion, periodically checking what the model would predict if we do action=0=reveal state and scoring this belief against the ground truth of the independently running env. After checking that we discard the action=0 (reveal) generation and continue the long rollout of action=1 (occlusion) until the next periodic evaluation. This lets us score all k`s with just a single rollout.
In practice we run 64 rollouts to get a more accurate stochasticity based results on the binary score features.
We score colors and square position match.

`n_ctx`: length of sliding context window
`K`: number of shortcut diffusion steps
`k`: length of occlusion

## Interface
- `recall(model, tokenizer, *, n_ctx=4, max_k, n_rollouts=64, K=4, device, window=None) -> dict` — THE
  entry point. `window` (total frames) optionally FORCES a shorter sliding context window than the model
  trained with (e.g. 8 vs 16) to probe memory under a tighter window; None = native `max_temporal_length`.
  One long occluded rollout per seed, a reveal-belief scored at k e {2, 4, 6, 8, 10, 12, 14, 8*2=16, 8*3=24, 8*4=32, 8*i,  ..., max_k}; returns per-k curves
  `{"model": {position_acc, position_score, color_acc each {k:v}}, "copy_last":…, "oracle":…, "chance":…}`.
- Helpers (internal): `roll_and_score(model, tokenizer, seed, n_ctx, max_k, K) -> per-event records`;
  `score_reveal(pred_frame, true_state, colors) -> {pos_correct, pos_score, color_correct}`.
- `__main__` CLI (run + serialize, keeps `recall()` pure): `--checkpoint --tokenizer --max-k [--n-ctx
  --n-rollouts --K --window --out]` loads a checkpoint, runs `recall()`, and writes the curves + a `meta`
  block (n_ctx, max_k, n_rollouts, K, n_memory, window, native_window) to JSON (default
  `outputs/recall/recall_<stem>.json`). `--max-k` may exceed the window (the rollout slides/evicts);
  `--window` forces a shorter window. That JSON is what `plot_recall.py` consumes.

## Behavior
- Per seed: reset env; step `n_ctx` REVEALED frames (action=0) = the model's observed context;
  tokenizer-encode them to latents. The env keeps advancing independently — its physics is
  action-independent, so it yields the true square cell at every future step (the ground truth).
- One long rollout, for k = 1..max_k:
  1. advance the MAIN rollout one OCCLUDED step (action=1): `model.generate` one latent (carrying memory; technical rollout logic managed by model).
  2. BRANCH (read-only): from the current carried state, predict ONE REVEAL latent (action=0), decode it,
     `read_square`, and score against the env's true state at step k → the model's belief at occlusion k.
  3. DISCARD the reveal branch; continue the occluded rollout. ⇒ one rollout scores every k.
- Aggregate over `n_rollouts` seeds (binary scores → P(correct) per k): position exact (chance 1/36); square colour 4-way.
- References, same readout: `chance` analytic.

## Invariants
- ONE long occluded rollout scores all k — the reveal branches are **read-only** and must NOT advance or
  corrupt the main rollout's carried memory / latent window. (This is the whole efficiency point; never
  re-roll per k.)
- The env advances physics independently to supply true state at each k (the curtain never affects physics).
- Roll with `model.generate` (the carrying inference) feeding the true actions; read ONLY the predicted
  reveal frame (occluded frames are never read for the answer).
