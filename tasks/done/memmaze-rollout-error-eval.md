# Quantitative Memory Maze rollout-error evaluation

## Goal
Create a quantitative Memory Maze evaluation that measures how ordinary visual prediction error grows
over a short autoregressive rollout. This is intended to compare the rollout quality of vanilla,
memory-token, and archive-capable dynamics models on the same held-out data.

## Evaluation protocol
- Evaluate on multiple episodes from the Memory Maze evaluation split.
- Warm up each model by moving its normal sliding context window through 128 sequential ground-truth
  frames and their corresponding actions before scoring. This is a 128-frame streamed prefill, not a
  request to enlarge the model's context window to 128 frames. As old frames leave the sliding window,
  memory and archive mechanisms remain free to propagate information from them.
- After the prefix, generate the next 32 frames autoregressively using the ground-truth action sequence.
  Generated frames, rather than ground-truth frames, become the visual history during this scored rollout.
- Decode every generated frame and compare it with the corresponding real ground-truth frame using
  pixel-space mean squared error.
- For each rollout horizon from 1 through 32, average the error across the evaluated episodes.
- Use the same episodes, prefix locations, actions, rollout settings, and tokenizer for every model in a
  comparison.

## Throughput requirement
Run episodes in parallel at the highest batch size supported by the available accelerator and evaluated
model. The full 128-frame streamed prefill and 32-frame rollout must remain batched across episodes; a
serial episode-by-episode evaluation is not an acceptable fallback because it would make the instrument
impractical to use. Batch size may differ between model families when their memory requirements differ,
but this must not change the evaluation protocol or results.

## Outputs
Save a reusable result artifact for each evaluated model. It must contain the mean pixel MSE at every
rollout horizon and enough evaluation metadata to establish that different model results are directly
comparable. Results should be plottable later without rerunning model inference.

## Interpretation
This is a short-horizon visual-error instrument, not a perfect measure of world-model correctness.
Autoregressive butterfly effects can give a high pixel error to a rollout that remains visually plausible
but diverges slightly from the exact recorded trajectory. Restricting the scored rollout to 32 frames is
intended to keep that limitation manageable. Claims based on this evaluation should retain this caveat.

## Done means
- Multiple held-out Memory Maze episodes can be evaluated with the 128-frame prefix and 32-frame scored
  rollout protocol above.
- Memory and archive models receive a genuine 128-frame ground-truth streamed prefill through the normal
  sliding context window, without increasing that window to 128 frames.
- Evaluation runs episodes in parallel at the highest feasible batch size for each model and accelerator.
- Per-horizon decoded pixel MSE is averaged across episodes and saved as a reusable result artifact.
- The evaluation treats compared model families consistently and records sufficient provenance for a fair
  comparison.
- A small validation run demonstrates correct frame/action alignment and produces finite results for all
  32 rollout horizons.
