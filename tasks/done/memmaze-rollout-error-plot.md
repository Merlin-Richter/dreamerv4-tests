# Compare Memory Maze rollout-error curves

## Goal
Create the comparison view for the quantitative Memory Maze rollout-error evaluation. The graph should
make it easy to compare how visual error accumulates over rollout length for several dynamics models.

## Prerequisite
The `memmaze-rollout-error-eval` task must produce reusable per-model results from the same evaluation
protocol: 128 sequentially committed ground-truth prefix frames followed by 32 autoregressively generated
and scored frames. The 128 frames are streamed through the model's normal sliding context window; they do
not imply a 128-frame context window.

## Plot
- Load one or more saved evaluation results without rerunning inference.
- Plot rollout horizon on the x-axis, covering generated frames 1 through 32.
- Plot mean decoded pixel-space MSE across evaluation episodes on the y-axis.
- Draw one clearly labeled curve per model so vanilla, memory-token, archive, and future model variants can
  be overlaid in the same figure.
- Reject or clearly flag result series whose recorded evaluation settings are not directly comparable.
- Label the metric and protocol clearly enough that the graph can be interpreted without consulting the
  evaluation code.

The figure should carry the evaluation's central limitation: exact pixel error can penalize visually valid
rollouts after small trajectory divergences, so this is a short-horizon reconstruction comparison rather
than a complete measure of world-model quality.

## Done means
- Any selected set of compatible saved model results can be plotted together without reevaluation.
- The output graph has rollout length on the x-axis, mean pixel MSE on the y-axis, and an unambiguous model
  legend.
- Incompatible evaluation runs cannot be silently presented as a fair comparison.
- A comparison figure is produced for the available Memory Maze model checkpoints.
