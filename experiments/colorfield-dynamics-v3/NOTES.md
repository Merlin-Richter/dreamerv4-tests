# ColorField pixel-v3 dynamics arms

Three final checkpoints on the accepted `colorfield-pixel-v3` tokenizer:

1. vanilla, no memory, sustained train-only `p=0.5` tau-zero/d-min anchor;
2. rollout-only fast memory, no FF9;
3. the same rollout-only base plus 16-to-1 per-slot archive compression, no FF9.

## Window and comparison contract

- Model/eval window is pinned at `W=16`.
- Every mem2mem training slide uses `n_ctx=16`; sampling `{4,8,16}` is disabled.
- Archive training also uses the complete 16-frame window and advances by eight frames.
- The memory arms use a matched fork: a shared 20k-step rollout-only base, then a fresh optimizer
  on both a 20k plain continuation and a 20k archive continuation. Thus both final memory models
  receive 40k optimizer steps from the identical initialization/data recipe.
- The 40k target is a cost-controlled first gate. Extend both fork arms equally only if frozen eval
  shows the new zoomed environment has not converged.

All dynamics models use the small pixel-curve backbone (`E=128`, depth 6, heads 8, four memory
tokens where applicable), pure finest-step flow for rollout training, and `ff9_k=0`.
