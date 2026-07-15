# ColorField pixel-v3 dynamics arms

Three final checkpoints on the accepted `colorfield-pixel-v3` tokenizer:

1. vanilla, no memory, sustained train-only `p=0.5` tau-zero/d-min anchor;
2. rollout-only fast memory, no FF9;
3. the same rollout-only base plus 16-to-1 per-slot archive compression, no FF9.

## Window and comparison contract

- Model/eval window is pinned at `W=16`.
- Every mem2mem training slide uses `n_ctx=16`; sampling `{4,8,16}` is disabled.
- Both memory arms use 256-frame clips, the complete 16-frame window, eight-frame advances, and
  blockwise backward at TBPTT 32. Dense activation memory is bounded; compute still scales with all
  30 slides in the clip. The archive compressor uses `R=1` token per memory slot.
- The memory arms use a matched fork: a shared 5k-step rollout-only base, then a fresh optimizer
  on both a 5k plain continuation and a 5k archive continuation. Thus both final memory models
  receive 10k optimizer steps from the identical initialization/data recipe. A 256-frame step has
  30 rollout slides versus six at clip 64, so this is 50k clip-64 slide-loss equivalents.
- The 10k memory / 40k vanilla targets are a cost-controlled first gate. Extend both fork arms equally only if frozen eval
  shows the new zoomed environment has not converged.

Measured on the qualified Vast RTX 5090 at batch 128: vanilla is about 45 steps/s, rollout-only
is about 2.36 steps/s, and archive is about 1.88 steps/s. Neither 256-frame arm OOMed (peak device
memory was approximately 9.1 GiB rollout-only and 9.7 GiB archive). `run_sequence.sh` therefore
budgets 20, 45, 45, and 60 minutes respectively, while expected wall time is about 2h10 total.

All dynamics models use the small pixel-curve backbone (`E=128`, depth 6, heads 8, four memory
tokens where applicable), pure finest-step flow for rollout training, and `ff9_k=0`.
