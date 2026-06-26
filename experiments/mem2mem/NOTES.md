# mem→mem training — teaching memory tokens to be built from prior memory tokens

Task: `tasks/in-progress/test-new-memory-training.md`. Keeps `src/` untouched (imports the unmodified
`DynamicsModel` + `train_dynamics` helpers).

## Problem
The dynamics model gets a **latents→memory** signal (the windowed pass writes memory) and a
**memory→latents** signal (FF9 reconstructs future frames from memory). It gets **no memory→memory**
signal: during training the memory tokens present in context are always the learned-blank init, never
real *computed* ones. So it never learns to construct a memory token by reading earlier memory tokens —
which is exactly what the carrying rollout asks it to do at inference (read old memory K/V, write new).

## Mechanism (why this works without new model code)
Temporal attention in `DynamicsModel` is causal **per token slot**, so a frame's MEMORY slot attends to
the memory slots of earlier frames. Therefore, if we inject **real, graph-attached** memory tokens into
the OLD half of a window and let the model construct the NEW half's memory (loss only on the new half),
the new memory is *built by attending to the old memory*, and backprop flows new-memory → old-memory
**construction**. Slide the window by `n_ctx/2`, carry the new half's memory forward as the next old
half, and the gradient relays back through the memory chain (truncated at ~2N frames). No KV cache (it's
training — full recompute each window).

## Implementation
- `rollout.py :: mem2mem_rollout_loss(model, z1, actions, n_ctx, ...)` — one sliding rollout over a long
  clip. Per slide: old half = real carried memory (+ near-clean GT or full-noise latents), new half =
  blank memory (+ noised latents); forward → flow loss + FF9-sufficiency loss on the NEW half only;
  carry the new memory forward. **Noise modes** (independent per batch element, the task's 50/50):
  "clean" (old near-clean GT, new sampled-signal) and "noise" (ALL latents pure noise → the new half can
  only be reconstructed from memory). **TBPTT:** carried memory is `.detach()`-ed once the relay graph
  exceeds `tbptt_frames` (default 2N). Reuses the model's tested `_ff9_loss` for the new memories.
- `train_mem2mem.py` — standalone trainer: 50/50 per-batch mix of the normal shortcut-forcing loss (on a
  random ≤N window) and the mem→mem rollout. `n_ctx` sampled per batch from {4,8,…,N} (same across the
  batch, for GPU parallelism). Same warmup→flat→late-cosine LR schedule as the fixed `train_dynamics`.
  Long clips are **chunk-encoded** in tokenizer-window (16-frame) blocks (the tokenizer's RoPE table only
  spans its `max_temporal_length`). `--max-episodes` for smoke; `--max-frames` to cap rollout footprint.

## Verification
- **`test_autograd.py` (the must-have) PASSES.** In forced full-noise mode a grounding latent (frame 0,
  present only in the init window, never in a loss-bearing new half) can reach the loss ONLY through the
  memory relay. Result: `|grad z1[frame0]| = 3.25e-3` with the relay, **exactly 0.0** when the relay is
  detached (`tbptt_frames=0`). So the gradient genuinely flows new-memory → evicted memory's
  construction; nothing is silently severed. (This test already caught one real bug — a `x or default`
  idiom that turned `tbptt_frames=0` into the default.)
- **End-to-end smoke (GPU) PASSES.** 2 epochs / 40 episodes: both losses backprop and decrease
  (mem2mem 0.51→0.38, flow 0.27→0.20, ff9 0.79→0.61; val 0.23→0.13). Checkpoint saves.

## How to run a real experiment (cluster)
```
python -u experiments/mem2mem/train_mem2mem.py \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics_mem2mem.pt --epochs 50 --batch-size 64 --clip-len 64 \
  --ff9 3 --n-memory 4 --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem
```
Then `experiments/recall-ab/run.py` (add the mem2mem checkpoint) to A/B vs the FF9 and vanilla baselines
— the question is whether mem→mem lifts **position** recall past the window (the null the first A/B found).

## Open items / honest caveats (for Merlin)
- **TBPTT is relay-depth-bounded, not footprint-bounded yet.** The detach at 2N bounds how far gradient
  flows, but because the trainer sums all slide losses and backwards once, every slide's forward graph is
  alive at backward → memory ~ O(#slides). Mitigate with `--max-frames` / smaller `--batch-size`; the
  proper fix is segmented backward (backward + free each 2N-frame segment). Add if it OOMs at scale.
- **Teacher forcing residual** (noted in the task): near-clean mode trains the relay on GT latents, but at
  inference the visible scene is the model's own committed prediction. Consistent with how the model is
  already trained; if long rollouts still drift, the lever is free-running (re-present `z_hat1` on commit).
- Not yet run at scale / not yet compared on recall — that's the experiment, pending the r2 baselines.
