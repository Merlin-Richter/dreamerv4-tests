# Experiment: Testing new training to teach dynamics model on memory to memory writing

Currently the dynamics model only learns to write latents->memory and read memory->latents.
There currently is no training signal on memory->memory.

The practical goal of the memory token stays the same: predict the next n latents with just the memory token under random actions.
But we want to teach it to construct that memory token from other memory tokens.

To achive this, there need to be real completed memory tokens in context during training (currently they are not).

Proposed training idea:
Instead of taking random N-length slices from episodes, we use a sliding context window and rollout longer staying inside an episode and let the gradients flow through memorys to their now out of context construction (still batched at many episodes at once)

Example:
We first randomly pick some sliding context window size `n_ctx` out of {2^2, 2^3, 2^i, ..., N} (where N is the models max context)
We start with some episode at t=0 and take the next `n_ctx` frames and place them at near-clean signal levels into context and memory tokens learned blank. Then run the forward pass, to init the memory tokens. Then we have a context of size `n_ctx` with near-clean latents and real computed memory tokens. Then we slide the context window by `n_ctx/2` forward, evicting the oldest half and adding the new either randomly noised or all fully noised latents and blank memory tokens for the new half the context window while the old half is random_pick(near-clean ground truth latents, full noise latents) and real memory tokens.
We will use 50% chance that old latents are near-clean and new ones have some signal and 50% chance that all latents are full noise, forcing memory to account for everything. This random event is completely independent at each forward pass and random across the batch.
This is the starting point of the mem->mem training. half the context contains memory tokens, half doesn't.
Then we run the forward pass to denoise the latents and predict the new memory tokens. Then we apply the usual flow matching loss on latents and single-memory ff9 k reconstruction loss for all the new memory tokens, and let the gradient flow through its contrcution and preferably back through the past 2*N time steps using truncated bptt. We only do flow matching and FF9 loss on the new half of the context, the old half only provides rollout-like context without contributing any loss directly.
Its really crucial to get the autograd right on the memory tokens which first get constructed and then used on the next forward pass where its information propagates through space and time both repeatedly when the new tokens use it for construction. That contribution of the constructed memory token needs to backprop to the construction pass of that memory token. pytorch autograd should handle this.
Then repeat by sliding the the context window again by `n_ctx/2` such that we now make the old half contain the just computed memory tokens together with the near-clean latents (not the noisy one we just used) to construct the next batch and continue this repeatedly until 5 * N or 10 * n_ctx or episode end (whichever comes first).
This does not require or even benefit from a KV cache and therefore should not implement it. It will require caching/storing of grads (which should happen automatically) and the detaching at certian depths (2*N) to keep memory footprint under controll 

This new training rollout needs its own code.

Correctness check (must-have, not optional): autograd can silently break here — a stray no_grad
or in-place op on the carried memory would zero the relay gradient with no error, and the only
symptom would be that the eval never moves. So before trusting any run: build a window where a
memory token's grounding latent has been evicted, run backward, and assert a nonzero grad-norm at
that token's *construction* pass. This single test is what separates "training mem->mem" from
"silently training nothing."

Known residual (teacher forcing): the near-clean mode trains relay on ground-truth latents, but at
inference the visible scene is the model's own committed prediction, which carries error. The
full-noise mode does NOT fix this (it removes latents, not prediction error). It is consistent with
how the model is already trained (shortcut forcing denoises GT latents), so we keep it — but if long
rollouts still drift despite this training, the first lever is free-running: re-present the model's
own predicted latent (z_hat1) on commit instead of the GT latent.

Note: 
- Even though i think this should be sufficient as the only training signal, for safely we will run it in addition to the normal training with maybe a 50/50 wall clock split.
- During a batched rollout, all rollouts in the batch need to be of same `n_ctx` for GPU parallization to work properly.

Try to keep src/ untouched when running experiments.

---

## STATUS (2026-06-26) — DONE (implemented, verified, run; clear positive result — see RESULT below)

Implemented in `experiments/mem2mem/` (src/ untouched; imports the unmodified DynamicsModel +
train_dynamics helpers). See `experiments/mem2mem/NOTES.md`.
- `rollout.py` — the sliding-window mem→mem loss: real graph-attached old-half memory → construct
  new-half memory, flow + FF9 loss on the NEW half only, 50/50 clean/full-noise modes, TBPTT detach at 2N.
  Mechanism: temporal attention is causal per slot, so new memory is built by attending to old memory;
  gradient relays new→old memory construction. No KV cache (full recompute).
- `train_mem2mem.py` — standalone trainer, 50/50 normal + mem→mem, n_ctx∈{4,8,…,N} per batch, fixed LR
  schedule, chunk-encodes long clips through the 16-frame tokenizer window.
- **`test_autograd.py` (the must-have) PASSES**: an evicted grounding latent gets grad 3.25e-3 via the
  relay and **exactly 0.0** when the relay is detached → the construction pass genuinely backprops.
  (Caught one real bug en route.) End-to-end GPU smoke passes (losses decrease).

Open caveats in NOTES.md: TBPTT is relay-depth-bounded but not yet footprint-bounded (segmented backward
is the fix if it OOMs at scale); teacher-forcing residual noted.

## RESULT (2026-06-26) — DONE, the experiment WORKS.
Cluster run (job 410376, bs64 clip64 50ep, val 0.0027) -> `checkpoints/gridworld/dynamics_mem2mem.pt`.
3-way recall A/B (EXPERIMENTS `mem2mem-train`, `experiments/recall-ab/results_mem2mem_3way.json`):
- **mem2mem holds position recall ~0.96 FLAT through k=20**, where FF9 decays past k≈12
  (0.70@k12 → 0.14@k20). Long-horizon tail (k≥14) pos_acc: vanilla 0.03 / FF9 0.20 / **mem2mem 0.96**.
- The memory→memory training signal carries hidden position state **indefinitely past the window** —
  exactly the limitation the whole project targets. Not an env-periodicity artifact (uniformly high
  across all k). This is the headline positive result of the campaign so far.

Possible follow-ons (Merlin's call, NOT done): graduate the idea into `src/`+spec if it's keeping;
ablate (mem2mem-only vs 50/50; n_ctx schedule; segmented-backward TBPTT for footprint); test on a
harder env; longer max_k to find where mem2mem finally breaks.