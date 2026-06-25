# dynamics_model.py — the latent dynamics transformer

A block-causal transformer over a window of per-frame latent tokens, with the high-level goal of
accurately predicting future observations. It consumes the per-frame **tokenizer latents** (from a
separately-trained, frozen tokenizer) plus optional **actions**, and predicts the *clean* latents of
the next frame. Trained with **shortcut forcing**, so it generates each frame in 4 forward passes
without errors snowballing over long rollouts. It never sees pixels — it lives entirely in the
tokenizer's latent space `(B, T, n_latents, bottleneck_dim)`.

Our addition over vanilla Dreamer-4:
- optional per time step **memory tokens** that should encode the whole env state, so hidden state can survive past the latent window (Dreamer-4's open "long-horizon memory" problem). They are not getting carried. The model needs to predict the new memory tokens based on the old ones and the other context like actions and latents. Getting the model to actually do this is our goal.

Note that 'carry' does not mean that the tokens are carried physically forward in time, but rather that it allows the model to continue to write new tokens based on the old and carry that information through those tokens by repeated read and write.

---

## 1. What it consumes (per-frame token block)

Per timestep the transformer sees one block of tokens, assembled along the sequence axis as:

`[ action | latents (n_latents) | registers (n_registers) | memory (n_memory) | shortcut ]`

- **latents** — the (noised) frame latent, linearly projected to `embedding_dim`.
- **shortcut token** — encodes the shortcut conditioning: signal level `τ` and step size `d`, both
  discrete → two embedding lookups, concatenated into one token.
- **action token(s)** — a learned action embedding; if action-conditioned, the discrete action's
  per-frame feature is *added* to it. `n_actions=0` ⇒ only the learned embedding (lets it train on
  unlabeled video). For GridWorld the action is the curtain (2 values).
- **registers** — `n_registers` learned scratch tokens (base Dreamer-4 component; free working space).
- **memory** — `n_memory` tokens, our memory channel (§5). `n_memory=0` ⇒ omitted ⇒ vanilla model.

---

## 2. The transformer (space/time split)

A 2-D transformer with separate **space** and **time** attention, pre-RMSNorm, RoPE, SwiGLU MLPs, plus
**QK-norm** + **attention-logit soft-capping** + a learnable per-head logit scale for stability.

- **Spatial blocks** — full, *unmasked* self-attention over the tokens of a single frame (the block
  above). All token types mix here (so e.g. memory ↔ latents mix within a frame).
- **Temporal blocks** — applied only **once every 4 layers**; attention is **causal in time** and
  **position-wise per token slot** (RoPE on the time axis). So each slot — including each memory slot —
  is its own causal channel across frames, and does *not* see other-slot tokens at other times directly.
  Sparse temporal attention is cheaper and (Dreamer-4 finding) higher quality.

Consequence that matters for memory: latents and memory mix only *within a frame*; the temporal layers
relay each channel forward in time. (No GQA yet; the across-time KV-cache is covered in §4.)

---

## 3. Training objective: shortcut forcing

Combines **diffusion forcing** (a per-frame noise level) with **shortcut models** (conditioning on step
size so you can take big sampling steps). Let `f_θ` be the model.

**Setup.** Each frame `t` gets its own signal level `τ_t ∈ [0,1]` and step size `d_t = 1/K`. With `z_0`
noise, `z_1` the clean target latent:

```
z̃ = (1 − τ) z_0 + τ z_1            ẑ_1 = f_θ(z̃, τ, d, a)
```
`τ=0` is pure noise, `τ=1` is clean. (`t` = sequence index, `τ_t` = noise level — different axes.)

**x-prediction, not v-prediction (key).** `f_θ` predicts the **clean latent `ẑ_1`**, not the velocity.
v-prediction trains the net toward high-frequency outputs whose errors accumulate when you unroll one
frame at a time; x-prediction gives stable arbitrary-length rollouts. The loss is computed in x-space.

**The loss.** The flow term is plain MSE in x-space. The bootstrap term distills two half-steps (a
v-space quantity) and is scaled back to x-space by `(1−τ)²`:

```
b'  = ( f_θ(z̃, τ, d/2, a) − z̃ ) / (1 − τ)
z'  = z̃ + b'·(d/2)
b'' = ( f_θ(z', τ+d/2, d/2, a) − z' ) / (1 − (τ+d/2))

           ┌ ‖ ẑ_1 − z_1 ‖²                                     if d = d_min (finest step)
L  =       │
           └ (1−τ)² · ‖ (ẑ_1 − z̃)/(1−τ) − sg(b'+b'')/2 ‖²       otherwise   (sg = stop-grad)
```
At the finest step it's just flow-matching MSE on the clean target; for coarser steps it regresses the
model's velocity toward the average of the two bootstrapped half-step velocities. The `(1−τ)²` puts the
v-space bootstrap loss on the same scale as the x-space flow loss (`‖x̂_1−x_1‖² = (1−τ)²‖v̂−v‖²`).

**Ramp weight.** Low-`τ` frames carry little signal (flow collapses to the mean; bootstrap is easy), so
weight every term by `w(τ) = (1−ramp_min)·τ + ramp_min` (ramp_min≈0.1) to spend capacity where the
signal is.

---

## 4. Inference (autoregressive rollout)

Generation is autoregressive in time, over a sliding window of `max_temporal_length` frames. Each new
frame is denoised from pure noise in **K=4 shortcut steps** (`d=1/4`) — ~16× fewer than plain diffusion
forcing — with the in-window context held **near-clean** at signal `context_signal` (≈0.9).

- **Why corrupt the context slightly.** Holding context at a small corruption 1%-10% (≈0.1 noise → 0.9 signal)
  makes the model tolerant of imperfections in its *own* prior generations.
- **KV caching is across TIME, and a frame's cache entry is committed AFTER its 4 diffusion steps — not
  "for" them.** A frame's 4 diffusion steps merely *read* the existing cache of past committed frames (so
  we never recompute the past). The frame's OWN cache entry is then produced by a dedicated 5th forward
  pass (§5) that re-presents the frame exactly as it will be seen as context: held near-clean at
  `context_signal` with its correct memory token — NOT any of its noisy diffusion-step states. Because we
  use a sliding window + RoPE we must respect the rotation already baked into the cached K/V (we cannot
  change it without recomputing): rotate every token by its TOTAL (absolute) index in the running
  rollout, never its index within the current window. This is valid because RoPE depends only on the
  relative distance between tokens, not their absolute position.

---

## 5. Memory (our extension; subject to change)

The latent window evicts: once an informative frame slides out of the `max_temporal_length` window, its
state is gone unless it was relayed forward. Memory tokens are our bounded carrier for that state — the
DreamerV4 h-state analogue, aimed at the long-horizon-memory limitation.

- **Memory tokens** are an *activation* (a final-layer hidden state), with **no ground-truth label**.
- **FF9 sufficiency loss** (when `n_memory>0`, lookahead `ff9_k>0`): for each frame `t`, a short
  mini-window where the path latents are set to `τ=0` (pure noise, so no latent carries the scene) and
  the memory token written at `t` is injected; the model must reconstruct the next `1..ff9_k` frames
  from **memory alone**. This forces memory to contain the hidden (on/off-screen) state. Added to §3's
  loss with a weighted loss and using loss normalization (where the scaler is gradient detatched). The gradient of the reconstruction needs to flow back through the mechanism that constructed the memory tokens.
- **Carried inference (the load-bearing part).** At rollout, each frame's *written* memory token is
  taken on the last of the K=4 diffusion steps (when the frame is generated) and then placed at the input of itself (same time step). We also put the diffused frame latents at the input give it a near-clean signal level and run one final forward pass to get the frozen KV cache for the whole time step (this fifth forward pass to generate the KV cache can be done losslessly together with the first diffusion step of the next frame for a small optimization). Vanilla (`n_memory=0`) is the plain rollout with no carry.

---

## Interface
- `DynamicsModelConfig`: `embedding_dim, depth, n_heads, mlp_ratio, n_latents, bottleneck_dim,
  max_temporal_length`; `max_sampling_steps`(=K_max), `inference_steps`(=K), `context_signal`,
  `ramp_min`; `n_actions, n_registers, n_memory, ff9_k`.
- `forward(z_tilde, tau_idx, d_idx, actions=None, memory_in=None, return_memory=False) -> ẑ_1[, mem]`.
- `loss(z1, action_idx=None) -> scalar` — shortcut forcing (+ FF9 sufficiency when `n_memory>0`).
- `generate(context, n_generate, K=None, action_idx=None) -> latents` — carrying autoregressive rollout.

## Invariants
- x-prediction in x-space; Temporal attention causal; spatial
  unmasked within a frame; RoPE on time only. Memory is relayed by READING the old memory tokens' cached
  K/V and WRITING new ones each step (not threaded forward as a frozen activation).
- `n_memory=0` ⇒ vanilla (no memory tokens); `n_actions=0` ⇒ unconditioned. Frozen tokenizer latent
  space only; never sees pixels.
