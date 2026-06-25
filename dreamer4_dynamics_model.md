# How the Dreamer 4 Dynamics Model Works

The dynamics model is the predictive core of the world model. It takes the
interleaved sequence of **actions** and **tokenizer representations** (produced by a
separately-trained, frozen causal tokenizer) and predicts the *clean* representations
of future frames. It's trained with a **shortcut forcing** objective, which is what
lets it generate frames fast (4 forward passes per frame) without errors snowballing
over long rollouts.

It does *not* operate on raw pixels. The causal tokenizer compresses each video frame
into a small set of continuous latent representations first; the dynamics model lives
entirely in that latent space.

---

## 1. What it consumes (input tokenization)

At each timestep the dynamics transformer sees a block of tokens assembled from three
things:

**Representation tokens.** The (corrupted) latent representation of the frame is
linearly projected into `S_z` spatial tokens.

**Register tokens + a noise token.** Concatenated alongside are `S_r` learned register
tokens, plus a *single* token that encodes the shortcut conditioning — the signal level
`τ` and the step size `d`. Both `τ` and `d` are discrete, so each is an embedding
lookup and their channels are concatenated into that one token.

**Action tokens.** Actions can have several components (e.g. mouse and keyboard
separately). Each component is encoded into `S_a` tokens and the components are
**summed together** along with a learned embedding. Continuous components are linearly
projected; categorical/binary components use embedding lookups. When training on
*unlabeled* video (no actions), only the learned embedding is used — this is the
mechanism that lets the model absorb knowledge from action-free video and still ground
actions from a small labeled subset.

So the sequence fed to the transformer is, per frame, roughly:
`[ action tokens | noise/step token | register tokens | representation tokens ]`,
repeated and interleaved across time.

---

## 2. The transformer (shared "efficient transformer")

The dynamics model and the tokenizer use the **same** architecture: a 2D transformer
with separate **time** and **space** dimensions.

- **Block-causal attention.** Attention is masked to be causal *in time*: all tokens
  within a single timestep can attend to each other and to the past, but not the
  future. This is what makes interactive, frame-by-frame generation possible.
- **Base components.** Standard transformer with pre-layer RMSNorm, RoPE positional
  encoding, and SwiGLU MLPs, plus **QKNorm** and **attention-logit soft capping** for
  training stability.
- **Space/time split.** Instead of dense attention over all video tokens, it uses
  separate space-only and time-only attention layers.
- **Sparse temporal attention.** Temporal attention is applied only **once every 4
  layers**. This is cheaper *and* improves quality — the authors attribute the quality
  gain to the spatial inductive bias of focusing computation on the current frame.
- **GQA.** Grouped-query attention on the dynamics attention layers shrinks the KV
  cache (multiple query heads share key/value heads), which matters because long
  context + frame-by-frame decoding is memory-bandwidth bound.

For Minecraft the dynamics model runs with `N_z = 256` spatial tokens, context length
192 frames, and the dynamics portion is ~1.6B of the 2B total parameters.

---

## 3. The training objective: shortcut forcing

Shortcut forcing combines **diffusion forcing** (per-timestep noise levels in a
sequence) with **shortcut models** (conditioning on step size so you can take big
sampling steps). The dynamics model is the function `f_θ`.

### Setup

Each timestep `t` gets its own signal level `τ_t ∈ [0,1]` and step size `d_t`. Letting
`z_0` be noise, `z_1` the clean target representation, and `z̃` the corrupted input:

```
z_0 ~ N(0, 1)        z_1 ~ D        τ, d ~ p(τ, d)        τ, d ∈ [0,1]^T
ẑ_1 = f_θ(z̃, τ, d, a)             z̃ = (1 − τ) z_0 + τ z_1
```

`τ = 0` is pure noise, `τ = 1` is clean. Note `t` indexes the *sequence* timestep while
`τ_t` is the *noise* level at that step — two different axes.

### x-prediction, not v-prediction (a key decision)

Standard shortcut/flow models predict the **velocity** `v = x_1 − x_0` (v-prediction).
That works great when you generate a whole block jointly, but it trains the network to
emit **high-frequency outputs**. When you instead unroll one frame at a time for a long
video, those high-frequency errors accumulate and the rollout degrades.

Dreamer 4 parameterizes `f_θ` to predict the **clean representation `ẑ_1`**
(x-prediction). This gives stable rollouts of arbitrary length. The ablation shows it
plainly: switching to x-prediction + computing the loss in x-space drops FVD from ~326
to 151, and the final model's FVD is **57** vs **124** for the otherwise-identical model
using v-space prediction/losses.

### The loss

The flow term is computed directly in x-space (just MSE to the clean target). The
bootstrap term — which distills two half-steps into one — is naturally a v-space
quantity, so it's computed in v-space and then scaled back into x-space.

Bootstrap half-steps:

```
b'  = ( f_θ(z̃, τ, d/2, a) − z_τ ) / (1 − τ)
z'  = z̃ + b' · d/2
b'' = ( f_θ(z', τ + d/2, d/2, a) − z' ) / (1 − (τ + d/2))
```

Loss (sg = stop-gradient):

```
            ┌ ‖ ẑ_1 − z_1 ‖²                                          if d = d_min
L(θ)  =     │
            └ (1 − τ)² · ‖ (ẑ_1 − z̃)/(1 − τ) − sg(b' + b'')/2 ‖²      otherwise
```

At the finest step size `d_min` you just have the flow-matching MSE on the clean target.
For larger steps you regress the model's predicted velocity toward the average of the
two bootstrapped half-step velocities.

**Why the `(1 − τ)²` factor.** Predictions are made in x-space but the bootstrap target
is a velocity. The two MSEs are related by

```
‖ x̂_1 − x_1 ‖²  =  (1 − τ)² · ‖ v̂_τ − v_τ ‖²,   where  v̂_τ = (x̂_1 − x_τ)/(1 − τ)
```

so multiplying the v-space bootstrap loss by `(1 − τ)²` puts it on the same scale as the
x-space flow loss. (The paper writes the averaged bootstrap target as `sg(b_1 + b_2)`,
which is the same `b'`, `b''` defined above.)

### Ramp loss weight

Low signal levels carry little usable signal: when `τ` is near 0 the flow term just
collapses to predicting the dataset mean, while the bootstrap term is comparatively easy
because its targets are deterministic. To spend model capacity where the learning signal
actually is, each term is weighted by a linear ramp in `τ`:

```
w(τ) = 0.9 τ + 0.1
```

Adding this ramp drops FVD from 151 to 102 in the ablation.

### Loss normalization

Because one transformer carries multiple modalities and output heads, all loss terms are
normalized by running root-mean-square (RMS) estimates so no single term dominates.

---

## 4. Inference (how rollouts are generated)

Generation is **autoregressive in time**. Each new frame's representation is produced by
the shortcut model with:

- `K = 4` sampling steps, i.e. step size `d = 1/4`.

That's 4 forward passes per frame — roughly `16×` fewer than the ~64 steps a comparable
diffusion-forcing model needs, while reaching about the same quality. On a single H100
this hits real-time interactive inference (~21 FPS, matching Minecraft's 20 FPS tick
rate) with a 9.6-second context.

One robustness trick: past context inputs are **slightly corrupted** to a small signal
level `τ_ctx = 0.1` before being attended to. This makes the model tolerant of the small
imperfections in its own previous generations, instead of assuming the history is
pristine.

---

## 5. How it slots into the rest of the agent

The dynamics model is pretrained on tokenized video (optionally with actions) using the
shortcut forcing loss above, with the tokenizer frozen. After that:

- **Agent finetuning** inserts task tokens as an extra modality and trains policy +
  reward heads on top of the same transformer. Crucially, agent/task tokens can attend
  *out* to everything but **nothing attends back to them** — this prevents the world
  model's future predictions from being influenced by the task (which would be causal
  confusion); the future should depend only on actions.
- **Imagination training** then freezes the transformer entirely and trains only the
  policy and value heads via RL on rollouts the dynamics model generates of itself.

So at rollout time during RL, this dynamics model is what's being "unrolled with itself":
it samples the next representation from its flow head, the policy head picks an action,
and the loop continues — all inside imagination, with no environment interaction.

---

## Limitations the paper flags

The dynamics model is not a perfect game clone. Its temporal consistency is bounded by
the 9.6-second context, and inventory items in generated frames are sometimes unclear or
drift over time. It handles interface screens (inventory, crafting, furnace) and most
mouse movement well, but precise long-horizon memory and exact inventory state are open
problems.
