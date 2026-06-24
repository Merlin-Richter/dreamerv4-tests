# FF9 — the full idea (memory tokens for world-model memory)

> Concise standalone description of FF9 and the rollout-training extension. Companion to the registry
> in `IDEAS.md` ("three operations"); this is the canonical short statement of the method.

## What FF9 is (the corrected understanding)
The dynamics transformer carries, per frame, a set of **memory tokens** (a distinct token type, M=4),
alongside the frame's latent/register/shortcut tokens. Because temporal attention is position-wise,
each memory slot is its own causal channel through time: frame *t*'s memory tokens can attend to the
memory tokens of frames *t−1, t−2, …* in the window. The memory tokens are an **activation** (a final-
layer hidden state fed into the next frame), not a denoising variable — they have no ground-truth label.

**Inference is the ordinary autoregressive rollout** (`generate_cached`, plain): each step produces new
memory tokens and carries the previous frames' memory tokens in the sliding window via temporal
attention — exactly like the frame latents. (NOT the frozen-snapshot `generate_full_state_memory`; that
special case writes one snapshot and never updates it — it has no dead-reckoning and is wrong as the
default FF9 inference.)

## The training, today (the "sufficiency" loss, `_ff9_loss`)
For each frame *t*, a real memory token (written from the prefix window in the main pass) is injected at
frame *t*; the path frames are set to **τ=0 (pure noise → no latent carries the scene)**; the model must
reconstruct the next 1..j frames from **memory alone**, under the realized actions. This **forces the
memory tokens to contain the full hidden state** (on- and off-screen) — operations **(1) write memory←
latents** and **(2) read memory→latents**. EXP-028 confirms it works: FF9 tracks position + colour well
past the no-memory window, decaying to chance by ~k≈28.

## What is missing — operation (3): write memory ← memory
The sufficiency loss never puts **real, previously-written memory tokens in the context** and asks the
model to **read them and write the next memory token**. So **memory→memory propagation across context
windows is untrained** — the model relies on a relay it was never given gradient for. This is exactly
why recall decays with horizon (V-T014: a carry trained only within-window / without through-time
gradient does not extrapolate; only BPTT-through-the-carry does).

## The extension — rollout training (this proposal)
Add a second training mode, **~25% of steps** (75% stays the current sufficiency loss), that does real
rollouts so genuine relayed memory tokens are present in context to learn to write the next ones from:
- **Random sliding-context window length** per step, sampled from {2, 4, 8, …, 2^i, …, N}. Shorter
  windows hit the memory→memory transition sooner (the informative latent has already evicted), giving
  more memory-relay gradient per unit compute.
- **Losses unchanged in kind:** flow-matching on latents + the k-step memory sufficiency loss under
  random actions — but now evaluated on a window filled with REAL relayed memory tokens.
- **KV caching + gradient caching is mandatory** for this to be affordable: cache the context K/V across
  rollout steps (as inference already does, with eviction), and **retain the autograd graph for the
  memory-token K/V** so the loss flows back through the chain of memory writes (truncated BPTT through
  the memory channel). Detach the committed frame-latent K/V (held near-clean at `context_signal`);
  keep grad only on the memory tokens.

Net effect: the memory tokens are trained not just to *contain* the state but to *preserve and re-write*
it across many hops — the relay FF9 currently fakes at inference becomes a trained operation.

## Open questions (to settle before / during the build)
1. **Flow loss on the newest frame only?** Taking the latent flow loss on all frames in every window
   re-trains each frame ~(window-size) times — likely a harmful repeated/over-weighted signal, and the
   older in-window frames are the easy ones. Lean: flow-match the **newest** frame only (also matches
   inference, which only ever commits the newest frame). The memory sufficiency loss stays multi-frame.
2. **Hide all frame latents during the rollout flow loss?** Setting context latents to τ=0 so memory is
   the ONLY carrier maximises memory-specific gradient — but it diverges from inference (where latents
   ARE present near-clean), risking a train/test mismatch / mis-calibrated latent+memory combination.
   Lean: a **mixture** (some steps memory-only, some realistic), not memory-only as the sole mode.
3. **How far back should gradients flow?** Ideally the loss at *t* that reads a memory token written at
   *t−3* should reward that token's *construction*, i.e. BPTT through the memory chain — deeper than the
   attention window. Cannot be infinite (memory); keep the memory-token graph alive for **~4·N** steps
   and detach beyond. This truncation depth is the single most important knob (it is the ESC-014 "min
   BPTT depth that extrapolates" question; V-T014 showed tbptt-1 insufficient, full BPTT works).

## Provenance
Corrected FF9 understanding + this rollout-training extension: Merlin, 2026-06-24 (EXP-028 turn).
Realizes operation (3) from `IDEAS.md` "three operations" (2026-06-14). Relates to ESC-014 / V-T014
(relay gradient design) and the `stream_rollout_init/step` eviction-cache infra (T-012).
