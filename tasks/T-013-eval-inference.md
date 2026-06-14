# T-013 eval — `generate_full_state_memory` inference design (for independent audit)

Status: design fork, pre-implementation. This note is the artifact for an independent verifier.
Do NOT assume the author's lean is correct; the question below is falsifiable from the trained model
+ the training code.

## Context (what was trained)

A dynamics model (`src/D_dynamics_model/dynamics_model.py`) was trained on an occluded bouncing-ball
env with the **FF9 v2** objective (`_ff9_loss`, lines ~527-585). Architecture: per-frame token layout
`[action | latents(L=4) | registers(4) | MEMORY(4) | shortcut]`. Temporal attention is **position-wise
per token slot** (each slot is its own causal channel through time); spatial attention mixes all slots
within a frame. The trained checkpoint is `experiments/EXP-017/ff9v2_s0.pt`.

**The FF9 v2 loss exactly (read `_ff9_loss`):**
- The main windowed forward (`loss()`, `return_memory=True`) runs the full clip with **all real latents**
  and yields `mem` (B,T,n_memory,E): each frame's memory state, written from its causal window of real
  latents. This is the only place memory is *written from observations*.
- The FF9 aux forward: for each source frame t, a (k+1)-frame mini-window `[t..t+k]` (k=`ff9_k`=3):
  - sample horizon `j ∈ U{1..k}`.
  - **frames t..t+j-1 (incl. source t): signal τ=0 ⇒ their latent slots are PURE NOISE** (no GT latent).
  - `memory_in[:,0] = mem_t` (the real written memory, injected at the source frame); memory at
    t+1..t+j-1 = learned-init tokens.
  - **frame t+j (terminal): τ sampled freely** (a denoising target).
  - flow loss on frames t+1..t+j (un-ramped). Backprop through injected `mem_t` (TBPTT-1, write-side).
- So within ONE forward, frame t+j attends DIRECTLY (temporal memory channel) to frame t's injected
  memory. The model is trained to: (W) **write** a memory from a window of real latents; (R) **read** an
  injected memory at a τ=0 source frame to predict the next 1..j frames whose own latents are noise.
- **NOT trained:** memory→memory carry across hops (op-3 / "the relay"). The verifier on the design
  (V-T013) already established FF9 v2 trains read + 1-hop write but NOT preserve-across-N-hops.

## What the eval must do

The frozen revisit-probe (`src/probe/revisit_probe.py`, frozen @ 5503e75) calls `dyn.generate(context,
n_generate, action_idx)` and reads ball color from the decoded predicted frame at the reveal index. For
FF7 checkpoints, `generate()` auto-dispatches to `generate_memory` (register-carry relay). For FF9 we
add a dispatch to `generate_full_state_memory`. The probe occludes the ball for n_occ frames AFTER the
color-carrying prefix scrolls out of the N=8 latent window, so beyond n_occ≈7 the ONLY way to recall
color is a carrier that survives past the window. We compare FF9 v2 against vanilla_s0 (cliff to chance)
and ff7_k3 (register relay carries color, decays ~65 ΔRGB by n_occ=24).

## The fork — two independent binary choices

**Choice A — source-frame latent signal level during the memory read:**
- (A1) **τ=0 / pure-noise source**, matching FF9 v2 training exactly: the predicted frame reads the scene
  ONLY from injected memory (latent withheld). In-distribution for the read op.
- (A2) **τ=tau_ctx / near-clean source**, matching `generate_memory`/`_denoise_next` and plan T-013 §4
  ("latents flow normally; memory is the persistent channel"). The model gets BOTH the recent latent AND
  the memory — but it never saw a near-clean source latent paired with an injected memory (OOD pairing),
  and it makes the comparison structurally identical to FF7's relay.

**Choice B — does the carried memory get updated each rollout step?**
- (B1) **Static carry**: write `mem_carry` ONCE from the observed context window (the trained W op), then
  inject it UNCHANGED at every subsequent step's source frame. Uses only trained ops (W once, R each
  step). Cannot integrate new dynamic state, but exactly right for a static attribute (color).
- (B2) **Dynamic re-extract (op-3 relay)**: each step, re-extract the new frame's memory (forward with the
  carried memory injected, `return_memory`) and carry that forward — the structural analog of
  `generate_memory`. This invokes the memory→memory write, which FF9 v2 **never trained** (OOD), so it may
  drift/degrade (cf. V-T014: untrained detached carry drifts to chance past the trained depth).

## The claim to test (falsifiable)

> **Claim C:** For a faithful, fair, and interpretable measurement of FF9 v2's *beyond-window* memory on
> the frozen color probe — comparable to FF7's `generate_memory` — the correct `generate_full_state_memory`
> is **A1 + B1** (τ=0 source, static memory written once from the observed prefix and injected unchanged):
> it uses only operations FF9 v2 was actually trained on (write-from-window, read-at-τ=0), so a color-recall
> result reflects the *trained objective* rather than an OOD inference trick; whereas A2 and/or B2 inject
> OOD conditions (near-clean source paired with injected memory; untrained memory→memory carry) that could
> either flatter or corrupt the result, confounding the FF9-vs-FF7 comparison.

Counter-position to weigh: plan T-013 §4 specifies A2+B2 (the generate_memory analog) so FF9 and FF7 run
the *identical* inference shape; under that view A1+B1 changes two things at once vs FF7 (carrier type AND
inference shape), so a FF9≠FF7 result couldn't be attributed cleanly either.

## What I'm asking the verifier to determine

1. Which of {A1,A2}×{B1,B2} most faithfully measures *what FF9 v2 was trained to do* with hidden memory,
   such that a beyond-window color-recall number is attributable to the FF9 objective and not to an OOD
   inference artifact? Reason from `_ff9_loss` and the forward, and probe the trained checkpoint if useful
   (e.g. does injecting a τ=0 source + real memory actually predict t+1 well? does a near-clean source make
   memory inert? does B2 drift?).
2. Is there a fairness problem in comparing FF9(chosen) vs FF7(generate_memory) — and if so, what is the
   right framing (e.g. report both A1+B1 and A2+B2; or hold inference shape fixed and accept the OOD)?
3. Any correctness traps in either implementation (e.g. memory injected at the wrong frame so the predicted
   frame can't attend to it through the position-wise channel; sliding-window vs the 2-frame training shape;
   action alignment).

Deliver a verdict (SUPPORTED / REFUTED / UNDETERMINED for Claim C) + the recommended concrete inference
spec to implement, with the cheapest probe(s) that decide it. Artifacts under `experiments/verify-T013-eval/`.
