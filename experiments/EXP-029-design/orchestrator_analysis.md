# Orchestrator analysis — what P1 means and the memory-method idea space (2026-06-25, overnight)

My own reasoning (not the method-architect's), written so it survives the session. Merlin encouraged
independent thinking; this is the honest state of my thinking on the memory problem.

## 1. The P1 result and its hard implication
EXP-029 (dynamic-secret relay, tbptt-k sweep, continuous 1-D bounce): **for state that must be
INTEGRATED each hop, there is no free extrapolation — a relay trained to depth D works to ~D and then
drifts to chance or worse (full BPTT overshoots to >2× chance at 6× depth).** This is fundamentally
unlike STATIC state (V-T014: full BPTT stays flat forever). Mechanism: a static value sits at an
identity fixed point (the relay just copies); a dynamic value has no fixed point — any per-hop
transition error compounds geometrically, and there is no content anchor during blind occlusion to
correct it. Within the trained depth, you need tbptt-k ≈ depth/2 to learn the transition well.

**Consequence for the method:** to recall hidden POSITION through k occluded steps, the rollout must be
trained to depth ≈ k. There is no architecture-free shortcut. This is why I made the deep-clip variant
(train the relay to h≈44) the scientifically necessary run, not an afterthought — and why h=13 within a
16-window (my first instinct) would have been nearly pointless (it never trains the cross-window hops it
must extrapolate into).

## 2. The crucial caveat — GridWorld is DISCRETE, P1 is CONTINUOUS
P1 uses continuous position+velocity → continuous drift. GridWorld position is a FINITE, BOUNDED,
PERIODIC state: 36 cells, deterministic 8-direction step with wall reflection, **period 10**. A relay
on a discrete state can learn *attractor* dynamics — each hop snapping the memory back toward the
nearest valid cell — which would NOT drift the way a continuous integrator does. So P1 is a
**pessimistic lower bound** on the credit mechanism; whether GridWorld extrapolates better is the open
empirical question the running A/B answers. I genuinely don't know which way it goes, and that is the
point of running it.

BUT note the memory representation is still a CONTINUOUS vector (M=4 × E=256 floats). Discreteness of
the *target* doesn't force the *representation* to be stable — it depends on whether the learned relay
develops snapping dynamics. If it doesn't (GridWorld drifts like P1), the principled fix is to make the
memory itself discrete (next section).

## 3. The idea menu (ranked by my current belief)
1. **FF9 rollout-training (op-3 relay), trained to the eval depth.** RUNNING tonight (EXP-030 h24,
   EXP-031 h44). Direct attack: put the cross-window memory write on the gradient path, train it deep.
   P1 says this is necessary; the GridWorld discreteness may make it sufficient.
2. **Discrete / quantized memory (VQ).** If the continuous relay drifts on GridWorld too, force the
   memory onto a codebook (VQ-VAE-style, straight-through). A finite-state memory cannot drift
   continuously — each hop re-quantizes to a valid state, giving a true finite-state-machine relay.
   This is my top follow-up if rollout-training alone decays. Bigger lift (codebook + commitment loss);
   deferred to a logged decision, not rushed tonight.
3. **Frozen-snapshot of INITIAL CONDITIONS + phase-aware readout.** Reframe: instead of integrating
   position each hop (drift-prone), store a STATIC code (position + direction at occlusion onset) and
   COMPUTE current position from it given elapsed steps. Static codes don't drift (V-T014). Evidence
   this is latent in the data: EXP-028's frozen-snapshot inference gave *period-10 spikes* in position
   recall — i.e. the snapshot DOES hold the trajectory, the readout just isn't phase-aware. Training a
   phase-aware readout (needs an elapsed-step/phase signal fed to the model — an architecture change)
   could fill the spikes into a flat high recall WITHOUT any relay. Elegant; needs a phase input. A
   real alternative to the whole relay apparatus — worth a probe.
4. **Bigger context window (brute force).** The control (EXP-032, window 32). Not a "memory" solution
   (the whole research premise is that you can't grow the window forever — bounded recurrent state is
   the goal), but the relay must beat it at equal-ish recall to justify its complexity. Important
   yardstick.

## 4. Decision tree for the morning
- **If EXP-030/031 relay extends recall past the 16-window AND deeper training (h44) reaches further
  than h24** → P1's horizon=depth holds on GridWorld too, but the relay WORKS (discreteness helps
  enough). Win. Next: push depth / consolidate / 2nd seed.
- **If the relay decays to chance like EXP-028 regardless of training depth** → GridWorld's continuous
  memory drifts like P1's probe. Pivot to discrete memory (VQ, idea 2) or the frozen+phase readout
  (idea 3). The rollout-training negative is itself a clean, publishable result (credit-assignment is
  not enough; representation stability is the binding constraint).
- **If in-window recall (k≤8) or base val-diffusion REGRESSED** → the rollout branch fought ops 1+2;
  raise warmup / lower the rollout weight / lower the hidden fraction, re-run.
- **If vanilla-window-32 (EXP-032) matches or beats the relay** → the machinery doesn't yet earn its
  keep; either push the relay harder or reframe around bounded-state efficiency (compute/memory) rather
  than raw recall.

## 5. Honest uncertainties
- The updating-memory INFERENCE is result-defining and subtle; a faithful mirror of training is
  required or the trained relay won't show. I'm building it carefully and will sanity-check k=1≈1.0 and
  that it matches training op semantics before trusting any A/B.
- M=4 may be too small to hold integrated position with the precision the readout needs (a capacity
  limit orthogonal to credit). P1's BPTT-ceiling arm (M=32 GRU) carried it in-window, so M is probably
  not the in-window bottleneck, but the GridWorld latent is higher-D — widening memory is a cheap knob
  to try if recall is capacity-limited rather than drift-limited.

> SUPERSEDED NOTE (2026-06-25): the "relay" inference framing here is deprecated — there is only the normal sliding-window inference. Corrected verdict: rollout-training does NOT beat FF9-no-rollout under it. See ESC-022.
