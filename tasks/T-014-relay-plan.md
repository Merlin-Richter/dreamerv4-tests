# T-014 — FF9 relay training (operation 3: write-memory-from-memory) + FF9 v2 50/50 + mode alternation

Status: **design note for critical-claim-verifier — pre-decision (D-025 after the verdict).**
Author: orchestrator, from Merlin's design dialogue 2026-06-14. Builds on T-013 (FF9 v2 architecture +
loss, BUILT, smokes green) and D-024. Code grounded in `src/D_dynamics_model/dynamics_model.py`:
`forward` (memory tokens are a layer-0 input → final-layer activation, read via `return_memory`),
`_ff9_loss` (FF9 v2), `_ff7_loss` (the FF7 predecessor), `generate_streaming` / `stream_rollout_init/step`
(the cross-frame sliding-window eviction KV cache, T-012), `memory_rollout_init/step` (FF7 register relay).

## 0. Why op 3 is missing and what it is
A complete memory training has three operations; FF9 v2 has only 1 & 2:
1. **write memory ← latents** ✅ (main windowed pass writes memory_t from real latents).
2. **read memory → latents** ✅ (FF9 v2 reads injected memory to predict the next 1..j frames).
3. **write memory ← memory** ❌ — memory_t written from the PREVIOUS, *relayed* memory. FF9 v2 never
   trains it: its context frames carry UNIFORM learned-init memory (no information to read). Op 3 = the
   relay write-side = parked option A = V-T013 finding (2) "preserve-across-N-hops". Including it = A+B
   combined, which V-T013 predicted is needed for beyond-window depth.

**Memory is an ACTIVATION** (final-layer hidden state of the memory-token positions; no signal level, no
noise, not x-predicted, no GT target). So it is produced by ONE forward per frame (teacher-forced latents,
no K substeps) and is cacheable — which is exactly what makes Mode B below cheap.

## 1. Two training modes — alternate ("take turns")
### Mode A — parallel windowed (FF9 v2, ops 1 & 2). Unchanged from T-013 except the 50/50 split (§2).
Full-clip parallel; cheap. Trains base diffusion + write-mem-from-latents + read-mem-into-latents.

### Mode B — sequential relay (op 3). NEW.
Sliding window N; carry memory tokens across steps **DETACHED**. Per step t:
1. **One grad-carrying forward** for the newest frame: it reads the **cached, detached** context (frames
   t−N+1..t−1: their K/V, incl. their relayed memory tokens) + frame t's (teacher-forced, noised) latents,
   and **produces memory_t** (with grad).
2. **FF9 loss on frame t** (the §2 loss, anchored at t): memory_t must be sufficient to predict the next
   1..j frames. Backward — gradient reaches ONLY this one forward (the write of memory_t + its reads of
   the detached context + the FF9 read head). **No out-of-window / cross-step BPTT.**
3. **Detach** memory_t, append its K/V to the cache, **evict** the oldest column, slide to t+1.
Run ~200 steps so the context fills with genuinely **deep-relayed** memory. **Batch heavily** across
episodes (lockstep in time → one big-batch grad step per time index). **Prefer small/variable N** (faster +
forces memory use — less explicit context to lean on; ragged N → bucket episodes by N). The T-012 streaming
eviction cache provides the detached context K/V so each step computes only the new frame.

## 2. FF9 loss — 50/50 GT split (Merlin refinement 2026-06-14)
Per FF9 rollout, choose with p=0.5:
- **strict no-GT** (current v2): path frames t..t+j−1 at signal τ=0 (no GT latent on the path; memory is
  the sole carrier), terminal frame t+j at sampled τ; loss on frames 1..j; un-ramped.
- **noised-GT**: path frames get random-noised GT latents like the main diffusion (memory composes with
  present-but-noisy context). Rationale: at real rollout, when predicting t+2 the frame t+1 is usually
  already decoded → context carries generated latent info. Matches the inference distribution.
Both backprop usefully into memory construction; the strict half guarantees memory CAN be the sole carrier
(V-T013 forcing), the noised half matches reality.

## 3. Why detached carry still trains operation 3 (the load-bearing claim)
Each token's memory is written WITH grad, conditioned on REAL (detached) prior memory, supervised by its
OWN FF9 sufficiency. The SAME weights write every token, so "read-a-sufficient-memory + frame →
write-a-sufficient-memory" is a learnable fixed point (FF7/Bellman single-step-sufficiency). Per-hop
sufficiency propagates the relay WITHOUT long-range credit; the detach only drops credit for HOW prior
memories were made (trained when they were the newest token). Train/test consistent: the reader reads the
actual produced (detached) values, the same it sees at inference.

## 4. What the verifier should pressure-test (falsifiable claims)
**Central claim:** "Mode B (detached-carry, per-step-FF9, sliding-window relay) trains operation 3 such
that memory relays *sufficient* hidden state across many hops — the detached carry does not break the
credit needed to learn a USEFUL write, and the relay is stable over ~200 hops."
Specific risks:
1. **Bootstrap trap / fixed-point soundness.** With detached carry, memory_t is trained only for t's own
   FF9 prediction, never for t+1's. Does the single-step-sufficiency fixed-point actually exist and get
   reached, or can the model satisfy each step's FF9 loss while the relayed memory slowly loses the hidden
   state (drift/collapse) because nothing trains "write a memory the NEXT reader needs"? Is the Bellman
   analogy valid when the carrier is a continuous activation with no explicit target?
2. **Activation-relay stability.** Final-layer activation → layer-0 input over ~200 hops: drift / explosion
   / collapse to a fixed point? Is a norm / clamp / small projection needed on the relayed memory?
3. **Does the 50/50 GT split reopen the V-T013 shortcut?** In the noised-GT half memory can be
   non-load-bearing (latents carry the info). Is 50% strict enough to keep memory forced, or does the
   noised half dominate and let memory atrophy?
4. **Mode A/B interference.** Mode A writes memory from a learned-init context; Mode B from a relayed
   context. Two different input distributions for the memory writer — do they conflict (alternation
   thrash), and should B dominate / be curriculumed?
5. **Cache correctness under grad.** Using the streaming eviction cache with detached context K/V while the
   new frame carries grad — are the cache's K/V semantics correct (the cache was validated for no_grad
   inference, T-012)? Any aliasing / stale-rotation bug when reused in training?
6. **Stochasticity / target staleness.** The relayed memory is a moving target as weights update
   (target-network-like). Stable, or does it need a slow/EMA copy?

## 5. Build outline (after verifier + D-025) — NOT yet started
1. FF9 50/50 split in `_ff9_loss` (config knob `ff9_gt_frac`, default 0.5).
2. Mode B trainer: a sequential relay loop reusing `stream_rollout_init/step` for the detached cached
   context + a per-step grad forward that produces memory_t and calls the FF9 loss; detach + evict + slide;
   batched across episodes; small/variable N. New `train_dynamics_model.py` mode flag.
3. Mode alternation (take turns A/B; ratio a knob).
4. Smokes: gradient reaches only the current-step forward; detached carry verified (no graph across steps);
   memory relays a planted signal across hops on a toy clip; cache-vs-recompute parity under grad.
5. Train (occluded env) → EXP-018, present-then-stop. Measure: memory-sufficiency + frozen-probe color at
   DEEP occlusion (n_occ 24/32/48) vs FF9-v2-baseline + FF7 + vanilla_s0 (this is where depth should finally
   move if op 3 works), position reported (caveated).
