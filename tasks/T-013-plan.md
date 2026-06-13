# T-013 — FF9 memory-only sufficiency + distinct MEMORY tokens (design note)

Status: **planning — pre-verifier, pre-decision (D-024 to be written AFTER the verifier audit).**
Chosen by Merlin 2026-06-13 (over option A "sequential register relay"): build the **memory-token split +
FF9 memory-only-sufficiency** line. Measurement = **COLOR-first** (frozen probe 5503e75, T-004 bar, deep
occlusion), position = caveated secondary. This note is the artifact the `critical-claim-verifier` audits.

Code grounded in `src/D_dynamics_model/dynamics_model.py` (read 2026-06-13): token assembly :336-362,
`loss` :365-433, `_ff7_loss` :435-480, token layout `[action(n_action_tokens) | latents(L) |
registers(n_registers) | shortcut(1)]`, temporal attention position-wise per token slot (each slot = its
own causal channel through time), spatial attention = full within-frame mixing.

---

## 0. VERIFIER VERDICT (V-T013, 2026-06-13) — the spec below is REVISED to fold it in

`critical-claim-verifier` tested "FF9 forces the memory tokens to encode the full hidden state; no
trivial shortcut." **Verdict: REFUTED as originally specified.** Two findings, both load-bearing:

1. **Loss shortcut (fixable).** FF9 (copying `_ff7_loss`) gives successor frames t+1..t+k their OWN real
   latents noised at τ ~ Uniform(0,K_max), loss ramp-weighted `w=0.9τ+0.1`. The dominant, ramp-favored
   (high-τ) part is solvable by locally denoising each successor's own latent — memory non-load-bearing
   there. Memory only bites in the low-τ tail the ramp down-weights. Empirical probe
   (`experiments/verify-T013/probe_memory_loadbearing.py`): max memory benefit ~61% of self-denoise loss
   but concentrated at τ≈0.1 (+0.177) and ≈0 at τ≈0.9 (+0.002). → withholding the latent does NOT *force*
   full-state memory under the inherited ramp; it only permits it. **FIX (folded into §3 below): on FF9
   successor frames clamp τ low / flatten-or-invert the FF9 ramp; strongest — supervise frame 1 at τ≈0
   (pure noise) so the entire target must come from `mem_t` through the causal channel.**
2. **Single-hop credit limit (NOT fixable within v1 — strategic).** Even with the τ fix, FF9 v1's TBPTT-1
   gradient trains read + 1-hop write but NOT preserve-across-N-hops. Predicted outcome: reproduces FF7's
   split — static COLOR survives (a constant accumulates from even weak low-τ pressure) but dynamic
   POSITION and beyond-window retention *depth* do not improve over FF7. Depth/position need the
   sequential register-relay (option A, parked). → **FF9 v1 alone is a DIAGNOSTIC, not an expected win on
   depth; the objective (B) and the credit-assignment (A) fixes are COMPLEMENTARY, not alternatives.**
   This is escalated (ESC-013) because it bears on the A-vs-B choice.

Confirmed correct by the verifier (no change): Q#5 no main-loss corruption (fresh gather tensors; only
`regs`/`mem` carry intended grad); the `absent_latent` placeholder is a fair per-instance constant; Q#4
env is deterministic given the action sequence (`occluded_bouncing.py`), so the point-prediction loss is
well-posed. Full report folded; EXPERIMENTS.md row `V-T013`.

---

## 1. Why (the scientific motivation, from IDEAS.md + EXP-010/013)

FF7 (registers as carrier, single-timestep-sufficiency) carries static **color** beyond the window
(EXP-010) but not dynamic **position** (EXP-013). Two recorded problems FF9 is meant to fix:

1. **Role conflation.** FF7 makes the `n_registers` scratch tokens be BOTH the model's free scratchpad AND
   the memory carrier. FF9 splits them: a **distinct MEMORY token type** is the persistent carrier;
   REGISTER tokens revert to pure scratch (no memory duty).
2. **Off-screen-only memory churns.** FF7's loss overwrites frame-t's latent with the *real* (possibly
   occluded) latent, so memory only has to store what the latent does NOT already supply — i.e. *off-screen*
   info. When visibility switches (something moves on/off screen) the memory must reshuffle. FF9 **withholds
   the current latent entirely**, forcing memory to be a **complete, everything-included state object** (on-
   AND off-screen) whose contents don't churn as visibility changes.

Memory-only also gives **cleaner credit**: a few-frame reconstruction from a fixed memory object is a more
stable target than training memory against other memory tokens (a self-referential moving target).

---

## 2. What changes (architecture — minimal, mirrors the register machinery)

Add a distinct memory-token type alongside registers. New token layout per frame:
```
[ action(n_action_tokens) | latents(L) | MEMORY(n_memory) | registers(n_registers) | shortcut(1) ]
```
- New config: `n_memory: int = 4` (slots), `use_full_state_memory: bool = False` (dispatch flag, parallel
  to `use_register_memory`).
- New param: `self.memory_tokens = nn.Parameter(...)` learned init (like `register_tokens` :255).
- `forward(...)` gains `memory_in: Optional[(B,T,n_memory,E)]` and `return_memory: bool`, exactly parallel
  to `register_in`/`return_registers` (:309-360). When `memory_in is None`, use the learned init.
- Registers keep their existing path UNCHANGED but are **no longer injected** for memory in the FF9 line
  (they stay learned-init scratch). `register_in` stays for back-compat / the FF7 line.
- Memory tokens are position-wise temporal channels (free, same as registers) → carry across time with **no
  further architecture change**, same as the FF7 rationale. Spatial layers route latent→memory (write) and
  memory→latent (read) within a frame.

This is additive: `n_memory=0` ⇒ byte-identical to today. The vanilla and FF7 paths are untouched.

## 3. The FF9 loss (memory-only sufficiency) — exact, by analogy to `_ff7_loss`

Same combined-step structure as FF7 (main windowed diffusion loss unchanged; FF9 is an added term):
- Main pass `loss()` runs as today but with `return_memory=True` → yields `mem` (B,T,n_memory,E): the
  memory each frame wrote from its window (where the write is trained).
- `_ff9_loss(z1, mem, actions, k)`: for every frame t with k successors, a (k+1)-frame mini-forward folded
  into batch (identical windowing to `_ff7_loss` :456-458):
  - **frame 0 (t):** latent slots = a learned **`absent_latent` placeholder** (latent WITHHELD — the key
    difference from FF7, which puts the real latent there); `memory_in[:,0] = mem_t` (injected, the only
    carrier); registers = learned scratch; shortcut at `tau_ctx`.
  **→ SUPERSEDED by FF9 v2 (Merlin, 2026-06-13) — variable-horizon, pure-noise path. Use this:**
  - Per rollout sample a horizon **j ∈ {1..k}** (uniform). Build a (j+1)-frame mini-forward `[t..t+j]`.
  - **Hard constraint: no GT latent on the path.** Frames t..t+j−1 (incl. the memory-source frame t) are at
    **signal level τ=0** (pure noise) — so the latent slots carry NO ground-truth; memory_t (injected at
    frame t) + the learned-init memory tokens at t+1..t+j−1 (relayed via the temporal memory channel) are
    the ONLY scene info. (τ=0 replaces the `absent_latent` placeholder of §2 — withhold via signal level.)
  - **frame t+j (terminal only): τ sampled freely** ("any signal level" — so a well-posed denoising target
    exists and the memory-conditioned denoiser is calibrated at the τ levels generation visits). finest d.
  - **Loss on ALL of t+1..t+j** (Merlin refinement 2026-06-13 — not just the terminal frame): flow loss
    `(z_hat[:,1:j+1] − z1[:,t+1:t+j+1])²`, **un-ramped** for the FF9 term (config knob; drop FF7's
    `0.9τ+0.1` so memory-bearing low-τ samples aren't down-weighted — V-T013). **This stays leak-free:**
    frames t+1..t+j−1 are at τ=0, so each is a PURE memory-sufficiency target at its horizon (no own-latent
    GT to cheat from, and their τ=0 predecessors carry none either); the only signal-bearing frame is the
    terminal t+j, which has no successors → leaks to nothing. So we get j supervised memory targets per
    rollout (horizons 1..j) for free, instead of one.
  - Random j trains memory sufficiency over variable within-window horizons (1..k). NB (recorded): within
    one forward, frame t+j attends DIRECTLY to frame t's memory tokens — so this trains "memory = sufficient
    attendable full-state object," NOT the cross-window relay (preserve after the source leaves the window).
    The relay is option A, layered on next. This is why FF9 v2 is an architectural BASELINE, not a depth fix.
  - Config knobs: `ff9_k` (max lookahead), `ff9_ramp` (on/off, default off), `ff9_tau_last` sampling.
  - **50/50 GT split (Merlin 2026-06-14, → T-014 §2):** per rollout choose p=0.5 strict-no-GT (path τ=0,
    above) vs normal noised-GT path (memory composes with present-but-noisy context — matches the rollout
    distribution where t+1 is already decoded when t+2 is predicted). Knob `ff9_gt_frac`. To implement
    alongside the relay mode (T-014); not in the current built `_ff9_loss` yet.
- Backprop path: through the injected `mem_t` into the windowed pass that wrote it → trains the **write**
  side; k≥2 trains the in-pass memory→memory **relay** (one hop of gradient, TBPTT-1-like). Deep
  preserve-across-N-hops is the *sequential relay* extension (parked; option A) — NOT in FF9 v1.
- `total = diffusion + lambda_ff9 * ff9` (parallel to :430).

Difference from FF7 in one line: **withhold the latent (learned placeholder) instead of overwriting it with
the real latent, and carry distinct MEMORY tokens instead of registers.** Everything else reuses the proven
FF7 plumbing.

## 4. Inference

A `generate_full_state_memory` analog of `generate_memory`: carry each frame's final-layer MEMORY state
forward and inject as the next step's memory (reuse `memory_rollout_init/step` machinery, retargeted from
the register slot to the memory slot). Latents still flow normally between steps; memory is the persistent
channel. `generate()` dispatches here when `use_full_state_memory`. (KV-cache: the streaming cache T-012
applies unchanged — memory tokens are just more per-frame tokens in the window.)

## 5. Success criteria — FF9 v2 is an ARCHITECTURAL BASELINE (Merlin, 2026-06-13)

Reframed: the goal of this experiment is a sound, non-leaking memory-token model to build the relay (A) on
— NOT to beat FF7 on beyond-window depth (Merlin: "this alone will not fix FF7"). So the primary signals are
about the mechanism being healthy, with the frozen probe as a sanity/positioning check.

- **Primary (does the mechanism work?):** within-window **memory sufficiency** — with the FF9 v2 setup
  (path frames at τ=0, memory injected), memory-only prediction of frame t+j must beat the memory-free
  baseline (predict-the-prior) by a clear margin, across j ∈ {1..k}. This is the direct readout that memory
  encodes the full state (reuse the V-T013 probe machinery: L(memory) ≪ L(no-memory)).
- **No-regression tripwire:** base 1-step teacher-forced dynamics + ceiling/drift controls equal-or-better
  than vanilla_s0 (FF9 must not degrade base diffusion — as FF7 didn't, EXP-012). The main diffusion loss
  forward is untouched, so expect parity.
- **Positioning on the frozen probe (sanity, trusted metric):** color ΔRGB at n_occ {12,16,24,32,48} vs
  vanilla_s0 (chance beyond window) and FF7 (decays ~65 by 24). **Expectation: ≈ FF7** (within-window
  training only → OOD beyond window, like FF7). A *flatter-than-FF7* color curve would be a pleasant bonus
  (full-state object generalizes better); ≈FF7 is the expected, acceptable baseline result; *worse* than
  FF7 ⇒ the τ=0 path hurt and needs investigation.
- **Secondary (caveated):** position via the EXP-013 metric — reported, NOT a gate (uncertain strength,
  ESC-009). Real depth/position is the relay's (A) job, layered on this baseline next.
- Screen single-seed first (Merlin relaxed the 2-seed order); replicate the better config on promise.

## 6. Open questions / forks for the verifier to pressure-test

1. **Trivial-collapse / shortcut risk.** Can the model pass `_ff9_loss` WITHOUT storing hidden state — e.g.
   memory encodes nothing and the k-frame prediction rides the *real noised latents of frames 1..k* (which
   ARE provided)? In FF7 the same risk exists; the loss is on frames 1..k which get real latents. Is
   memory_t actually load-bearing for predicting t+1, or can the denoiser ignore mem_t and denoise frames
   1..k from their own noised latents alone? **This is the crux** — if frames 1..k carry enough signal in
   their own noised latents, FF9 (like FF7) may not pressure memory hard. Mitigation candidates: noise
   frames 1..k near-fully (low tau) so memory must supply the content; or supervise frame 1 only at very
   low signal. Verifier: is the objective actually identifying the memory, or is it satisfiable by the
   local denoiser?
2. **Absent-latent placeholder.** Learned token vs zeros vs the MAE mask token reuse — does a learned
   placeholder leak a shortcut (constant) the model games?
3. **Write-side credit with TBPTT-1.** Is one hop of gradient enough to teach memory to STORE (not just be
   read), or does FF9 inherit FF7's "trains read + 1-hop write, never preserve-across-N-hops" limit — i.e.
   is FF9 v1 expected to improve color *depth* at all over FF7, or only after the sequential-relay
   extension? (If the honest answer is "needs the relay," that reorders the plan.)
4. **Stochasticity.** Our billiard env is deterministic given the action sequence (incl. curtain) → the
   "distributional target" caveat (IDEAS.md) is likely moot for v1. Confirm no hidden stochastic source
   (e.g. spawn randomness inside the clip) breaks the point-prediction loss.
5. **Does withholding the latent at frame 0 starve the main diffusion objective?** (No — main `loss()`
   forward keeps latents; only the FF9 aux forward withholds. Verify no shared-tensor coupling.)

## 7. Build steps (D-024 — FF9 v2)

1. Add `n_memory`/`memory_tokens`/`memory_in`/`return_memory` + `use_full_state_memory` to config+forward
   (additive; `n_memory=0` ⇒ byte-identical to today — guard with a smoke test). No `absent_latent` token
   needed — FF9 v2 withholds via signal level τ=0.
2. `_ff9_loss` (FF9 v2: random j∈{1..k}, path frames τ=0, last frame τ sampled, loss on last frame only,
   un-ramped) + wire into `loss(ff9_k=..., lambda_ff9=...)`; `train_dynamics_model.py --ff9` flag + knobs.
3. `generate_full_state_memory` + dispatch; unit smokes (shapes, `n_memory=0` identity, τ=0-path builds,
   1-epoch finite, probe dry-run + memory-sufficiency probe through the new generate).
4. Train on occluded env at the EXP-010/012 budget (100 ep, bs32, lr3e-4, seed0); committed config.yaml +
   run.sh. Eval: memory-sufficiency (primary) + frozen-probe color n_occ {12,16,24,32,48} vs vanilla_s0 +
   FF7 + no-regression check. Present-then-stop.

NB §6 below was the pre-build verifier checklist (V-T013) — now folded; finding (1) fixed by FF9 v2,
finding (2) accepted (baseline framing), Q4/Q5 confirmed sound.

Provenance discipline: committed `config.yaml` + `run.sh` per EXP (the EXP-010 gap, fixed since EXP-012).
Run on the 4070 via `venv/Scripts/python.exe`.
