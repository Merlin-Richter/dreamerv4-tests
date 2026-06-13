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
  - **frames 1..k (t+1..t+k):** real latents noised at sampled tau, finest d; `memory_in[:,1:]` = learned
    init (the in-pass memory relay forwards mem_t through the k hops, exactly as FF7's k≥2 relays the
    register).
  - **target/loss:** flow loss on the latent outputs of frames 1..k vs real z1 (ramp-weighted), same as
    `_ff7_loss` :478-480. Frame 0's latent output is ignored (it was a placeholder).
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

## 5. Success criteria (COLOR-first; pre-registered, Merlin 2026-06-13)

- **Headline (trusted):** frozen probe 5503e75, hidden-COLOR ΔRGB vs the T-004 bar (< ~63) at **deep**
  occlusion **n_occ ∈ {24, 32, 48}** — extended past EXP-010's {12,16,24} because FF7 only *just* misses at
  24; the question is whether full-state memory holds color where FF7's drift breaks down. Baselines on the
  identical probe: vanilla_s0 (EXP-012, at chance beyond window) and FF7 k=3 (EXP-010, decays to ~65 by 24).
  **FF9 "works" (v1):** color ΔRGB stays < bar at n_occ where FF7 has crossed it (≥24), i.e. flatter deep
  decay than FF7.
- **Secondary (caveated):** position via the EXP-013 metric — reported, NOT a gate (metric of uncertain
  strength, ESC-009). If FF9's full-state objective moves position at all, that's the interesting bonus.
- **No-regression tripwire:** base 1-step teacher-forced dynamics + ceiling/drift controls must be
  equal-or-better than vanilla_s0 (FF9 must not degrade base diffusion — as FF7 didn't, EXP-012).
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

## 7. Build steps (after verifier + D-024)

1. Add `n_memory`/`memory_tokens`/`memory_in`/`return_memory`/`absent_latent` + `use_full_state_memory`
   (additive; `n_memory=0` ⇒ identical to today — guard with a smoke test).
2. `_ff9_loss` + wire into `loss(ff9_k=..., lambda_ff9=...)`; `train_dynamics_model.py --ff9` flag.
3. `generate_full_state_memory` + dispatch; unit smokes (shapes, n_memory=0 identity, 1-epoch finite,
   probe dry-run through the new generate).
4. Train on occluded env at the EXP-010/012 budget (100 ep, bs32, lr3e-4, seed0); frozen-probe eval at
   n_occ {12,16,24,32,48} vs vanilla_s0 + FF7. Present-then-stop.

Provenance discipline: committed `config.yaml` + `run.sh` per EXP (the EXP-010 gap, fixed since EXP-012).
Run on the 4070 via `venv/Scripts/python.exe`.
