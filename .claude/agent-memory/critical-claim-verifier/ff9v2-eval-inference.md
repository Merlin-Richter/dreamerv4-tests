---
name: ff9v2-eval-inference
description: V-T013-eval SUPPORTED — FF9 v2 beyond-window color eval must use A1+B1 (τ=0 source + static memory written once); B2 re-extract relay drifts (untrained op-3). Reusable load-bearing/drift probes.
metadata:
  type: project
---

**Finding (V-T013-eval, 2026-06-14): Claim C SUPPORTED — the faithful `generate_full_state_memory`
for FF9 v2 (ckpt `experiments/EXP-017/ff9v2_s0.pt`) is A1+B1**: source frame at τ=0 (pure-noise
latent) + memory written ONCE from the observed prefix and injected UNCHANGED each step. Both A2
(near-clean source) and B2 (re-extract relay) are wrong/OOD.

**Deciding numbers (seed 0, 4070, n_ep 32–48, color dRGB; no-mem floor ~93–107):**
- Read-op load-bearing (predict next-frame from injected mem): VISIBLE source → A1 no/with 0.90/0.19
  (4.8×, load-bearing), A2 0.24/0.18 (1.4×, **near-inert** — it just reads the latent). But OCCLUDED
  source (the real probe regime, curtain latent has NO color): BOTH load-bearing — A1 11.8 vs no-mem
  101.8 (+90), A2 11.2 vs 93.9 (+83). So A2's inertness is an artifact of visible sources ONLY; in
  regime A2 is not inert.
- Rollout vs n_occ: **B1 (static) FLAT ~12–14 dRGB out to n_occ=24** (A1 12.4/13.7/12.9 @ 2/8/24).
  **B2 (re-extract) DRIFTS monotonically 17→49** (A1) / 19→46 (A2) — the untrained memory→memory
  relay, exactly the [[detached-carry-relay-drift]] (V-T014) failure mode.
- Position: A1≈A2 byte-for-byte (6.4/6.4 … 28.7/28.6). Near-clean source buys NOTHING dynamic. Memory
  carries static COLOR well, NOT precise position (pos degrades with n_occ under all designs) —
  consistent with EXP-013 (FF7 relay carries color not position).

**Why A1 over A2 (the rationale is NOT "A2 inert"):** in-regime A2 is not inert, but A2 pairs a
near-clean source latent with an injected memory — a pairing FF9 v2 NEVER trained (`_ff9_loss`
always sets the source/path frames to τ=0). A2 gives zero benefit on color OR position, so it is pure
OOD risk with no upside. A1 is fully in-distribution and equal.

**Recommended concrete spec to implement:** per step a 2-frame window `[source | new]`; `memory_in`
= [mem_carry, learned-init]; source latent at τ=0 (A1); new frame denoised at frame 1 with K=4
shortcut steps; mem_carry written ONCE from the observed prefix via the W op (`return_memory` on a
windowed forward of the prefix held at tau_ctx), then injected unchanged (B1, static). Action-align:
src action = prev frame, new action = current. The generated latent becomes the next step's source.
Report A1+B1 as PRIMARY, A2+B1 as the shape-matched secondary (matches FF7 generate_memory shape),
and B2 only as a labeled OOD/drift demonstration — never the headline.

**Correctness traps confirmed:** (1) memory MUST be injected at the source frame the prediction
attends to (frame 0 of the window) — temporal attention is position-wise per slot, so a mis-placed
memory frame is invisible to the predicted frame; (2) write mem from the prefix once (do NOT
re-extract — that is B2 and drifts); (3) the color probe needs the predicted frame to be a REVEAL
(curtain UP) frame or `detect_ball` finds nothing.

**Reusable probes (`experiments/verify-T013-eval/`):** probe1_read_op.py (load-bearing via no-mem
ablation, visible source), probe2_occluded_source.py (load-bearing in the occluded regime — the
fair test), probe3_rollout.py ({A1,A2}×{B1,B2} color vs n_occ — the B2-drift detector),
probe4_floor_and_pos.py (no-mem floor + position). Pattern: to test whether an injected carrier is
load-bearing, ablate it with the learned-init token and measure the prediction-error gap; to expose
an untrained relay, roll it out past the trained horizon and watch monotonic drift.
