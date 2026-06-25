# FF9 — memory tokens for world-model memory

> Canonical short statement of FF9 + the op-3 rollout-training extension. Updated 2026-06-25 with
> what the EXP-029..033 campaign established (key correction: the inference did NOT carry memory).

## What FF9 is
The dynamics transformer carries, per frame, a set of **memory tokens** (a distinct token type, M=4–16)
alongside the latent/register/shortcut tokens. They are an **activation** (final-layer hidden state), not
a denoising variable — no ground-truth label.

**Architecture wiring (from code):** spatial blocks do full unmasked attention within a frame
(`[action|latents|registers|memory|shortcut]`) → memory↔latents mix **within a frame**. Temporal blocks
(every 4th) attend **position-wise per slot, causally** → each memory slot is its own causal channel
through time and does NOT see other-time latents directly. So latents and memory mix only within a frame;
the temporal layers carry each channel forward.

## The training today (`_ff9_loss`, "sufficiency")
For each frame *t*: a real memory token (written from the main windowed pass) is injected at *t*, the path
latents are set to **τ=0 (pure noise)**, and the model must reconstruct the next 1..j frames from **memory
alone** under the realized actions. Forces ops **(1) write memory←latents** and **(2) read memory→latents**.
The FF9-no-rollout model (this loss only) is the best memory model we have.

## ⚠️ CORRECTION (2026-06-25): the inference does NOT carry memory
The earlier claim here — "inference is the ordinary rollout where memory tokens are carried in the window
via temporal attention" — is **FALSE**, verified from the code. `generate` / `generate_cached(plain=True)`
pass **`memory_in=None` on every forward**, so the memory tokens are **re-initialised from the learned
parameter each step and never threaded**; the written memory activation is discarded. The cache is rebuilt
per frame (no cross-frame persistence). **The only thing carried across frames is the latent sequence in
the sliding window.** So under the inference used so far, the FF9 memory tokens are within-window scratch,
not a carried state — the "memory beyond the window" premise was never actually exercised.
- *Hypothesis (unverified):* FF9 beats vanilla past the window because the **generated occluded latents
  themselves encode the dead-reckoned position** (an open-loop latent relay), and the memory tokens
  improve the latent the model writes. Decay (to ~chance by k≈28) is open-loop compounding, not a window
  cliff. Needs a probe (linear-probe generated occluded latents for position; ablate memory at inference).

## op-3 (write memory ← memory) and the rollout-training extension
The sufficiency loss never puts **real, previously-written memory tokens in the context** and asks the
model to read them and write the next — so **memory→memory propagation is untrained** (op-3). The
rollout-training extension trains it: roll real memory hops, carry the written memory forward, keep the
autograd graph k hops (TBPTT-k); per-step hide the latents (memory-only) vs near-clean (re-anchor).

### Result (EXP-030/031/033) — NEGATIVE as implemented
Built + independently verified correct, then trained (M4 h24; M4 h44 deep; M16). **Under the correct
normal windowed inference it REGRESSES** — FF9-no-rollout stays best; rollout-trained models are worse,
near the vanilla floor past k12; wider memory (M16) does not rescue it. The overnight "win" was an
**artifact of a W=2 noise-source inference I built that crippled the baseline** (now deleted).
- **Root cause:** train/inference mismatch. The rollout loss threads a written memory activation forward
  (`mem_carry = mem_out`), but the inference **never consumes a carried memory activation** — so the
  trained behaviour has no inference counterpart. It also trained an isolated 2-frame, memory-only,
  noise-source regime far off the real windowed distribution.
- **Lesson:** the idea is only testable if **inference actually carries the memory token forward AND
  training matches that inference.** Same mechanism at train and test. (This is the current work item.)

## Design knobs (Merlin, 2026-06-24) — for the matched redesign
- Warmup the rollout fraction 0→~50% by wall-clock (contain, then propagate).
- Hide latents per-loss-step, all-or-nothing (whole context hidden or visible); memory-only steps give
  the carry gradient, visible steps re-anchor so a wrong guess can't compound forever.
- Teacher-force GT context latents (near-clean) so the only rolled-out recurrent element is the memory.
- Newest-frame flow loss only (matches inference); sufficiency loss stays multi-frame.
- TBPTT depth k is the key knob (V-T014: tbptt-1 insufficient, deeper needed); measure, don't hard-code.
- Don't over-punish butterfly effects (matters only in stochastic envs; GridWorld is deterministic).

## North star
Make the memory tokens the **recurrent world-state** (DreamerV4 h-state) and run imagination/policy purely
in memory space, decoding latents only when pixels are needed. M=4 may be too small for full-scene
memory-only imagination — capacity is an empirical knob.

## Provenance
FF9 + op-3 idea: Merlin 2026-06-24. Rollout-training built/tested + the inference correction:
EXP-029..033 campaign (D-048), 2026-06-25. Relates to IDEAS.md "three operations", ESC-014/V-T014.
