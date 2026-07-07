# ColorField-SYM — symbolic (tokenizer-free) tier of the autoresearch harness

Merlin's design (2026-07-07, signed off): remove the tokenizer entirely; the dynamics model
consumes the symbolic viewport directly. Purpose: the pixel tier's 20-min budgets died learning
the APPEARANCE prior (calibration log in autoresearch-harness.md); one-hot cells have no
appearance — budgets go to dynamics + memory, steps get ~20-50× cheaper, eval episodes ~100×
cheaper (no encode/decode). THE fix for Karpathy-density experiments on the 4070.

**Tier ladder (keeps the video-model thesis honest):** colorfield-sym (fast search, THIS) →
pixel ColorField (sealed frozen-v2.1 layer = promotion gate; relay must compress lossy latents,
not copy crisp symbols) → memmaze (the real claim). Symbolic wins are directional until they
re-validate up the ladder.

**THE GENERALITY RULE (Merlin, verbatim intent):** editable-layer changes must work in ANY
environment. Upweighting "every 5th frame" = CHEATING (env-specific knowledge); upweighting by
measured information novelty = legitimate (general mechanism). Goes in program.md + kept-diff
review. Memory: project-autoresearch-generality-rule.

## Env design (colorfield-sym v1)

- Board: 15×15 iid cells over 5 colors + OUT outside (same as pixel tier; same palette ids).
- **Viewport: 5×5 CELLS**, center cell c ∈ [0,14]² (viewport extends past borders: up to 2
  rows/cols of OUT visible at edges — the border anchor; corner start anchors all borders).
- Observation per tick: (5,5) uint8 grid of palette ids (0..4 map, 5=OUT) + phase ∈ {0..4}.
  (One-hot encoding is the MODEL side's job, not the env's.)
- **Phase-5 time dilation (Merlin)**: moves apply only at ticks with t % 5 == 0; on off-phase
  ticks the env FORCES STAY — valid_actions(off-phase) == [STAY], anything else raises (uniform
  invalid-action semantics). At phase-0: up/down/left/right (one cell) / stay; outward at
  board edge INVALID (raises, cannot be tried). Rationale: W=16 ticks then covers only 3.2
  effective moves (≈ pixel tier's 2.7-cell reach) — keeps relay training healthy while most of
  the board stays out-of-window. 25/225 = 11% visible.
- Episode T = 1024 ticks (~205 effective moves). Procedural storage (maps/starts/actions),
  render = trivial slicing. Fully deterministic.

## Datagen

Same 8-policy zoo, ported to the 15×15 center lattice (policies consulted at phase-0 only;
amplitude/lane parameters ÷6 vs the 90-lattice pixel versions, same spatial scale). Policy id
logged. ~5000 train + 250 val episodes, fresh map+start each.

## Eval (v2.1 semantics, simpler — no partial visibility exists)

- on-screen := cell in the 5×5 viewport; fully-left := not in viewport. Comeback = in → out →
  in, scored once per event at the return visit. NO partial-overlap machinery (the pixel tier's
  hovering exploits are structurally impossible; multi-gap bridging still real ⇒ **age = max
  contiguous out-of-viewport run, in TICKS** — unchanged).
- Two-provenance tracker (prefix real / imagination-born), same composite:
  real_cc × (0.7 + 0.3·consistency_cc); border (OUT) cells excluded from score, border_recall
  diagnostic; chance correction c = 1/5.
- Readout = identity/argmax (exact by construction); oracle must score exactly 1.0.
- Closed-loop eval policies: consult OUT band width at viewport edges (band == 2 ⟺ center at
  board edge ⟺ outward invalid — exact analogue of the pixel band≥30 rule); act at phase-0
  only, STAY off-phase. Prefix ~192 ticks from a corner start; imagination ~768 ticks.
- Gates: (1) fidelity — at phase-0 the predicted grid must equal the previous grid shifted by
  the action; PLUS off-phase ticks must predict an UNCHANGED grid (new, free gate); measured
  exactly on symbols. (2) entropy of imagination-born first-seen colors (unchanged).
- Age bins: min possible age ≈ 10-14 ticks (leave-and-return takes ≥2 effective moves × 5) —
  keep (1,17,33,65,129,257) but expect bin1 sparse; adjust at freeze if occupancy < min_events.
- Baselines ported: oracle, perfect_imaginary (grid-phase problem doesn't exist — symbols),
  noise, constant, copy_last, bounded-window regression fence.

## Model port (editable layer — the elegant part)

The DynamicsModel is unchanged: treat the one-hot viewport AS the latents —
**n_latents = 5 tokens (one per viewport row), bottleneck_dim = 35** (5 cells × 6 one-hot + 5
phase dims appended to every row; phase prediction is trivial and harmless). x-prediction
target = the one-hot rows themselves. No tokenizer anywhere; adapter decode = argmax.
train.py: dataset builds one-hot "latents" from symbolic frames on the fly (cheap, no cache
needed). W_PIN=16 (ticks), n_actions=5 unchanged.

## Build plan

frozen_sym/{env.py, datagen.py, eval_comeback.py (+tracker), eval_policies.py, adapters.py,
tests/} mirroring autoresearch/frozen/ 1:1 where semantics are shared; editable/train_sym.py +
adapter_sym.py (thin variants). Gate tests incl. oracle==1.0, brute-force tracker cross-check,
bounded-window fence, forced-STAY/phase semantics, policy validity fuzz. Then: 20-min budget
probe on the 4070 (prediction: crisp cells + fidelity gate passing within minutes; the memory
curve becomes the object of study). Focused adversarial delta-review (fidelity gate, no-partial
tracker, phase semantics) before MANIFEST-sym freeze.

## Done when

Gate tests green; 20-min probe run + sheets/eval reviewed by Merlin; adversarial delta-review
processed; MANIFEST-sym recorded; harness calibration retargeted to this tier
(autoresearch-harness.md updated).
