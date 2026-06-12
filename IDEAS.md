# IDEAS.md — H3 memory mechanisms: living registry

H3 (force the model to carry hidden/global state past the context window) is the
**high-iteration, high-failure** phase (Merlin, 2026-06-12). Capture every idea here the
moment it appears — anyone (Merlin or orchestrator) just adds a row. When an idea is tried
to any degree, append a one-line outcome + status + EXP/commit ref. Don't delete failed
ideas — the map of what doesn't work is half the contribution (protocol §8).

Status tags: `untried` | `trying` | `failed` | `partial` | `promising`.
Success bar (T-004, frozen): color ΔRGB < ~63 at n_occ ∈ {12,16,24} on the frozen probe.

## Hard constraints (Merlin, 2026-06-12) — non-negotiable, applies to every idea below
- **No privileged data to the model, EVER.** The model/training sees only environment
  observations + reward + data the env generated. No ground-truth color/pos/state fed in.
- **Must generalize across environments.** No bouncing-ball-specific hacks; the mechanism
  has to be environment-agnostic.
- **Eval exception:** our *measurement* instrumentation (the probe) may read the sim's
  hidden state to *score* recall — that's measurement, not a model input. Forbidden only
  as a training/inference signal to the model.

## The problem decomposed — three independent axes
An H3 attempt = one CARRIER × one FORCING FUNCTION × one TRAINING REGIME. Most failures
will be in the forcing function, not the carrier.

1. **CARRIER** — where hidden state physically survives window eviction.
2. **FORCING FUNCTION** — the loss/target that makes the carrier store the *right* thing.
3. **TRAINING REGIME** — how gradients flow once state persists across steps.

## Cross-cutting constraint (Merlin, 2026-06-12) — read before designing any attempt
Most carriers break **full-sequence parallel** training: persistence means you must roll
**timestep-by-timestep** and carry state. Gradients then link back indefinitely →
**exploding**; you must truncate/detach (TBPTT). **But truncation severs the credit path
from the reveal (loss) back to the observation (write)** — the very path memory needs.
Tension is fundamental. Self-supervised mitigations (privileged supervision is forbidden):
(a) **single-timestep-sufficiency** (FF7) — force the register to be a sufficient statistic
so the credit path is short (1 step → next-k) instead of reveal→observation; (b) **FF8**
bootstrapped horizon-1 credit (Q-learning analogy); (c) keep occlusion ≤ TBPTT span so the
gradient survives. Watch for: optimizer blowups (clip + detach), and the model gaming a
per-frame loss by emitting the color prior (= chance).

## A. Carriers (mechanism)
| ID | Idea | Notes | Status | Outcome / ref |
|----|------|-------|--------|---------------|
| MC1 | Persistent memory tokens (slots), read/write via attention, **exempt from window eviction** | minimal change; the existing `n_registers=4` scratch tokens are a natural home if made persistent | untried | |
| MC2 | Recurrent state (GRU/LSTM) summarizing the latent chain, state vector persists | explicit update; classic long-range | untried | |
| MC3 | SSM / Mamba-style linear recurrent state | stable long-range, cheap rollout | untried | |
| MC4 | Compressive memory: summarize evicted frames instead of dropping (Compressive-Transformer / ∞-former) | keeps a lossy trace of everything | untried | |
| MC5 | External read/write memory matrix (NTM/DNC), addressed by a write head | likely overkill, high complexity | untried | |
| MC6 | Slot-attention / RIM-style world-state slots updated each step | structured global state | untried | |
| MC7 | **Selective attention-salience memory** (Merlin): keep M timesteps; overwrite by a running *usage score* = how much each stored token/timestep is attended (QKV scores) by newly generated tokens. Frequently-recalled (esp. after gaps) → keep; rarely-used → evict. Brain-like consolidation, **no decider network needed**; RoPE likely tolerates missing middle positions. Variant: per-token retention (keep only the relevant tokens, e.g. the ones carrying the state) instead of whole timesteps. Needs retraining so the model learns to use very-old info + handle overwrites. | heuristic eviction, learned usage; future: learned salience score (maybe overkill) | untried | |

## B. Forcing functions (training target / loss) — THE HARD PART
| ID | Idea | Notes | Status | Outcome / ref |
|----|------|-------|--------|---------------|
| FF1 | Revisit-consistency loss at the reveal frame (our probe metric as the objective) | sparse, gameable by color prior; needs long credit path | untried | |
| FF2 | ~~Privileged decode-from-memory during occlusion~~ | **FORBIDDEN — feeds the model privileged hidden state. Violates the no-privileged-data constraint.** Kept only as a record of a rejected path. | forbidden | |
| FF3 | Predict-the-reveal (CPC / contrastive): memory must predict the future revealed observation | self-supervised, env-agnostic | untried | |
| FF4 | ~~Reconstruct hidden ball through occlusion from privileged target~~ | **FORBIDDEN — privileged target.** (A *self-supervised* version that reconstructs the model's own future *observations* is fine — that's FF7.) | forbidden | |
| FF5 | Memory-stability regularizer: penalize memory change when no new evidence arrives | anti-forgetting / anti-overwrite | untried | |
| FF6 | Auxiliary "what will I see if I look back" query head at reveal | task-shaped variant of FF1 | untried | |
| FF7 | **Single-timestep sufficiency** (Merlin): from ONE timestep's latent+register, predict the next **k frames (k small: 1–3, even 1)** under arbitrary actions. **k is the supervised lookahead depth, UNRELATED to the context window** — do NOT scale it to span occlusion. Retention is NOT from large k; it emerges because the loss is imposed at *every* timestep under *arbitrary* actions (incl. "lift curtain", reachable in 1 step) with the register carried recurrently: every occluded register must be able to produce the revealed ball next frame ⇒ must hold color ⇒ recurrence passes it forward indefinitely. Bellman/Q-learning logic — a 1-step-sufficient statistic under all actions is sufficient for the whole future (ties to FF8). **Overwrite latents with real latents** (stronger than detach: detach still lets the forward pass read color off the latent; overwriting with the color-free occluded latent forces the register to be the only carrier) so only **register tokens** learn off-screen info. Self-supervised (target = env's own future frames). **NO architecture change needed** — the carry already exists: temporal attention (`dynamics_model.py:110-121`) is *position-wise* over frames, so each register slot is its own causal channel through time; spatial layers move info latent↔register within a frame. FF7 is a training-procedure change only. Retention beyond the window = a **relay**: each frame re-copies the info into its own register from the previous frame before the source scrolls out; the per-frame sufficiency loss trains each hop locally (no single backprop spans the occlusion — Bellman). Retention length bounded by relay reliability, NOT by window or k. | the proposed first attempt; gradient flows from the k-rollout back through the registers into the normal windowed diffusion forward pass | untried | |
| FF8 | **Bootstrapped backward memory credit** (Merlin, speculative): propagate a "memory worked / didn't" signal back one timestep at a time, Q-learning-style (horizon-1 updates that still encode long-range outcomes). Unknown if mathematically sound — worthy attempt once FF7 works. | future; addresses long-range credit assignment without full BPTT | untried | |

## C. Training regimes
| ID | Idea | Notes | Status |
|----|------|-------|--------|
| TR1 | Full-sequence parallel (current) | **only valid for non-persistent carriers** — i.e. not real memory | n/a |
| TR2 | TBPTT: roll stepwise, backprop K steps, detach beyond | workhorse; pick K ≥ occlusion to preserve the credit path (cost/explosion tradeoff) | untried |
| TR3 | Detached carry: stop-grad on the carried state, learn only local read/write | cheapest; **requires** local supervision (FF2/FF4) since there's no long gradient | untried |

## Training augmentations (compose with any attempt)
- **Adversarial action policy** (Merlin): instead of testing memory under *random* actions,
  train an adversary that picks actions to make the memory fail. More sample-efficient
  pressure on the carrier than random rollouts. Compose with FF7.

## Proposed first attempt — FF7 v1 (CONVERGED with Merlin 2026-06-12; awaiting build go-ahead)
**FF7 single-timestep-sufficiency. NO architecture change** — registers already carry via
position-wise temporal attention (`dynamics_model.py:110-121`); the carrier is the existing
register-slot time channels + spatial latent↔register routing. This is a **training-procedure
change to `train_dynamics_model.py` only**; frozen tokenizer untouched.

Converged v1 scheme:
1. **One combined training step.** The normal windowed diffusion forward pass runs as today
   and produces each frame's register (built *with* window context — where the relay lives).
2. **FF7 loss on top:** for frame t, run a **window-1 rollout** — from frame t's tokens only
   (its register + its latent **overwritten with the real frozen-tokenizer latent**), generate
   t+1…t+k, each step seeing only the immediately preceding frame. Reconstruction loss on
   those k frames backprops *through the registers* into the windowed pass that built register_t.
3. **k small, start k=1** (k=1: register_t sufficient for t+1; k≥2 also trains the multi-hop
   register→register relay inside the rollout). k is lookahead depth, **NOT** scaled to occlusion.
4. **Overwrite latents with real latents** in the rollout (stronger than detach) so the register
   is the only carrier of off-screen state.
5. **"Arbitrary actions" — v1 simplification:** supervise the k-rollout on each episode's
   *actually executed* future; get action / curtain-timing diversity from the **dataset**
   (generate occluded episodes with curtains lifted at varied times). Counterfactual /
   adversarial action sampling (the adversary idea) is a follow-up if coverage is the bottleneck.
6. **Eval:** frozen probe (commit 5503e75), ≥2 seeds, against the T-004 bar
   (color ΔRGB < ~63 at n_occ ∈ {12,16,24}). Runs on the 4070.

Retention beyond the window comes from the **relay** (each frame re-copies state into its own
register before the source scrolls out), trained hop-by-hop by the per-frame loss — no single
backprop spans the occlusion (Bellman). FF8 = the future idea to extend retention further.

**Status: design locked, NOT yet committed as a decision (D-014) or built.** Gated on (a)
Merlin's build go-ahead and (b) the harness methods-critic gate question (see ESCALATIONS
ESC-005) — if we add a methods-critic, it should review D-014 before it's committed.
