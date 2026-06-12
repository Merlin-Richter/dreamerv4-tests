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
| FF7 | **Single-timestep sufficiency** (Merlin): from ONE timestep's latent+register, predict the next **k frames (k small: 1–3, even 1)** under arbitrary actions. **k is the supervised lookahead depth, UNRELATED to the context window** — do NOT scale it to span occlusion. Retention is NOT from large k; it emerges because the loss is imposed at *every* timestep under *arbitrary* actions (incl. "lift curtain", reachable in 1 step) with the register carried recurrently: every occluded register must be able to produce the revealed ball next frame ⇒ must hold color ⇒ recurrence passes it forward indefinitely. Bellman/Q-learning logic — a 1-step-sufficient statistic under all actions is sufficient for the whole future (ties to FF8). **Overwrite latents with real latents** (stronger than detach: detach still lets the forward pass read color off the latent; overwriting with the color-free occluded latent forces the register to be the only carrier) so only **register tokens** learn off-screen info. Self-supervised (target = env's own future frames). Implies effective window→1 + register recurrence (MC1/MC2). | the proposed first attempt; stepwise training + detach-grad on carried state | untried | |
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

## Proposed first attempt (not yet committed — awaiting Merlin)
**FF7 × register-recurrence carrier × TR2/TR3** (Merlin's single-timestep-sufficiency
objective). Self-supervised, env-agnostic, satisfies the hard constraints. Carrier = the
register tokens propagated frame-to-frame (effective window→1); forcing function = predict
next-k frames from one truncated timestep with latents detached/overwritten so only
registers learn. Train stepwise with detach to control gradient explosion.
Open design questions to settle before building: (a) overwrite-with-real latents (preferred)
vs detach; (b) k small (1–3, likely 1) — NOT scaled to occlusion; random vs adversarial
actions; (c) the register recurrence: how register_t reads register_{t-1} at inference once
context→~1 (this is the carrier that must actually exist for FF7 to retain anything); (d)
keep the frozen tokenizer, change only the dynamics model's register pathway — yes, to stay
comparable on the probe.
