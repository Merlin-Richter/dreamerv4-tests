# IDEAS.md — H3 memory mechanisms: living registry

H3 (force the model to carry hidden/global state past the context window) is the
**high-iteration, high-failure** phase (Merlin, 2026-06-12). Capture every idea here the
moment it appears — anyone (Merlin or orchestrator) just adds a row. When an idea is tried
to any degree, append a one-line outcome + status + EXP/commit ref. Don't delete failed
ideas — the map of what doesn't work is half the contribution (protocol §8).

Status tags: `untried` | `trying` | `failed` | `partial` | `promising`.
Success bar (T-004, frozen): color ΔRGB < ~63 at n_occ ∈ {12,16,24} on the frozen probe.

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
Tension is fundamental. Main mitigation: **observation-time / during-occlusion auxiliary
supervision** (privileged, sim-only) so the write is trained *locally* and doesn't depend
on a long gradient surviving truncation. Watch for: optimizer blowups (clip + detach),
and the model gaming a per-frame loss by emitting the color prior (= chance).

## A. Carriers (mechanism)
| ID | Idea | Notes | Status | Outcome / ref |
|----|------|-------|--------|---------------|
| MC1 | Persistent memory tokens (slots), read/write via attention, **exempt from window eviction** | minimal change; the existing `n_registers=4` scratch tokens are a natural home if made persistent | untried | |
| MC2 | Recurrent state (GRU/LSTM) summarizing the latent chain, state vector persists | explicit update; classic long-range | untried | |
| MC3 | SSM / Mamba-style linear recurrent state | stable long-range, cheap rollout | untried | |
| MC4 | Compressive memory: summarize evicted frames instead of dropping (Compressive-Transformer / ∞-former) | keeps a lossy trace of everything | untried | |
| MC5 | External read/write memory matrix (NTM/DNC), addressed by a write head | likely overkill, high complexity | untried | |
| MC6 | Slot-attention / RIM-style world-state slots updated each step | structured global state | untried | |

## B. Forcing functions (training target / loss) — THE HARD PART
| ID | Idea | Notes | Status | Outcome / ref |
|----|------|-------|--------|---------------|
| FF1 | Revisit-consistency loss at the reveal frame (our probe metric as the objective) | sparse, gameable by color prior; needs long credit path | untried | |
| FF2 | **Privileged decode-from-memory during occlusion**: small head predicts hidden color/pos from memory tokens every occluded step | strong *local* signal; dodges long-range credit assignment; sim-only crutch | untried | |
| FF3 | Predict-the-reveal (CPC / contrastive): memory must predict the future revealed observation | self-supervised, no privileged labels | untried | |
| FF4 | Reconstruct the hidden ball through occlusion (privileged latent/pixel target each step) | like FF2 but full reconstruction | untried | |
| FF5 | Memory-stability regularizer: penalize memory change when no new evidence arrives | anti-forgetting / anti-overwrite | untried | |
| FF6 | Auxiliary "what will I see if I look back" query head at reveal | task-shaped variant of FF1 | untried | |

## C. Training regimes
| ID | Idea | Notes | Status |
|----|------|-------|--------|
| TR1 | Full-sequence parallel (current) | **only valid for non-persistent carriers** — i.e. not real memory | n/a |
| TR2 | TBPTT: roll stepwise, backprop K steps, detach beyond | workhorse; pick K ≥ occlusion to preserve the credit path (cost/explosion tradeoff) | untried |
| TR3 | Detached carry: stop-grad on the carried state, learn only local read/write | cheapest; **requires** local supervision (FF2/FF4) since there's no long gradient | untried |

## Privileged vs self-supervised
We have sim ground truth (ball color/pos). **Privileged** losses (FF2/FF4) = fast mechanism
de-risking but a crutch unavailable on real video. **Self-supervised** (FF1/FF3) is the real
goal. Plan: prove a carrier can beat the cliff *privileged*, then ablate the crutch away.

## Proposed first attempt (not yet committed — awaiting Merlin)
**MC1 × FF2 × TR3** (persistent tokens · privileged decode-from-memory · detached carry).
Rationale: de-risks the carrier and *sidesteps* the long-range credit-assignment wall, so a
failure here cleanly implicates the carrier rather than the optimizer. Cheap on the 4070.
If it beats the cliff → ablate toward FF1/FF3 (self-supervised) and TR2 (real credit path).
If it *can't* even privileged → the carrier (MC1) is wrong, move down the carrier list.
