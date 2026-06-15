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
| MC1 | Persistent memory tokens (slots), read/write via attention, **exempt from window eviction** | minimal change; the existing `n_registers=4` scratch tokens are a natural home if made persistent. **REFINED 2026-06-13 (Merlin):** make MEMORY tokens a *distinct* token type and revert REGISTER tokens to pure scratchpad — see "Memory tokens as a distinct carrier" below | **SELECTED 2026-06-13** | the carrier for the FF9 build Merlin chose; distinct MEMORY token type + registers revert to scratch. Planning (T-013). |
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
| FF9 | **Memory-only full-state sufficiency** (Merlin, 2026-06-13): from the MEMORY tokens **alone** (current latent withheld), predict the next k states under random/adversarial actions. Memory must encode the *full* state (on- AND off-screen), not just off-screen — withholding the latent is what forces that and prevents memory churn when visibility switches. See dedicated section below. **v2 (Merlin, after V-T013):** variable-horizon, pure-noise path — see below. | refines FF7; decouples memory from register-scratch; reduces target-chasing | **BUILDING (v2)** | V-T013 REFUTED v1 (loss solvable by local self-denoising; memory non-load-bearing under FF7's τ~Uniform+ramp). **FF9 v2 fix (Merlin):** per rollout pick j∈{1..k}; path frames t..t+j−1 at **τ=0** (NO GT latent → memory is the only carrier); **last frame t+j at sampled τ** (target exists; low-τ_j forces memory); loss on last frame only, **un-ramped**. Built as the architectural BASELINE (NOT a depth fix — within-window only; cross-window relay = option A next). D-024 / T-013 / EXP-017. |

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

**Status: go-ahead given (ESC-005), D-014 committed, building (T-009 → EXP-010).**
Build-time correction (D-014, code-grounded): registers do NOT persist across `generate()`
steps (each forward re-expands the learned tokens, dynamics_model.py:282; only latents carry
between steps, :405) — so FF7 also needs a **param-free inference change**: carry + inject
each frame's final-layer register state (`generate_memory`, the same interface the training
rollout trains). "train_dynamics_model.py ONLY" was too strong; zero-new-params still holds.

---

## Sequential stop-grad register-relay training (parked 2026-06-13, Merlin dialogue; not yet a decision)

**Problem.** Train/inference mismatch on the FF7 register channel. At inference
(`memory_rollout_step`) frame t-1's FINAL-layer register is injected as frame t's LAYER-0 input
register — an output→input recurrence that compounds over many steps. In the **main diffusion
loss** the context frames carry the *learned-init* register placeholder (`register_tokens.expand`,
dynamics_model.py:346; loss forward passes no `register_in`, :391); cross-frame register info flows
only *horizontally at matched depth* via temporal attention, never the deep relay. The **FF7 aux
loss** injects registers but only **1-deep** (the injected register came from a learned-init
context) and only the single context frame. So the model is never trained on deeply-relayed
registers like it sees at inference — exposure bias on the memory channel.

**Idea (Merlin's, refined).** Run the register relay **sequentially** at train time (a faithful
n-deep register requires the recurrence to have actually run — `r_t^in=f(r_{t-1}^out,frame_t)`,
non-associative, so no parallel scan; frame-by-frame is required). Carry the register **value**
across steps so context registers look like real relayed ones. Keep gradients **local** (no BPTT
past the window) — Transformer-XL / scheduled-sampling territory.

**The subtlety that decides success (read vs write side).** Fully detaching the carried register
(TBPTT-0) trains the model to READ realistic registers but gives ZERO gradient to WRITE a useful
one → may learn no memory. Two coherent options:
- (A) detached relay for read-side realism + keep FF7's in-pass injection for write-side training.
- (B) **TBPTT-1 (recommended):** carry the register value faithfully (n-deep) but keep ONE step of
  gradient — at step t the t-1 register is attached (trains t-1's write), everything folded into
  t-1 is detached. Trains read+write in one mechanism, gradient never exceeds one step, subsumes
  the FF7 aux loss into the rollout. Both keep O(1) memory in rollout length.

**Open forks to pin in the eventual decision:** (1) relay over GT latents in context (only the
*register* is relayed; latents teacher-forced — cheapest, isolates the memory-channel effect;
Merlin leaned here) vs self-generated latents (full closed-loop). (2) diffusion sampler: feed the
carried register alongside which latent, and do we differentiate the sampler (probably not).
**KV-cache relevance:** small for the window-1 FF7 relay (already cheap per step); the cache's
value grows with relay window size (then the detached cross-step window K/V = the TXL recurrence).
**Pre-registered question if pursued:** does TBPTT-1 sequential relay beat FF7-as-is on the frozen
memory probes (color beyond-window, and—if it helps—position)?

### Refinement (2026-06-13, Merlin dialogue) — the structural argument + efficiency

**Why this is the *only* way to train deep-occlusion memory (strongest motivation).** Verified in
code: the registers FF7 ever learns to *read* are written by the main full-clip forward
(`loss()`:391) and read exactly ONE hop later (`_ff7_loss` reg0=regs[:, :n_t] :460 → predicts t+1
:476-478). So every register the model interprets was minted in-flight from a window that still
held the info. Within a single parallel forward, anything outside the window is simply ABSENT —
there is no ground-truth signal to construct the example "register must hold info that already left
every window." Sampling clips differently can't fix this: if the source frame is in the clip it's
in the window (trained as in-window); if it's before the clip there's nothing to carry. **Only
carrying register STATE across the window boundary — a sequential relay — manufactures the
deep-occlusion regime.** That EXP-010 generalizes to it at all (color held < bar to n_occ 16,
never trained on it) is impressive OOD generalization; relay-training would put it in-distribution.

Two untrained regimes the relay fixes: (a) **read-side** — interpret a register whose info isn't
re-derivable from the window; (b) **preserve-side** — hold info stable across MANY output→input
relay hops (FF7 only trains write-from-window + read-1-hop, never preserve-across-N-hops).

**Color vs position asymmetry (mechanistic, predicts what relay-training buys).** Color is static:
the relay `reg_t=f(reg_{t-1}, occluded_frame)` has an easy COPY fixed point (occluded frame adds no
ball info → "pass register through unchanged"), position-invariant and easy to approximate → color
survives, and the slow approach to the bar by n_occ 24 is the *drift* of that approximate copy
compounding. Position is dynamic: no copy fixed point — the register would have to run a motion
integrator through occlusion, a real computation the 1-hop objective never trains → position at
chance. Prediction: relay-training should tighten color (less deep-hop drift) and is the ONLY thing
that could plausibly buy position (no free fixed-point to generalize from).

**Efficiency (Merlin).** Sequential relay loses the full-clip parallelism, so it needs careful
**sliding-window KV caching** (the absolute-RoPE foundation from D-017 already handles the
rotation-continuity for this) to be competitive. Expected profile: FLOPs ~comparable to full-clip,
but it becomes **memory-bandwidth-bound** (fetch cached K/V each step) + smaller per-step kernels
(low occupancy). Recover parallelism by **batching across episodes** (B large; sequential in time,
parallel across the batch). TBPTT-1 keeps *activation* memory LOW (graph for 1 step only, O(window)
not O(T)); the new memory cost is the KV cache itself (O(B·window·depth·heads·head_dim), bounded by
the sliding window). Net: memory footprint + bandwidth up, compute roughly even — matches Merlin's
read.

**Mixed context-window-size curriculum (Merlin).** Train most episodes at small **N≈4** (cheap per
step AND stresses the register harder — info older than 4 frames MUST live in the register), plus
some **N≈16** episodes so it also learns to reason over longer explicit context. N is the knob for
how much burden sits on the register (N=1 → register carries everything, like current FF7
inference; larger N → register only carries info older than N). Impl note: variable N is ragged —
bucket episodes by N into homogeneous batches (or pad) rather than mixing N within a batch.

---

## Memory tokens as a distinct carrier + memory-only sufficiency (Merlin, 2026-06-13)

**High-level plan (unchanged):** experiment with architectures + objectives to find one with
*persistent* memory. We're currently on the memory-token line; **its capability to persist memory is
still unproven** (EXP-010 = beyond-window COLOR only; EXP-013 = blind position near-absent). The ideas
below are a refinement of that line — captured for the record, not a committed build.

**Split the roles (carrier).**
- **MEMORY tokens** — a *distinct* token type, the persistent carrier; objective = *encapsulate the
  full current state*.
- **REGISTER tokens** — revert to their original role: throwaway scratchpad for the model, **no memory
  duty**. Decoupling stops FF7's conflation (the same `n_registers` tokens being both scratch and
  carrier). (Architecture note: this adds memory-token slots distinct from registers.)

**Objective (FF9, memory-only sufficiency) — concrete.** Given **only the memory tokens** (the current
latent is *withheld*), the model must predict the next **k** states under **random and/or
adversarially-picked** actions.
- *Stochasticity caveat (worth thinking about):* with inherent environment randomness the memory
  can't predict the exact future — the objective must tolerate it (encode the predictable/sufficient
  state; don't penalize env noise — distributional / expectation target, TBD).

**Why memory-ONLY, not latent+memory (the key choice).** If the loss provides current latent **and**
memory, the model learns to store *only off-screen* info in memory (the latent already supplies
on-screen) → **unstable**: when screen content switches (different things on/off screen) the model
must *reshuffle* the memory tokens. Feeding **only** memory forces it to be a clean,
**everything-included** object (full state, on- and off-screen), so its contents don't churn as
visibility changes.

**Why this reduces target-chasing.** A clear objective focused on a *few-frame reconstruction* from
memory-only is a more stable target than training memory against *other* memory tokens (a
self-referential moving target). Cleaner credit assignment, less chasing.

**Compute optimizations — DEFER until persistence is proven (performance-only, do NOT do first):**
- *Sparse memory frames:* memory tokens may not be needed every frame — e.g. add memory tokens to
  denoise only every 4th frame. Saves compute; revisit only once memory works.
- *Parallel multi-set training (slide-by-m):* don't train one memory set at a time — generate+train
  the next **4 or 8** memory sets in one parallel window, then slide the window by that much.
  *Example:* context window 16, memory tokens already known for frames 1–12 → generate (and train on)
  frames 13–16's memory in one step. The freshly-generated memory (13–16) isn't grounded yet, so keep
  it honest one of two ways (both should work):
  - (a) tell the model a **"memory noise level"** so e.g. memory@t=14 knows memory@t=13 is not-yet-real
    memory; OR
  - (b) simply **mask** memory@t=14 from attending to memory@t=13.
  This reclaims parallelism over the sequential relay — the "produce m memory sets per forward" form of
  the produce-vs-generate point below.

### Refinement (2026-06-13, pt 2) — "produce vs generate": the cost is NOT ÷window

From the EXP-015 perf discussion. The ÷window / ~0.15×-throughput fear applies to **generating**
during training (per-frame, K denoising substeps, sequential). But relay/memory training never has to
generate: **latents are teacher-forceable** (GT frame → frozen encoder, fully parallel), so the only
thing with no ground truth — the thing we must *produce by running the model* — is the **memory/register
state** itself (Merlin's asymmetry: unlike latents, memory has no label, it's a model-dependent
recurrent quantity). Producing the handoff state is **one forward per window** (teacher-forced latents
at tau_ctx, **no denoising substeps**), and it can be `no_grad` (stop-grad handoff / TBPTT-1). So the
recurrence is **window-granular**: cost ≈ (relay depth in windows)+1 forwards + one parallel grad step
≈ **2–4× plain training, NOT ÷window**.
- **The genuine blocker is learning, not throughput.** With a stop-grad handoff you train *carry* but
  not *store* (credit assignment): gradient must reach the write step (≥ TBPTT-1 through the last
  production window) or the model never learns to put the right thing in (bootstrap trap), compounded
  by handoff **staleness** (the memory is a moving target as the model updates — target-network-like).
- **Where the KV cache (T-012) earns its place in training:** it accelerates exactly the no-grad
  memory-PRODUCTION rollout — the "we can't get the memory tokens for free like latents" step — most
  when the handoff is deep or frame-granular. Not on the gradient path; on the production path.
- The slide-by-m trick (above) is the same insight from the other side: produce m memory sets per
  parallel forward to claw parallelism back.

---

## Memory handling — the three operations + "memory is an activation" (Merlin, 2026-06-14)

A complete memory training must include all THREE operations; FF9 v2 (built) has only 1 & 2:
1. **write memory ← latents** — HAVE. The main windowed pass writes memory_t from real latents.
2. **read memory → latents** — HAVE. FF9 v2 reads injected memory to predict the next 1..j frames.
3. **write memory ← memory** — MISSING. memory_t written from the PREVIOUS (relayed) memory + frame_t.
   = the relay WRITE-side = parked **option A** = V-T013 finding (2) "preserve-across-N-hops". FF9 v2
   never trains it: the context frames that write memory carry UNIFORM learned-init memory (no info to
   read). So including op-3 is **A+B converging** — which V-T013 predicted is needed for depth.

**What memory IS now: an ACTIVATION** — the final-layer hidden state of the memory-token positions
(`forward` reads `x[:,:,mem_start:...]` at the last block). NOT a denoising variable: no signal level /
noise, not x-predicted, no GT target (memory is a model-internal recurrent quantity with no label —
Merlin's produce-vs-generate asymmetry). The relay is an **output-activation → next-frame-layer-0-input**
recurrence (FF7 `memory_rollout_init/step`; FF9's `generate_full_state_memory`, not yet built).

**Why activation is correct AND enables op-3 training:** memory has no GT to denoise toward → denoising is
ill-defined; as an activation it is PRODUCED by one forward per frame (teacher-forced latents → no K
substeps), cacheable (T-012). So Merlin's op-3 recipe is cheap: run the (mostly no-grad, cached)
production rollout ~200 steps to fill the window with REAL relayed memory, then a gradient-carrying step
writes memory_t from that context + applies the FF9 loss; gradient **bounded to the window** (no
out-of-window BPTT). This is the sequential relay realized on the memory carrier.

**Design risk to verify:** relaying a FINAL-layer activation into LAYER-0 input over ~200 hops may drift /
explode (representational mismatch / exposure bias). Op-3 training is partly meant to cure it, but
stability (norm / detach-on-overflow / a relay projection) needs care → design note + verifier before build.

**Status:** op-3 = A+B combined. Pending: (i) sequencing decision (run ops-1&2 FF9 baseline first as a
cheap checkpoint, or fold op-3 in before any training run); (ii) op-3 design note → critical-claim-verifier
→ D-025 → build (with the cached production rollout). FF9 v2 baseline (ops 1&2) is built + smoke-green.

## Uncertainty-aware rollout: context_signal as a per-frame confidence channel (Merlin, 2026-06-16)

**Origin.** Merlin asked how rollouts treat noise on self-generated frames. Code answer (D-028,
dynamics_model.py): `generate()` -> `_denoise_next` stores each generated frame as a CLEAN
100%-signal latent (the x-prediction, appended verbatim) and, on the next step, noises the WHOLE
context window (real prefix + all self-generated frames, undifferentiated) to a single global
`context_signal=0.9`. So there is **no provenance marker and no per-frame confidence channel** — a
frame the model hallucinated is fed back at the exact same trust level as a real observation. This is
the exposure-bias mechanism EXP-018 diagnosed, in concrete form.

**The lever.** The model already conditions on a PER-FRAME signal level (`tau_col`), so we can vary
trust per context frame with NO new params: feed observed frames near-clean (tau≈0.95–1.0) and
self-generated frames at a LOWER signal s(j) that reflects their unreliability (optionally decaying
with how many self-steps ago frame j was produced). This *tells* the model "this frame is my own
guess, don't over-trust it" — which neither the rollout nor C1 currently does.

**Why it's plausibly in-distribution (not a hack).** Training (shortcut-forcing main loss) noises
every frame at an INDEPENDENT random tau, so the model has already seen context frames at the full
tau range, including low signal. Feeding generated frames at lower tau at inference is therefore not
OOD the way the untrained memory->memory relay (V-T014) was.

**The tension / degenerate modes (why it might NOT help).**
- Too-low signal throws away usable trajectory info (regresses toward the original 90%-noise inference
  bug, EXP-008): a self-generated frame that is actually good shouldn't be half-erased.
- Lowering context trust can push the model to IGNORE context and emit a context-independent prior
  (the prior-emission collapse V-T013 / V-T017-C1 C-B warned about) -> MUST pair with the open-loop
  displacement collapse monitor (src/eval/motion.open_loop_displacement).
- So the hypothesis is a MILD reduction / schedule has a sweet spot, not "more noise = better."

**Relationship to C1.** Coupled, not competing. C1 fixes the *distribution* half of exposure bias
(trains on self-generated context); this fixes the *confidence-representation* half. If inference
feeds generated frames at signal s, C1's self-rollout (`_multistep_loss`, currently `_noise_to_ctx`
at `context_signal`) should train at the SAME s (or schedule) for consistency. Cleanest combined
version: one tau schedule used by both C1 training and rollout inference.

**CHEAP first probe (inference-only, NO retrain, existing checkpoints) — do this BEFORE building.**
Sweep `context_signal` at INFERENCE and measure the OPEN-LOOP pos_err curve (the existing P2 τ-sweep
is teacher-forced 1-step; this is the open-loop analog) on vanilla_s0 / ff7_k3 / the C1 checkpoint.
`dyn.config.context_signal` is already runtime-overridable (tau_context_sweep does it). If varying
inference trust measurably shifts the open-loop compounding curve, the lever has signal and a trained
per-frame-confidence version is worth building; if the curve is flat in context_signal, the confidence
channel is inert and we drop it. Discriminating, ~minutes of GPU, reuses src/eval/motion. (Extension:
per-frame DECAY schedule s(j) vs flat s, both inference-only via a small generate() shim.)

**Status:** idea + cheap probe specified; not a decision yet. Candidate as an EXP-020/EXP-021 follow-on
once C1's base value is established (don't conflate two levers in one run).
