# DESIGN: sparse memory writes + spatial injection of the stale set + staleness conditioning

> **SUPERSEDED (v2) by `sparse-memory-write-slots.md` (v3, Merlin 2026-07-04):**
> memory attends only to every-nth memory set (masked temporal channel); no injection, no
> broadcast keys. Kept for the analysis/scrutiny record.

Proposed by Merlin 2026-07-04. Status: DRAFT under scrutiny — supersedes the READ mechanism of
`sparse-memory-tokens.md` (Design A's temporal broadcast keys are dropped; this promotes that
draft's Design B from "cheap prototype" to the actual design, plus staleness conditioning and a
larger-memory-set axis).

## 1. Context (what exists today)

- **Memory tokens now**: `n_memory` tokens in every frame's block
  `[action | latents | registers | memory | shortcut]`. Spatial layers mix memory ↔ latents
  within a frame; temporal layers are strictly SLOT-WISE (memory slot j at frame t attends only
  to slot j at frames ≤ t). At rollout, each committed frame's *written* memory (final-layer
  memory-slot activations) is injected at the input of its own commit pass (`memory_in`), its K/V
  enters the carrying cache, and later frames' memory slots read it through their temporal
  channel: the relay is **read-old / write-new at EVERY frame**.
- **Training that works**: the mem2mem sliding rollout (50% of slides hide the whole window's
  latents at τ=0 → the carried memory is the only scene carrier; GT flow targets). FF9 is NOT
  necessary on GridWorld (`mem2mem-rollout-noff9-fair`: no-FF9 recall ≈ 1.0 flat to k=64); the
  memmaze no-FF9 arm (415143) is running. Relay gradient EXPLODES ~2–3×/hop at init (probes in
  `mem2mem-rollout-noff9-fair`), self-correcting once trained; per-hop grad clip exists, off by
  default.
- **Recent relevant facts**: the honest vanilla baseline (τ0-anchor) proves in-window competence
  needs no memory; GQA (`gqa_groups`, now in src+spec) cuts cached K/V 4×; the eval spine
  (GridWorld recall + sheets) is model-agnostic.

## 2. The idea

**Write memory sparsely; read it spatially; condition on its age. Optionally make each set bigger.**

1. **Sparse writes (1/8× or 1/16×).** A fresh memory set is written only at write-frames
   `t_w ≡ 0 (mod N)`, N ≈ 8–16. Motivation (primary): each memory set should fully encapsulate
   the ~Markov state of the env; demanding a full re-encoding of that state after EVERY frame is
   redundant, and each per-frame rewrite is a lossy re-encoding — a chain of T rewrites gives
   errors T opportunities to compound. Updating the belief every N frames cuts the rewrite chain
   ÷N. Compute savings are explicitly NOT the main point.
2. **Spatial read (the mechanism change).** The set written at `t_w` is INJECTED (`memory_in`)
   into the per-frame token block of every subsequent frame `t_w+1 .. t_w+N` during rollout — so
   frame t's SPATIAL attention reads the memory tokens directly, within-frame, at every layer,
   from every slot. No temporal-attention surgery (Design A's broadcast keys dropped). The
   temporal channel's job becomes: let the model check what has CHANGED since the memory was
   written (e.g. "the inventory was opened / something was added after this set was created"),
   using the ordinary latent/register channels of the in-window frames.
3. **Staleness conditioning (required for 2 to be safe).** The model must know how old the
   injected set is, so it knows how far back the temporal check must reach and how much
   extrapolation the stored state needs. Default mechanism: a learned **age embedding** added to
   every injected memory token (discrete lookup on `age = t − t_w ∈ {0..N}`, mirroring how action
   features are added to the action token; no layout change). Alternative: one extra "staleness
   token" per frame. At the write frame the fresh output is captured; the injected set there is
   the previous one at age N (the write is a function of old set + full in-window evidence).
4. **Bigger sets.** With writes 8–16× rarer, each set may be LARGER (`n_memory` up, e.g. 8 → 16
   or 32 on memmaze) — capacity per write rises where the per-write information (a whole segment
   + carried past) is larger. Note honestly: injected tokens cost spatial attention at EVERY
   frame, so set size is not free; what sparsity actually frees is the write path and the
   temporal-channel/cache clutter (cache further cheapened 4× by GQA).

## 3. Mechanics (spelled out)

- **Rollout** (`rollout_step` cadence): keep memory slots in every frame's layout (rectangular
  tensors, zero architecture change). At every commit, inject `memory_in = M(t_w)` + age
  embedding. When `pos % N == 0`, capture the commit pass's memory-slot outputs as the new
  `M(t_w)`. The read-only branch (`commit=False`) NEVER writes or advances the set — the recall
  eval's contract stays absolute.
- **Training** (mem2mem-trainer variant): identical sliding rollout; the injection cadence
  mirrors inference (fresh capture every Nth frame, stale+age injection between); the 50%
  full-noise mode now trains exactly the needed skill — reconstruct any frame in the gap from
  (stale set, age) alone. The FF9 analogue, if ever needed, is "latents at τ=0, inject M(t_w) at
  age j, reconstruct t_w+j" — but the no-FF9 recipe is expected to carry it (gated by 415143).
- **The relay across writes** still exists at period N (write at t_w reads the set from t_w−N);
  TBPTT spans N× more env-time at the same graph depth.
- **Semantics shift to note**: the current spec says memory is relayed by reading old K/V, "not
  threaded forward as a frozen activation". This design DOES thread the written set forward
  physically — but only for ≤N frames (bounded staleness), then a fresh rewrite; and the injected
  set is re-processed by every layer of each reading frame (only its INPUT is frozen). It also
  sidesteps the frozen-deep-K/V subtlety (V-cache-equiv divergence) for memory reads.

## 4. What capability actually moves (the honest trade)

Dense relay: belief advanced +1 tick per frame (easy per-step job; T lossy rewrites).
Sparse+stale: belief stored at `t_w` must be extrapolated by UP TO N ticks at READ time
(`state(t) = g(M(t_w), age)`) — harder per-read job; T/N lossy rewrites. For slowly-changing /
static hidden state (memmaze layout, object positions, inventory) the extrapolation is ~identity
and the design is near-pure win. For fast dynamics behind occlusion (GridWorld square: position,
velocity, wall reflections over ≤N ticks) the read-time extrapolation is a real added burden —
FF9 models do k-step POSITION extrapolation only to ~k=10 (1.0 to k≈8, 0.70@k12, decaying —
R-gridworld-retrain2), so read-time extrapolation of DYNAMIC state is learnable only for small
ages; this is the axis on which the design can lose to the dense relay, and it argues N ≤ ~8 for
envs with fast hidden dynamics. Note also (scrutiny finding): under FULL occlusion the
"temporal-check-what-changed" mechanism is inert — occluded frames' latents encode the curtain,
not the scene, so the gap is bridged by pure (stale set, age) extrapolation. The check mechanism
is alive precisely in the PARTIAL-observability regime (memmaze: gap frames are visible and
informative, e.g. the inventory-opened event sits in an in-window frame) — a further reason the
design's natural home is memmaze, not GridWorld recall.

## 5. Evidence status of the central premise

"Per-frame rewriting compounds errors" is a HYPOTHESIS, not an established result here:
GridWorld's dense relay held position recall ~1.0 FLAT to k=64 (411133) — no visible compounding
on that env. [CORRECTED after independent scrutiny, 2026-07-04: the backward-path explosion
(~2–3×/hop at init) is NOT evidence of forward error compounding — it is an init-time backward
transient that self-regularizes to ≈1 once trained, while the trained FORWARD per-hop factor is
≈1.0 (probe_relay_decay.py). No forward-compounding evidence exists in this repo.] GridWorld's
Markov state is tiny (position+velocity+color), so its flatness does not certify the dense relay
at memmaze scale (richer state, 8 tokens, W=32, longer horizons) — but equally, nothing yet shows
the dense relay decays there. The design's value must be shown ON MEMMAZE, against the
dense-relay mem2mem arm, same compute — and the dense relay's memmaze decay must be MEASURED
FIRST (a fix for compounding cannot be demonstrated before the compounding is).

## 6. Discriminating experiments (cheapest first)

1. **GridWorld prototype via `--model-module`** (subclass injecting stale sets + age embedding;
   mem2mem-trainer variant with the sparse cadence): N ∈ {1, 8, 16} at fixed n_memory=4, winner
   recipe minus FF9. Recall w8 max_k64 vs the dense winner (0.99 flat). Also report recall as a
   function of `k mod N` (periodicity there = reads leaning on write boundaries).
2. **Capacity axis**: N=8 with n_memory ∈ {4, 16}. Tests "sparser but bigger beats denser but
   smaller" at matched token-count×writes.
3. **Memmaze arm** at the winning (N, n_memory) — same 512/12/16 W=32 recipe, vs 415104/415143 —
   where the premise (§5) actually predicts a win.

## 7. Risks / open questions

- Read-time extrapolation burden (§4) — the design's main way to lose on dynamic state.
- Events that occur AND leave the window within one gap are lost only if N > W−1; with N ≤ 16 ≤
  W−1 (memmaze W=32) every frame is in-window for ≥1 write — the old draft's invariant holds.
- Staleness embedding must actually be used (check: ablate age → recall inside the gap should
  degrade; if not, the model ignores age and may be interpolating unsafely).
- Attention imbalance: the injected set is visible to every slot at every frame — watch for
  attention-sink behavior on memory rows.
- Train/inference cadence must match exactly (write phase alignment; window slides vs write
  boundaries).
- Interaction with long-context prefill (`rollout_init` teacher-forced commits): the prefill
  must run the same cadence, capturing writes at true-frame write-frames.

## 8. Independent scrutiny verdict (critical-claim-verifier, 2026-07-04)

Full analysis delivered to Merlin; the load-bearing findings, incorporated above:

- **Premise (per-frame rewriting compounds errors): CONTRADICTED on GridWorld** (411133 flat to
  k=64; trained forward per-hop factor ~1.0), **UNMEASURED on memmaze** (no recall eval exists
  yet). The backward-explosion citation was a category error (init-time backward transient, not
  forward drift) — fixed in SS5.
- **Goal-attainment: conditionally possible** — for slowly-changing/static hidden state and small
  N; for dynamic state the read-time extrapolation demand lands exactly where the repo's evidence
  is weakest (position 0.70@k12) -> N <= ~8 there.
- **Novelty: this is the old draft's Design B + age embedding + capacity knob**, with the honest
  additions that (i) "temporal channel freed" is aspirational while memory slots stay in every
  frame's layout, and (ii) the trainer change is a real rewrite of the carry loop
  (window-relative vs absolute positions; W/2 slides misalign with write cadence), not a flag.
- **Gaps found**: 415143 (dense no-FF9) does NOT gate the sparse gradient-density question;
  occlusion onset straddling a write boundary is a one-gap transient the dense relay doesn't have;
  write-as-copy degenerate needs an across-write decay check (recall at k = multiples of N).
- **The age embedding is the one clearly load-bearing correct addition** — keep it, gate it on an
  ablation (mask age at eval -> in-gap recall must degrade).
- **Recommended sequencing (falsify cheapest first):**
  1. Build the memmaze recall/probe eval (already the campaign's gating task) and MEASURE whether
     the dense relay (415104) decays with horizon on memmaze. If it is flat -> the premise is dead
     everywhere measurable and the design is a hedge, not a fix.
  2. Only if dense decays: GridWorld/memmaze sparse prototype via --model-module, scored by the
     FALSIFYING probes: position_acc at maximal staleness (age = N-1), sawtooth in k mod N,
     position-vs-color decomposition, age-embedding ablation. Confirming framing ("recall vs the
     dense winner") is not sufficient.
