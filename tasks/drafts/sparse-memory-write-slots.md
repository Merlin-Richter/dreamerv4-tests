# DESIGN v3: sparse WRITE-SLOTS — memory attends only to every-nth memory set

Proposed by Merlin 2026-07-04 (supersedes the READ mechanisms of both
`sparse-memory-tokens.md` [v1: broadcast temporal keys] and `sparse-memory-spatial-inject.md`
[v2: inject the stale set into later frames' inputs]). Status: DRAFT.

## The mechanism

Nothing is injected. Every frame keeps its memory slots with the normal learned-init input,
exactly like today. ONE change, in the temporal-attention mask for the memory channel:

> **Memory slots at time t may only attend to memory slots at times t' ≤ t with t' % n == 0.**

Consequences, by role:
- **Write slots** (t % n == 0): their memory K/V is the ONLY memory visible to the future — the
  actual carrier. Writing well is crucial exactly here.
- **Scratch slots** (all other t): their memory tokens FETCH from past write-sets (temporal) and
  distribute into the frame via spatial mixing — but their own output is never read by any later
  frame. They are per-frame retrieval heads, not storage.
- Latent/register/action/shortcut channels: completely unchanged (ordinary slot-wise causal).

**Role signal**: RoPE gives each query its relative distance to the visible write-keys (the phase
is in principle decodable — a write slot sees itself on the diagonal, a scratch slot doesn't) but
an explicit learned 2-entry **role embedding** (write vs scratch), added to the memory tokens'
input by `t % n`, makes the role direct rather than inferential. Cheap; include it. (Staleness
conditioning from v2 becomes unnecessary — RoPE relative distance to the write-keys IS the age.)

**Gradient**: flows back only through the every-nth write sets automatically — the mask means
only their K/V participate in later computation. No loss change.

**Invariant**: sliding window W ≥ 2n, so a write frame always has the previous write in-window
(a new set can always be written FROM the old one). n=8 fits W=16 (GridWorld) and W=32 (memmaze);
n=16 needs W=32.

**Cache**: non-write frames' memory rows are dead by construction — `rollout_step` can skip
committing them. The carried memory cache becomes ~W/n rows instead of W: sparsity shows up as a
genuinely smaller memory cache (on top of GQA's 4×).

## The end goal (the actual prize): long-reach memory attention

Because write-sets are sparse, letting the memory channel attend PAST the ordinary window is
cheap: with C=32 and n=8, giving memory reach over the last 32 write-sets covers **256 frames**
of history at the key-count cost of one ordinary window. Mechanically this is the
eviction-exempt memory bank (v1 draft's P4): memory rows at write positions are exempted from
window eviction (legal under absolute-index RoPE — cached K/V never re-rotate; eviction becomes
"slice ordinary rows, keep bank rows"). Reads then reach arbitrarily old writes DIRECTLY — no
relay squashing at all, which fully answers the scrutiny's "the relay still exists at period N"
objection to v2.

## Why v3 > v2 (spatial injection)

1. No frozen activation threading — reads go through cached K/V like every other read in the
   model; the spec's read-old/write-new invariant is preserved (v2 violated it).
2. Staleness is implicit and exact (RoPE relative distance), not a bolted-on age table.
3. The temporal memory channel is ACTUALLY freed/decluttered (v2 left stale copies in every
   frame's cache — the scrutiny's inconsistency finding #10); later frames physically cannot
   lean on intermediate rewrites.
4. Extends naturally to the long-reach bank (v2 had no story past the window).
5. Costs: one mask variant in `Attention.forward`'s temporal branch (slot-group-dependent mask,
   shape (N_slots, T, T_all), broadcasts over heads/batch — small surgery, but REAL surgery this
   time: unlike v2 this cannot be done purely via `--model-module` input manipulation; it needs
   an attention-mask hook. Prototype route: experiment subclass overriding the Attention forward
   for memory slots, or a mask-injection seam.)

## TBPTT cost analysis (Merlin's question: 32 dense sets vs 32 sparse sets)

Ground truth from `experiments/mem2mem/rollout.py`: each slide is ONE forward over the W-window;
cross-slide gradient flows only through the carried memory; `tbptt_frames` = frames of relay
depth retained before detach; memory cost = (number of retained slide graphs) × (per-slide
activation footprint) — this is what OOM'd memmaze mem2mem at bs6.

- **At a fixed tbptt_frames (env-time) budget: identical cost.** Dense and sparse retain the
  same slide graphs; the sparse writes are outputs of those same forwards — their "more
  independent construction" adds nothing, because the per-slide forward activations dominate and
  are shared by all writes in the slide. What changes is only the gradient PATH: tbptt_frames/n
  write-hops instead of tbptt_frames hops — fewer nonlinear re-encodings on the path (better
  conditioning), same env-time reach.
- **At a fixed NUMBER of graph-linked sets: sparse costs n× more.** 32 dense sets span 32
  frames (~2 retained slides at W=32); 32 sparse sets at n=8 span 256 frames (~16 retained
  slides) → ~8× the activation memory and backward compute. But that buys 8× the env-time reach;
  per env-time the two are identical.
- Practical rule: budget TBPTT in env-time (frames), as today. At the same budget sparse trains
  n× fewer writes, each with n× more context per hop.
- Long-reach reads BEYOND the TBPTT horizon (and at inference): the old writes' cached K/V act
  as constants — the READ path still trains; the old write's construction just gets no gradient
  from that read. Same truncation semantics as today.

## What carries over from the v2 scrutiny (still binding)

- The compounding PREMISE is still unmeasured (contradicted on GridWorld, no memmaze recall eval
  yet). v3's long-reach bank weakens the dependence on that premise — reach beyond the window is
  a capability win even if the dense relay never compounds — but the sparse-write half of the
  claim still needs the premise or a measured win.
- Falsifying probes stand: position_acc at maximal staleness (k mod n sawtooth), across-write
  decay at k = multiples of n (write-as-copy degenerate), role-embedding ablation.
- GridWorldV2 (action-conditioned, now built) is the discriminating testbed: under occlusion the
  fetch must combine a far-back write with the SUBSEQUENT ACTION STREAM — scratch slots must
  integrate actions since the last write, which is exactly the hard case v2's scrutiny said the
  design underplays.
