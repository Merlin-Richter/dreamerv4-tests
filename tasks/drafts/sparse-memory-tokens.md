# DESIGN: memory tokens only every Nth frame (sparse memory), reads for all frames

Requested by Merlin 2026-07-04 ("invent a plan for making memory tokens not exist for every frame
but only for every nth frame ... while still making memory inform all frames and be informed by all
past frames"). Status: DRAFT for Merlin — promoting any phase to backlog is his call.

## 0. The architectural fact that shapes everything

Temporal attention is strictly SLOT-WISE: `Attention` permutes to `(heads, B, N_slots, T, d)` and
attends over T with the slot axis batched (dynamics_model.py:141) — slot j at frame t attends ONLY
to slot j at frames ≤ t. So today, memory informs frame t exclusively through frame t's OWN memory
slots (relayed along the memory channel, mixed into latents by that frame's spatial attention).
Remove the slots from a frame and, with no other change, that frame is cut off from memory entirely.

The second fact: every frame's latent/register slots already see ALL in-window past frames through
their own temporal channels. Memory's unique job is carrying state PAST the window (and across
rollout commits). This makes sparse memory principled: a write every N frames loses nothing that
in-window attention doesn't already provide — **provided every frame lands in the in-window
receptive field of at least one write** (invariant: N ≤ W−1; N ≤ W/2 keeps ≥1 previous write
in-window at each write, which the relay wants anyway).

## 1. What sparsity buys (and what it doesn't)

- **Relay depth ÷ N.** The carried rollout writes memory every committed frame; the relay gradient
  explodes ~2–3×/hop at init (GridWorld probes, `mem2mem-rollout-noff9-fair`). A horizon of H frames
  becomes H/N hops: 128-frame clip @ N=8 → 16 hops instead of 128. Fewer multiplications through the
  write path = tamer relay dynamics, and each hop is trained to summarize a SEGMENT, not a tick —
  memory becomes a slower-timescale summarization channel (the conceptual win).
- **Compute/KV, honestly modest.** Memmaze frame = 1 action + 32 latents + 4 registers + 8 memory +
  1 shortcut = 46 slots. Removing memory from (N−1)/N frames: spatial attention ≈ (38/46)² ≈ 0.68,
  temporal ≈ 0.83 on those frames, plus 8·(W/N) fewer cached temporal K/V rows per window. ~20–30%
  attention savings at N=8 — real but secondary.
- **Not** a free window extension: information between writes still only reaches memory while those
  frames are in-window (§0 invariant).

## 2. Design A — presence sparsity (the actual ask)

Memory tokens EXIST only at "memory frames" t ≡ 0 (mod N). Three mechanisms:

**(a) READ — memory informs all frames — temporal broadcast keys.**
In temporal layers, every slot's key set becomes `k_all = [broadcast memory K/V ; own-channel K/V]`:
the memory slots' cached (rotated) K/V from all past memory frames are prepended as EXTRA keys for
EVERY query slot (expand across the slot batch axis — cheap, it's a view). Causal mask by absolute
frame index (the RoPE machinery already takes explicit `positions` and never re-rotates cached K/V,
so sparse positions just work). Memory becomes the one BROADCAST channel in an otherwise slot-wise
temporal attention; each frame's denoising reads memory directly instead of via its own memory slots.

**(b) WRITE — memory informed by all past frames — nothing new needed.**
The write at memory frame t is the commit-pass output of its memory slots, which see (i) frame t's
tokens spatially, (ii) through those, the full in-window past (slot-wise temporal), (iii) previous
writes via (a). Induction over writes covers ALL history: segment content enters at the first write
whose window contains it (guaranteed by the §0 invariant), then relays write-to-write. Optionally
let memory queries ALSO cross-attend to past latent/register channels for a more direct write path —
extension, not needed for correctness.

**(c) Layout/rollout cadence.** Memory frames carry `[action|latents|registers|memory|shortcut]`,
others drop the memory block. `rollout_step` writes+commits memory K/V only when
`pos % N == 0`; other commits append only the ordinary slots. Eviction unchanged (pure slice; the
memory rows just sit at sparse positions).

Implementation obstacles (the honest list):
- **Ragged frames.** Rectangular `(B,T,N,·)` tensors assume identical slots/frame. Training forward:
  keep memory slots in the tensor for ALL frames but mask them out of spatial+temporal attention (and
  the loss) on non-memory frames — rectangular, wastes 8 dead slots/frame in the MLPs (fine for a
  first version); or split the forward into ordinary-token and memory-token passes (the clean but
  invasive version). Inference commits are per-frame anyway — no raggedness there.
- **Mask surgery** in exactly one place (`Attention.forward`, temporal branch): the broadcast-key
  concat + its causal mask. Everything else (RoPE, cache, eviction, soft-cap, logit scale) is reused.
- **N=1 must reduce to (almost) the current model** — the regression anchor. Not bit-exact (reads go
  through broadcast keys instead of own-slot relay) — pin the equivalence claim precisely in tests:
  same information flow, causality probe green (no slot reads a future or gapped-out memory).

## 3. Design B — write sparsity (cheap prototype, zero architecture change)

Keep memory slots in every frame's layout; change only WHAT is injected: fresh memory is WRITTEN
every Nth commit, and the intermediate N−1 commits re-inject the LAST WRITTEN tokens (piecewise-
constant memory channel). Same for the training rollout's injection cadence. No mask or layout work
— expressible as an experiment-local `DynamicsModel` subclass via the existing `--model-module` seam.
Buys the relay-depth/timescale effect (hops ÷ N) with none of the compute win. **This is the
discriminating experiment**: if GridWorld recall stays flat at N=4/8, the timescale hypothesis holds
and Design A is worth the surgery; if recall collapses with N, presence sparsity would too.

## 4. Training signals under sparsity

- **mem2mem rollout (the winner signal): unchanged in structure.** Slide windows over a long clip,
  carry written memory across slides, 50% full-noise mode forces memory to carry the scene — now the
  read path (a) is what the noise mode trains. Injection points become the memory frames.
- **FF9 is awkward under sparsity** (its per-frame "inject the written memory at t, reconstruct
  t+1..t+k from memory alone" assumes a write at every t). Nearest-past-write injection is the
  analogue, but better: **the no-FF9 arms gate this.** GridWorld already showed FF9 unnecessary
  (`mem2mem-rollout-noff9-fair`); memmaze no-FF9 is running (415143). If it holds, sparse memory
  trains on the rollout signal alone and FF9 never needs a sparse variant.
- **Evals unchanged.** `generate()` API identical; GridWorld recall + sheets, memmaze sheets (and the
  future memmaze recall/probe) run as-is. Ablation axis: N ∈ {1, 2, 4, 8, 16} vs vanilla.

## 5. Phased plan (each phase a separate task; GridWorld first — cheap + recall eval exists)

1. **P0 (running):** memmaze no-FF9 arm 415143 — gates the FF9 question for free.
2. **P1 — Design B prototype on GridWorld** (~1 day + one ferranti job/N): experiment-local subclass
   (`experiments/EXP-NNN/model.py`), N ∈ {4, 8}, winner config minus FF9; recall w8 max_k64 vs the
   N=1 winner (0.99 flat). Decides whether the idea carries ANY retention cost before real surgery.
3. **P2 — Design A in src/ + spec** (the real change; needs Merlin's sign-off on the spec diff —
   architecture decision, outside the campaign delegation): broadcast-key temporal attention,
   masked-dead-slot forward, sparse commit cadence, `n_memory_every: int = 1` config field (1 =
   exactly today's per-frame behavior). Gate tests: causality probe (no future/gap leak), N=1
   regression vs current checkpoints, carried-vs-uncached equivalence rerun (V-cache-equiv style).
   GridWorld recall ablation N ∈ {1, 4, 8, 16}.
4. **P3 — memmaze** at the winning N (50ep arm, same 512/12/16 W32 recipe) → 4-way compare.
5. **P4 — extensions, if P2/P3 win:** exempt memory K/V from window eviction (a long-lived sparse
   memory bank — reads reach ARBITRARILY old writes directly, no relay squashing; legal under the
   absolute-index RoPE design since cached K/V never re-rotate; eviction becomes a gather, not a
   slice); hierarchical N (fast memory every frame, slow every 16).

## 6. Risks / open questions
- Under-trained read path for frames far from a write (mitigated by the full-noise rollout mode, but
  check recall as a function of `k mod N` — periodicity there = the read path leaning on writes).
- Relay explosion may CONCENTRATE (fewer, bigger hops): the per-hop factor could grow with segment
  length; `--relay-grad-clip` exists if so.
- Broadcast keys make memory globally readable — attention sinks? (memory rows visible to every
  query; watch attention entropy on memory rows in P2).
- Interaction with `context_signal` commits and the read-only branch (`commit=False`): the branch at
  a would-be memory frame must NOT write — read-only contract stays absolute (recall depends on it).
