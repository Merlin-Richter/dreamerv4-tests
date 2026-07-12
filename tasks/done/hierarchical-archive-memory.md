# DESIGN v4: hierarchical fast memory + sparse segment archive

Proposed by Merlin and Codex, 2026-07-12. Status: **DONE**.

This document is an implementation-level design for the next Memory Maze dynamics experiment. It supersedes `sparse-memory-write-slots.md` and the earlier sparse-memory v1/v2 drafts.

This is deliberately an experiment design, not yet an authoritative `specs/` contract. The first
implementation belongs under `experiments/`; promote it into `src/` and update the source-backed specs
only after the mechanism survives the correctness gates and demonstrates useful long-horizon recall.

## 1. Executive summary

The existing dynamics model has **fast memory**: `n_memory` tokens on every frame. They attend
slot-wise through the ordinary temporal window and mix with all modalities in spatial blocks. Fast
memory is good at maintaining immediate state, but retaining an old observation for hundreds of
steps requires repeatedly rewriting it through the per-frame memory relay.

Memory Maze needs a second timescale. Correctly predicting what lies beyond a corner often depends
on a view from 50-100+ steps ago, and sometimes on an even older view from a compatible angle.

The proposed system adds a sparse **segment archive** without changing the main per-frame token
layout:

1. Keep the existing fast-memory tokens and the fixed local rollout window `W=32`.
2. Buffer the final written fast-memory states for a completed segment of `N` frames.
3. Detach those source memories from the dynamics graph during training.
4. Once per segment, run a tiny dedicated compressor. For each fast-memory slot, it compresses that
  slot's `N` written states into `R` archive embeddings.
5. Store each archive set with the absolute position of the segment's final frame.
6. In every temporal transformer block, fast-memory slot `m` has an additional grouped archive reader
  that may attend only to archive group `m`. Other token modalities retain the ordinary local window.
7. At inference, each temporal layer projects a newly written archive embedding into its own pre-
  rotated K/V and retains that sparse K/V far beyond local-window eviction.

With `n_memory=M=8`, `N=16`, and `R=1`, one eight-token archive set replaces 128 source memory tokens.
A 512-frame history contains only 32 archive sets (256 archive tokens total). The long-memory
attention cost is `M` times smaller than an all-to-all memory-to-archive reader.

The archive is **not recurrent**: archive set `j` summarizes only segment `j` and never reads earlier
archive sets while being written. Old segments remain separately addressable. Current fast memory
performs retrieval across the archive bank.

```text
committed frames
      |
      v
final written fast memories M_t --(buffer N frames, detach)--> tiny slot-wise compressor
                                                               |
                                                               v
                                                    canonical archive set S_j
                                                               |
                                      +------------------------+------------------------+
                                      |                        |                        |
                                      v                        v                        v
                              temporal layer 1 K/V     temporal layer 2 K/V     ... layer L K/V
                                      |                        |                        |
                                      +----------- grouped long-memory reads ----------+
                                                               |
                                                               v
                                                   current fast-memory slots
                                                               |
                                                    next spatial transformer block
                                                               |
                                                               v
                                                     latent/action prediction
```



## 2. Motivation and design choice



### 2.1 Why fast memory alone is not enough

The existing fast-memory carrier is a repeated read-old/write-new relay. Once the informative frame
leaves the latent window, old information can survive only insofar as every subsequent fast-memory
write preserves it. That is a suitable state tracker, but a poor direct retrieval mechanism for a
specific view hundreds of frames in the past:

- the information undergoes many nonlinear rewrites;
- training gradients through the dense relay are expensive and poorly conditioned at long horizons;
- the current KV cache exposes only the most recent local window;
- a static scene observation may not become useful until the agent returns to that place much later.

The archive gives current memory a direct attention path to old, separately stored segment summaries.
It shortens the read path without forcing one recurrent state to lossily absorb all history.

### 2.2 Why a separate compressor instead of sparse tokens in the main sequence

Putting archive tokens beside latents and fast-memory tokens only on boundary frames is possible, but
it creates avoidable complexity:

- training tensors become ragged or require masked dummy archive tokens on every frame;
- spatial attention can leak current latents directly into the archive unless carefully masked;
- archive queries need special cross-slot access to the previous `N` fast-memory states;
- adding tokens perturbs the spatial-attention softmax of a warm-started checkpoint;
- a main-transformer archive would need a write pass and then a second commit pass so its cached K/V
represents the written archive rather than its learned input token;
- its graph becomes entangled with a full dynamics-window graph.

Instead, the archive compressor is a small external module operating on already-written fast-memory
states. It emits the final canonical archive embedding directly. There is no archive denoising and no
archive commit pass through the dynamics transformer.

### 2.3 Why grouped reads instead of all-to-all reads

Temporal attention in the dynamics model is deliberately slot-wise. Memory slot `m` already follows a
persistent temporal channel and reads earlier states of slot `m`; spatial blocks give it access to all
within-frame modalities and other memory slots.

The archive preserves this structure. Fast-memory slot `m` reads only the archive tokens assigned to
group `m`. This is not limited to one archive token per memory slot: `R` archive subslots may be created
for each channel, so total archive-set size is

```text
n_archive = n_memory * archive_per_memory = M * R.
```

For `J` archive sets, all-to-all archive reading costs `O(J * R * M^2)` query-key pairs per head.
Grouped reading costs `O(J * R * M)`, an `M`-fold reduction (8x for the intended model).

## 3. Current model assumptions

The design assumes the existing Memory Maze dynamics checkpoint and its current contracts:

- embedding dimension `E=512`;
- depth `12`, arranged as repeated `[spatial, temporal, spatial]` triples;
- approximately 46 tokens per frame: action, 32 tokenizer latents, four registers, eight fast-memory
tokens, and one shortcut token;
- temporal attention is causal and slot-wise across time;
- spatial attention is full within a frame and mixes all token types;
- local rollout window `W=32`, represented during generation as `W-1` committed past entries plus the
current query frame;
- absolute-position RoPE in the cached path;
- K denoising reads followed by a dedicated near-clean commit pass;
- the written fast-memory state is injected into the commit pass;
- `commit=False` is a strictly read-only branch;
- the current mem2mem rollout slides by `W/2` and carries graph-attached fast memory across slides.

The fast-memory extension is a recurrent **information path**, not a tensor copied unchanged from one
frame to the next. Each frame begins with learned/fresh memory inputs (or explicitly injected memories
in rollout training). In temporal blocks, memory slot `m` reads cached K/V from earlier memory slot `m`;
in spatial blocks it exchanges information with the current action, tokenizer latents, registers, and
other memory slots. The final-layer memory slice is the newly written state `M_t`. At inference that
state is injected into the dedicated near-clean commit pass, whose per-layer K/V becomes the persistent
representation read by future frames. The archive is added alongside this mechanism rather than
replacing it.

The new archive path must preserve these contracts when disabled.

## 4. Goals, non-goals, and invariants



### 4.1 Goals

- Retain segment information for 256-512+ environment steps at modest K/V and attention cost.
- Give current fast memory a direct retrieval path to old segments.
- Preserve the existing fast-memory path for immediate state and action integration.
- Warm-start from the trained Memory Maze checkpoint without changing the main token layout.
- Train the archive compressor from long-future losses without backpropagating those losses into the
source fast-memory construction graph.
- Keep archive storage sparse and inference-compatible with absolute-position RoPE.
- Allow archive capacity to scale as an integer multiple of `n_memory`.



### 4.2 Non-goals for version one

- Archives do not summarize or rewrite previous archives.
- Archives do not directly receive latents, actions, registers, or shortcut tokens.
- Non-memory modalities do not attend directly to archive K/V.
- Archive groups do not communicate during compression in the first implementation.
- The first implementation does not promise unbounded memory: archive storage grows as `O(T/N)` until
an optional maximum-set cap is reached.
- The archive does not replace fast memory or the local temporal cache.
- No new auxiliary loss is required initially; the existing rollout objective remains the prediction
objective.



### 4.3 Hard invariants

1. **Segment locality:** archive set `j` is a function only of the final written fast memories from
  segment `j`.
2. **Source detachment:** archive losses do not backpropagate into those source fast memories in the
  initial design.
3. **Grouped retrieval:** fast-memory slot `m` can read only archive group `m`.
4. **Causality:** an archive becomes writable only after its final source frame has been committed and
  can affect only later frames.
5. **Read-only safety:** `commit=False` never changes the segment buffer or archive bank.
6. **Absolute positions:** archived K is rotated exactly once at the segment-end rollout position and
  never re-rotated.
7. **Independent local cache:** ordinary local-window eviction does not mutate archive K/V.
8. **Disabled equivalence:** with no eligible archives or archive-reader gate zero, the base dynamics
  path is unchanged apart from numerically inert branch plumbing.



## 5. Notation and initial configuration


| Symbol / field                 | Meaning                                   | Initial value                           |
| ------------------------------ | ----------------------------------------- | --------------------------------------- |
| `W`, `max_temporal_length`     | Local dynamics window                     | 32                                      |
| `N`, `archive_interval`        | Frames summarized per archive set         | 16; compare 32                          |
| `M`, `n_memory`                | Fast-memory slots per frame               | 8                                       |
| `R`, `archive_per_memory`      | Archive subslots per memory channel       | 1                                       |
| `A=M*R`                        | Total tokens in one archive set           | 8                                       |
| `E`, `embedding_dim`           | Dynamics/archive embedding width          | 512                                     |
| `archive_compressor_depth`     | Dedicated compressor blocks               | 1                                       |
| `archive_compressor_mlp_ratio` | Query-only compressor SwiGLU ratio        | 1-2 initially                           |
| `dense_tbptt_frames`           | Fast-memory gradient reach                | 64 minimum; profile 96                  |
| `archive_credit_frames`        | Future horizon contributing to compressor | 256 minimum; target 512                 |
| `clip_len`                     | Long rollout-training clip                | 512 target                              |
| `archive_max_sets`             | Optional inference/training bank cap      | unlimited within clip/episode initially |
| `archive_gate_init`            | Per-temporal-layer reader scale           | small, e.g. `1e-3`                      |


`archive_per_memory=0` disables the archive path and omits the compressor/readers. When the archive is
enabled, `N` must satisfy `1 <= N <= W`. Version one uses non-overlapping archive segments aligned to the
rollout's absolute position clock: segment `j` covers frames `[jN, (j+1)N)` and has end position
`b_j = (j+1)N - 1`. A sampled training clip defines its first frame as rollout position zero, so every
element in a training batch shares the same segment phase and `(J,)` archive-position vector.

## 6. Archive representation

One archive set has shape

```text
S_j: (B, M, R, E)
```

and the raw bank has conceptual shape

```text
S: (B, J, M, R, E)
archive_positions: (J,)  # b_j, the final frame in each source segment
```

For memory group `m`, the reader flattens only the set and subslot axes:

```text
S_group_m: (B, J*R, E)
```

It never flattens the memory-group axis into the key axis.

All `R` archive subslots in a set share the same absolute temporal position `b_j`. Their learned query
identity and content distinguish their within-set roles.

## 7. Dedicated archive compressor



### 7.1 Inputs

The source buffer contains the final written fast-memory state that was or will be injected into the
ordinary frame commit pass:

```text
memory_segment: (B, N, M, E)
```

During training:

```python
source = memory_segment.detach()
```

Detachment is deliberate. Long-future archive losses train the compressor and archive readers but do
not retain or update the dynamics graph that originally produced the segment memories. The ordinary
mem2mem rollout continues to train fast-memory construction with its own shorter TBPTT horizon.

### 7.2 Slot-wise batching

The compressor reshapes the source into independent memory channels:

```text
(B, N, M, E) -> (B*M, N, E)
```

It expands `R` learned archive queries for every channel:

```text
queries: (B*M, R, E)
```

The compressor weights are shared across channels, matching the main transformer's shared token-wise
projections. A learned memory-channel embedding `e_m` and archive-subslot embedding `e_r` identify the
destination lane:

```text
q[m, r] = archive_query[r] + memory_slot_embedding[m].
```

The source receives a segment-relative temporal position `0..N-1`; under a segment-local RoPE clock,
all archive queries use query position `N`, so their source-key phases encode age within the segment.
Learned relative position embeddings are also acceptable. The compressor does not need the rollout's
absolute position because the archive reader applies absolute time when the output is cached.

### 7.3 One restricted cross-attention block

The version-one compressor is decoder-style restricted cross-attention, not a standard self-attention
block over `N+R` tokens:

1. Archive queries are projected to Q.
2. Detached source memories are projected to K/V.
3. Each channel's `R` queries attend over only that channel's `N` source memories.
4. The attention result updates only the archive queries.
5. A query-only pre-norm SwiGLU MLP refines the archive outputs.
6. Source memories are not updated and do not pass through an MLP.

Pseudocode:

```python
src = source.reshape(B * M, N, E)
q = archive_queries(M, R).expand(B, M, R, E).reshape(B * M, R, E)

q = q + cross_attention(
    query=norm_q(q),
    key=norm_src(src),
    value=norm_src(src),
    source_positions=arange(N),
    query_position=N,
)
q = q + swiglu(norm_out(q))
archive = q.reshape(B, M, R, E)
```

The compressor has no shortcut/tau/d conditioning, no diffusion process, and no causal mask within a
completed segment. Compressor dropout should be zero in the initial implementation so deferred
recomputation is exact without replaying RNG state.

### 7.4 What the compressor does not read

The compressor receives neither old archives nor raw latents/actions. This makes compressor calls
explicitly non-recurrent, keeps credit assignment simple, and directly tests whether the trained fast-
memory representation is a sufficient source for long-term compression.

The source memories may themselves contain information retrieved from older archives because they were
written by the full dynamics model. Version one does not attempt to scrub that content. "Segment-local"
means that the compressor has no direct previous-archive input and summarizes only the `N` source states
presented to it; it does not imply statistical independence from all earlier history.

If this source proves insufficient, later experiments may add detached latents or actions, but that is
not part of version one.

## 8. Archive reader inside each temporal block



### 8.1 Separate grouped residual branch

Every temporal block gains a new archive-attention branch operating only on the fast-memory slice.
The existing local temporal attention remains unchanged.

Conceptually, for temporal block `l`:

```python
x = x + local_temporal_attention_l(local_norm_l(x))

mem = x[:, :, mem_start:mem_end]
mem = mem + archive_gate_l * archive_attention_l(
    archive_norm_l(mem), eligible_archive_bank
)
x[:, :, mem_start:mem_end] = mem

x = x + mlp_l(mlp_norm_l(x))
```

The archive branch sits after the ordinary local temporal attention and before the block MLP. The next
spatial block distributes retrieved archive information from fast memory into latents, actions,
registers, and shortcut tokens.

`archive_gate_l` is a learned scalar (or one scalar per query head) initialized to a small nonzero value
such as `1e-3`. This minimally perturbs the warm-started checkpoint while allowing reader and compressor
gradients on the first update. A zero-init gate is an acceptable exact-equivalence alternative, but it
delays internal archive-branch gradients until the gate moves away from zero.

### 8.2 Grouped attention rule

For every current time `t` and fast-memory slot `m`, the archive key set is

```text
{ S[j, m, r] : archive j is eligible, r in [0, R) }.
```

There is no attention edge from memory query `m` to archive group `m' != m`.

The implementation should batch `M` as a group/channel dimension, analogous to the existing slot-wise
temporal attention:

```text
Q: (query_heads, B, M, T_query, head_dim)
K/V: (kv_heads, B, M, J*R, head_dim)
```

GQA follows the base model: query heads sharing a K/V head broadcast over the group dimension without
materializing repeated archive K/V.

### 8.3 Layer-specific K/V from one canonical archive

The same raw archive embedding `S_j` is supplied to every temporal layer. Each layer owns its own
archive K/V projection:

```text
K_archive_l = rope(Wk_archive_l(S_j), position=b_j)
V_archive_l =      Wv_archive_l(S_j)
```

This is the same pattern as encoder-decoder cross-attention: one encoder memory is projected into a
different retrieval representation by every decoder layer. The archive embedding does not have to
traverse the dynamics transformer to obtain a valid K/V for each layer.

At inference these projections are computed once when the archive is written and cached. At training,
raw archive proxies are retained and reprojected in each independently-backwarded rollout block so one
K/V graph is not reused across multiple backward calls.

### 8.4 Absolute-time RoPE and eligibility

Archive K is rotated at the absolute segment-end position `b_j`; current memory Q is rotated at the
absolute query position `t`. Their dot product therefore represents relative age `t-b_j`. Cached archive
K is never re-indexed or re-rotated. Each archive reader uses matching Q/K head dimensions and the same
RoPE frequency convention on both sides; reusing the corresponding local temporal layer's frequencies is
the default.

To avoid exposing a summary before any of its source information has left local memory, and to preserve
exact first-window bulk prefill, archive `j` becomes eligible only when the earliest frame in its segment
begins to leave the local cache.

With `W-1` committed past frames retained, the first eligible query position is

```text
t_first = b_j + (W - N + 1)
min_archive_age = W - N + 1.
```

Examples:

- `W=32, N=16`: a segment ending at 15 becomes readable at query position 32.
- `W=32, N=32`: a segment ending at 31 becomes readable at query position 32.

The precise boundary must be unit-tested against the actual `max_ctx=W-1` cache semantics. There must
be no frame for which information has left local memory but its archive is still ineligible.

For a multi-frame training query, eligibility is a per-query mask, not one bank-wide selection based on
the final query in the window:

```text
archive_visible(t, j) = (t - b_j) >= min_archive_age.
```

The mask broadcasts over `R` and the memory-group dimension. If a query has no eligible archive, the
archive branch returns exact zero rather than applying softmax to an all-masked row.

## 9. KV-cache design



### 9.1 Local cache remains unchanged

The existing per-temporal-layer cache continues to store all ordinary token slots for the most recent
`W-1` committed frames and evicts by tail slice. Archive memory is not inserted into this rectangular
cache.

### 9.2 Separate sparse archive cache

Every temporal block has an archive cache:

```text
archive_k: (n_kv_heads, B, M, J, R, head_dim)
archive_v: (n_kv_heads, B, M, J, R, head_dim)
archive_positions: (J,)
```

The reader may flatten `(J, R)` at attention time. Position metadata is explicit because archive rows
are sparse and non-consecutive; unlike the local cache, their positions cannot be reconstructed from a
consecutive tail.

Local eviction never touches this cache. If `archive_max_sets` is configured, eviction removes complete
oldest sets consistently from K, V, and `archive_positions`; no remaining K/V is re-rotated.

### 9.3 Storage scaling

Archive storage grows as

```text
O(number_of_temporal_layers * B * n_kv_heads * M * R * T/N * head_dim).
```

The system reduces growth by `N` relative to storing a full memory set every frame, but it is not
asymptotically bounded. A fixed bank cap, hierarchical archive, or retrieval policy is a future extension.

## 10. Exact inference lifecycle

For generated frame at absolute position `t`:

1. **K denoising passes (read-only).** Each pass reads the ordinary local K/V and all eligible archive
  K/V. It mutates neither cache nor the raw segment buffer.
2. **Obtain written fast memory.** The last denoising pass returns predicted latent `z_t` and the final
  written fast-memory state `M_t`, exactly as the current rollout does.
3. **Ordinary near-clean commit pass.** Re-present `z_t` near-clean with `M_t` injected, append the
  frame's ordinary K/V, and evict the local cache to `W-1` committed frames.
4. **Append the committed memory source.** Append exactly the same `M_t` that was injected into the
  commit pass to `state.segment_memory`.
5. **Archive boundary.** When the segment buffer contains `N` consecutive committed states:
  - run the compressor once;
  - obtain `S_j: (B,M,R,E)`;
  - for every temporal block, project `S_j` to that reader's K/V;
  - rotate K at absolute segment-end position `b_j=t`;
  - append K/V and `b_j` to the archive bank;
  - clear the raw segment buffer.
6. **Advance position.** The archive can affect only later query positions and only after its eligibility
  age is reached.

There is **no archive commit pass through the dynamics transformer**. The compressor output is the final
stored state; per-layer projection and cache append are the archive commit operation.

### 10.1 Teacher-forced long prefill

`rollout_init` must build archives from true context with the same committed-memory semantics:

- the initial bulk window may produce written memories for frames `0..W-1` as today;
- partition those committed memories into complete aligned `N`-frame segments and compress them;
- cache their archive K/V immediately, with eligibility enforced by absolute query position;
- retain an incomplete trailing segment in `state.segment_memory`;
- teacher-forced context beyond `W` proceeds one committed frame at a time and may read newly eligible
archives.

Because `min_archive_age=W-N+1`, no archive written inside the first bulk window should have influenced
an earlier frame in that same bulk pass.

### 10.2 Read-only branches

`rollout_step(..., commit=False)` may read eligible archive K/V but must not:

- append ordinary K/V;
- append `M_t` to the segment buffer;
- run the compressor;
- append archive K/V;
- advance segment phase or absolute position.

This preserves the recall evaluator's branch-and-discard contract.

## 11. Training design



### 11.1 Fixed local window and long clips

Archive training uses a fixed local rollout window `W=32`; it does not sample smaller windows. A fixed
window gives archive boundaries and eligibility a stable meaning and matches the intended inference
regime.

Every training slide must pass its absolute rollout positions, e.g. `positions=arange(s, s+W)`. Do not
reset positions to `0..W-1` on each slide. A constant absolute offset leaves ordinary local RoPE
relations unchanged, while the archive reader requires the true distance between current queries and
sparse segment-end keys.

The target clip length is 512 frames. A 256-frame clip is an acceptable calibration stage but is not the
final Memory Maze horizon. The intended initial horizons are:

```text
dense fast-memory TBPTT: 64 frames minimum (32 frames beyond local eviction)
                         profile 96 if affordable
archive compressor credit: 256 minimum, target 512
```

The existing 50/50 rollout modes remain the initial prediction curriculum:

- **latent-present:** old context is near-clean and the new half follows shortcut noising;
- **memory-only:** relevant latents are pure noise, so carried memory is the scene carrier.

Later phases add archive-required combinations (section 11.5).

### 11.2 Which memory states enter a segment

Each absolute frame contributes once. With half-window rollout slides:

- the initialization window contributes its written memory states once;
- every later slide appends only the newly constructed half-window memories;
- overlapping old-half memories are context and must not be appended again.

Archive segmentation is based on absolute rollout/clip position, not slide-relative position.

### 11.3 Detached compressor sources

At every completed segment:

```python
source_j = stack(final_written_memories_for_segment_j).detach()
```

Long-future archive loss therefore trains:

- compressor parameters;
- archive queries and slot/subslot embeddings;
- per-temporal-layer archive Q/K/V/output projections;
- archive gates;
- downstream dynamics parameters consuming retrieved information.

It does not train the source fast-memory writer through the archive path. The ordinary dense mem2mem
path remains responsible for learning useful fast-memory representations.

### 11.4 Long-horizon backward without retaining every dynamics graph

The current rollout implementation sums all slide losses and calls backward once. Detaching a relay
limits gradient connectivity but does **not** free each slide's activations while its loss remains in the
sum. A 512-frame clip therefore requires blockwise backward, not merely a larger `tbptt_frames` value.

Use archive proxies and deferred compressor VJPs:

#### Archive creation

```python
source_j = memory_segment.detach()
with torch.no_grad():
    archive_value_j = compressor(source_j)
archive_proxy_j = archive_value_j.detach().requires_grad_()
```

Store `source_j`, `archive_proxy_j`, and `b_j`. Raw proxies, not persistent differentiable K/V, form the
training archive bank.

#### Dense TBPTT blocks

1. Run 64-96 frames of the ordinary differentiable rollout.
2. For each forward in the block, project all eligible raw archive proxies into the current reader
  layer's K/V. This projection is cheap and creates a fresh graph for this backward block.
3. Scale each block loss by the total planned rollout-loss normalization so blockwise backward matches
  the magnitude of the former single backward.
4. Backward the block. Parameter gradients accumulate, and every referenced `archive_proxy_j.grad`
  accumulates `dL/dS_j` from that future block.
5. Detach the dense fast-memory carrier at the configured dense TBPTT boundary and free the block graph.
6. Do not call `optimizer.step()` yet.



#### Deferred compressor backward

After all blocks in the clip:

```python
archive_real = [compressor(source_j) for source_j in sources]
grad_outputs = [proxy_j.grad for proxy_j in archive_proxies]
torch.autograd.backward(archive_real, grad_outputs)
optimizer.step()
```

This is exact chain-rule credit into the compressor provided that:

- compressor parameters are unchanged during the clip;
- compressor dropout is disabled or RNG is exactly replayed;
- `archive_value_j` and the recomputed `archive_real_j` use identical source values and parameters;
- loss normalization is independent of backward partitioning.

The dynamics and archive-reader gradients have already accumulated from the block backwards. The final
VJP adds compressor gradients without retaining the source dynamics graph or all future rollout graphs.

If a proxy is never used or has `grad is None`, its deferred gradient is zero and it may be skipped.

### 11.5 Planned training modes

Version one may start with the existing two modes to validate plumbing and cache behavior. The archive
can otherwise be ignored because dense fast memory remains a viable relay, so subsequent training must
include archive-forcing modes:

1. **Normal / latent-present:** preserves rollout quality and local dynamics.
2. **Fast-memory-only:** current 50% full-noise mode; validates the existing dense memory carrier.
3. **Fast-memory-hiding:** after at least one archive is eligible, hide/temporarily remove the memory tokens on the clean half while keeping (a) or hiding (b) latents, forcing (a)only memory-requiring-information or (b)all information to pass through the archive.
4. **Mixed removal (optional):** independently sample latent-present/removed, fast-memory-present/removed, and
  archive-present/removed subject to at least one valid information path. Improves Robustness.

The exact mixture is an experimental axis. Do not introduce every mode before the base archive path
passes deterministic correctness and gradient tests.

## 12. Expected graph and compute cost

For the intended model:

```text
dynamics window: 32 frames * ~46 tokens/frame * 12 layers = 17,664 token-layers
compressor:       8 lanes * (16 source + 1 query) * 1 layer = 136 token-layers
```

The compressor is approximately 0.77% of one dynamics-window graph by token-layer count. Its grouped
attention matrix is smaller still:

```text
8 lanes * 16 heads * 1 query * 16 keys = 2,048 score elements
```

versus roughly 12 million spatial+temporal score elements across one 12-layer dynamics window. K/V
projections over the 128 detached source tokens, not the attention matrix, should dominate compressor
activation memory.

Conservatively estimating one compressor graph as 0.5-1% of one dynamics-window graph:

- `N=16`, 512-frame clip: 32 compressor graphs ~= 0.16-0.32 dynamics-window graphs;
- `N=32`, 512-frame clip: 16 compressor graphs ~= 0.08-0.16 dynamics-window graphs;
- dense TBPTT 64 with a 16-frame slide retains roughly four dynamics-window graphs.

Thus all 512 frames of compressor credit should cost only a few percent of the live dense-TBPTT graph.
Measure this with CUDA peak-allocation calibration before the full run; do not treat the estimate as a
substitute for profiling.

Detached source storage across a 512-frame clip is also small:

```text
512 * 8 * 512 BF16 elements ~= 4 MiB per sample.
```

One full-width `E=512` cross-attention + SwiGLU block is activation-cheap but still has a few million
parameters. Compressor width, GQA, and MLP ratio are parameter/optimizer-memory knobs distinct from the
horizon-dependent graph cost. Avoid adding an internal-width bottleneck until the full-width mechanism
has established whether archive compression works.

## 13. Checkpoint and warm-start behavior

The first archive model should load the trained Memory Maze mem2mem checkpoint:

- preserve every existing dynamics parameter;
- add the compressor, archive reader projections, archive norms, gates, and archive slot embeddings;
- do not change the main token sequence or existing local K/V shapes;
- serialize all archive configuration fields in the checkpoint;
- reject an archive checkpoint if `n_memory`, `embedding_dim`, `archive_interval`, or
`archive_per_memory` is incompatible with the constructed model.

Use a small archive-reader gate initialization so early predictions remain close to the warm-started
model. A gate-zero forward-equivalence test must exist even if production initialization is `1e-3`.

## 14. Suggested experiment-local implementation structure

The exact names may change, but the prototype should separate responsibilities:

```text
experiments/<archive-experiment>/
  model.py
    ArchiveCompressor
    GroupedArchiveAttention
    DynamicsModelArchive
  rollout.py
    long archive rollout loss
    blockwise dense TBPTT
    archive proxies + deferred compressor VJP
  train.py
    warm-start/load/save
    long-clip optimizer loop
  smoke.py
    deterministic architecture/cache/gradient gates
```

`DynamicsModelArchive` extends the existing dynamics model and adds archive readers only to temporal
blocks. Its inference rollout state minimally contains:

```python
state = {
    # existing fields
    "cache": local_cache,
    "next_pos": ...,
    "max_ctx": W - 1,
    ...,

    # archive fields
    "archive_cache": per_temporal_layer_archive_kv,
    "archive_positions": tensor_of_segment_end_positions,
    "segment_memory": trailing_committed_memory_states,
    "segment_start": absolute_position_of_first_trailing_state,
}
```

Training uses raw archive proxies rather than persistent projected archive K/V. Inference uses projected
K/V and may discard raw archive embeddings after all temporal-layer caches are populated.

## 15. Correctness gates

All gates below are required before allocating a long Memory Maze run.

### 15.1 Compressor and grouping

- Shape test for arbitrary `B`, `N`, `M`, and `R`.
- Perturb source memory channel `m`; before any future archive-spatial extension, only archive group `m`
may change.
- Perturb archive group `m`; only fast-memory query group `m` may receive a direct archive-attention
change.
- Compressor output must be invariant to old archive-bank contents.
- Compressor source tensors must have no gradient after an archive-only future loss.
- Compressor parameters must receive nonzero gradient from a loss more than one local window later.



### 15.2 Causality and age

- An archive cannot affect its own source frames or its boundary frame.
- Eligibility begins exactly when the first source frame leaves the local cache.
- No gap exists between local visibility and archive eligibility.
- Archive attention sees no future set.
- All `R` tokens of one set use exactly the segment-end absolute position.



### 15.3 Cache equivalence

- Cached incremental archive reads must match an uncached reference that explicitly projects all raw
eligible archives at their absolute positions.
- Local-cache eviction must leave archive K/V byte-identical.
- Archive-set eviction under `archive_max_sets` must preserve positions and not re-rotate retained K.
- GQA grouped archive attention must match a materialized repeated-K/V reference.



### 15.4 Rollout lifecycle

- `commit=False` leaves local K/V, archive K/V, archive positions, segment buffer, segment phase, and
`next_pos` unchanged.
- Boundary commit creates exactly one archive set.
- Non-boundary commit creates none.
- Long teacher-forced prefill and stepwise prefill agree once the same eligibility policy is applied.
- The exact memory injected into the ordinary commit is the memory appended to the segment source buffer.



### 15.5 Training/backward

- One-shot backward on a short clip must match blockwise-backward + deferred-VJP gradients within
floating-point tolerance when dropout is disabled.
- Loss scaling must be invariant to the number of TBPTT blocks.
- Archive proxy gradients must accumulate across multiple future blocks.
- Optimizer parameters must not change before deferred compressor backward completes.
- Peak CUDA allocation must remain bounded as dense TBPTT stays fixed while clip length grows; only the
small raw source/proxy storage may grow linearly.



### 15.6 Disabled regression

- With archive gates forced to zero, base forward/loss/generate outputs match the warm-start model within
the expected floating-point tolerance.
- `archive_per_memory=0` or archive disabled must preserve the existing checkpoint and rollout API
behavior.



## 16. Evaluation and falsification

Training loss alone cannot establish long-term memory. The result must compare the same checkpoint under
controlled archive interventions.

Required evaluations:

1. **Archive-on vs archive-zeroed:** identical long prefill, actions, noise seeds, and rollout; only
  archive values differ.
2. **Age bands:** allow archives older than 32, 64, 128, 256, or 512 frames to determine which history
  actually affects prediction.
3. **Bank truncation:** cap at different numbers of archive sets and plot performance versus effective
  historical reach.
4. **Lane permutation:** permute archive groups across memory slots. A meaningful grouped reader should
  degrade.
5. **Segment permutation:** preserve archive content but scramble segment timestamps/order to test
  whether temporal addressing matters.
6. **Corner/revisit cases:** prioritize Memory Maze sequences where the correct future view depends on a
  location last observed 50-100+ frames ago and from a compatible angle.
7. **Local-quality regression:** verify that short-horizon action response and visual stability remain at
  least as good as the warm-start checkpoint.

Qualitative rollout sheets are useful but insufficient. A quantitative Memory Maze revisit/recall
instrument is eventually required to support the memory claim.

The most important falsifier is simple: if zeroing eligible archives does not change long-horizon
prediction on cases whose informative observation has left dense TBPTT reach, the model has ignored the
archive regardless of its training loss.

## 17. Risks and mitigations

### Archive is ignored

The dense fast-memory relay remains a shortcut. Add archive-required training modes after the plumbing
is validated, and always measure same-checkpoint archive ablations.

### Slot-wise source is too isolated

Spatial layers make every final fast-memory slot globally informed, but stable lane specialization is not
guaranteed. The first extension is an archive-only spatial organizer (section 18), not an immediate
all-to-all long-memory reader.

### Compression capacity is too small

Increase `R` so `n_archive=R*M`, or add compressor depth. Test `R in {1,2,3}` only after `R=1` establishes
the mechanism.

### Raw RoPE distances extrapolate poorly

Train on the intended 256-512 frame distances, log attention versus archive age, and compare a separate
archive-time RoPE clock if raw absolute-time generalization fails.

### Attention dilutes over many archive sets

Measure attention entropy and performance versus bank size. Future options are a capped recent bank,
content retrieval, or hierarchical archive levels.

### Long clips reduce optimizer-step frequency

Blockwise backward frees activation memory but parameters cannot step until deferred compressor VJPs are
complete. Use gradient normalization based on predicted loss terms and compare examples/tokens per
optimizer update against the baseline.

### Training/inference representation mismatch

Training projects raw archive proxies on demand; inference caches projected K/V. The cached-vs-uncached
equivalence gate is load-bearing.

## 18. Deferred ablations and extensions

These are explicitly outside version one but should remain architecturally possible.

### 18.1 Archive-spatial organizer

A natural deeper compressor mirrors the model's separate temporal/spatial organization:

1. slot-wise segment cross-attention (`N` states -> `R` archive queries per channel);
2. archive-only full self-attention over all `M*R` archive outputs;
3. optional slot-wise refinement cross-attention or MLP.

The archive-spatial layer lets output lanes coordinate and reorganize information while retaining grouped
read incentives: final archive group `m` is still directly readable only by fast-memory slot `m`. With
`M=8,R=1`, its self-attention covers only eight tokens once per segment and is computationally negligible.

### 18.2 Deeper compressor

Compare one block with approximately three blocks, including the archive-spatial organizer. Compressor
depth increases fixed parameter memory more than it increases activation cost.

### 18.3 More archive subslots

Test `R=2` and `R=3`. Source K/V can be shared across the `R` queries within each channel; query/MLP and
bank storage scale with `R`.

### 18.4 All-to-all write, grouped read

Allow each archive output to read all `N*M` source memories during compression while retaining grouped
future reads. Write cost occurs only every `N` frames, so this is affordable, but it weakens the clean
per-channel compression hypothesis.

### 18.5 One-softmax local+archive attention

Instead of a separate gated residual, concatenate grouped archive K/V with local memory K/V inside one
softmax. This requires splitting memory slots from ordinary modalities because only memory has the long
key axis. Consider only after the separate branch proves useful.

### 18.6 Source-side archive gradients

Remove `stopgrad` or checkpoint/recompute the source dynamics window so long-future archive losses train
the fast-memory writer. This is much more expensive and should be justified by evidence that detached
fast-memory features are insufficient.

### 18.7 Bounded or hierarchical archives

Introduce fixed bank capacity, archive-of-archives, learned retrieval, or geometric timescales if episode
length makes `O(T/N)` storage or attention too large.

## 19. Version-one acceptance criteria

The mechanism is ready for a full Memory Maze run only when:

1. all correctness gates in section 15 pass;
2. a 512-frame synthetic/real-data training smoke runs with bounded peak memory;
3. blockwise gradients match the short one-shot reference;
4. cached and uncached archive reads agree;
5. warm-start behavior is preserved with the archive gate disabled;
6. archive parameters receive nonzero gradient from losses beyond the local and dense-TBPTT horizons;
7. the archive-required mode can overfit a small controlled batch;
8. inference cost and cache growth match the derived `O(T/N)` scaling;
9. a same-checkpoint archive ablation produces a measurable effect on a controlled long-horizon memory
  case before expensive training is launched.



## 20. Final design statement

Version one is a two-tier memory system:

```text
FAST MEMORY
  eight tokens every frame
  ordinary slot-wise temporal attention
  fixed 32-frame local window
  dense TBPTT over at least 64 frames

SEGMENT ARCHIVE
  one independent archive set every 16 (or 32) committed frames
  one archive token per fast-memory channel initially
  tiny one-block slot-wise compressor over detached written memories
  same canonical archive embeddings projected into every temporal layer's own K/V
  grouped slot-wise long-memory reads
  sparse absolute-position RoPE cache retained for 256-512+ frames
```

Fast memory integrates the immediate observation/action stream. The archive preserves separately
addressable evidence from old segments. The main dynamics transformer remains responsible for deciding
which old evidence matters now; the compressor's job is only to turn each completed fast-memory segment
into a cheap, durable retrieval object.

**Result (2026-07-12):** Implemented under `experiments/hierarchical-archive-memory/`; all deterministic
archive gates and existing dynamics/cache regressions pass, 512-frame production-shape CUDA training is
memory-bounded at 3.66 GiB (batch 1), and the controlled archive-required overfit/ablation passes.
