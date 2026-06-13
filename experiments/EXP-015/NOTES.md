# EXP-015 — rollout KV cache perf (D-022)

Basic perf tool: `generate_streaming` (cross-frame KV eviction cache) vs `generate_windowed` (matched
uncached twin — same semantics, so the delta is purely the cache). GPU RTX 4070 Laptop, venv python.
Real model dims (vanilla_s0 config+weights: E256, depth 8, 16 heads, K=4, n_actions=2), **batch B=32**
(training-relevant), context window N ∈ {8,16,32,64}, ~8 s/config. Tool: `perf_rollout.py` (CLI:
`--batch --windows --budget`). results.json + perf.png.

## Headline numbers (B=32)
```
method      N   steps/s  frames/s  ms/step  peakMB  resvMB
cached      8     28.3     904.9    35.4     113.5   170
windowed    8     21.2     679.6    47.1     122.1   190
cached     16     28.1     899.4    35.6     122.2   190
windowed   16     13.8     442.4    72.3     155.6   244
cached     32     28.3     906.5    35.3     153.8   210
windowed   32      7.2     229.4   139.5     232.6   672
cached     64     27.3     874.3    36.6     217.7   438
windowed   64      5.7     182.7   175.1     495.6  4708
```
Speedup (cached ÷ windowed steps/s): N=8 **1.33×**, N=16 **2.04×**, N=32 **3.93×**, N=64 **4.79×**.

## Reconciliation
Expected (D-022): cached flat ms/step as N grows (O(1) attention/step) while windowed grows with N
(re-encodes the whole window each step); speedup widens with context length; cached uses more memory but
bounded by the window.
Observed: **cached latency is flat at ~35 ms/step (≈28 steps/s ≈ 900 frames/s) across N=8→64**, while
windowed climbs 47→175 ms/step (21→5.7 steps/s). Speedup widens with context exactly as predicted
(1.3×→4.8×). The flatness is the whole point: with the persistent cache the per-step new-frame forward
attends to cached K/V (O(1) in N), so context length is ~free; without it each step rebuilds and
re-encodes the entire window.
Surprise: **mild, favorable — cached also wins MEMORY, the opposite of my prediction.** I expected the
persistent cache to cost more memory; instead cached uses LESS than windowed at every N, and the gap
explodes at N=64 (cached 218 MB alloc / 438 MB reserved vs windowed 496 MB alloc / **4708 MB reserved**).
Cause: windowed materializes a large transient `(B, W, n_tokens, E)` window tensor + full recompute
activations every step → big short-lived allocations → allocator churn (reserved ≫ allocated). The cache
holds compact per-layer K/V that grow only linearly with the window (W=N−1) and evict — so cached memory
is bounded by the window and far smaller in practice.
Tripwires (D-022): (1) cached NOT faster → did not fire (faster at all N). (2) cached memory blows up
unboundedly → did not fire (grows linearly with window, ≪ windowed; eviction frees as intended).
(3) unstable measurements → did not fire (cached ~28 steps/s repeatable across N).

## Where time goes (profiler, N=32, B=32, 20 steps; CUDA self-time)
Total CUDA work: cached **889 ms** vs windowed **3268 ms** (~3.7× more) for the same step count — the
cache removes the window-recompute. Split (compute = matmul/SDPA; memory = elementwise/cat/norm — a
PROXY for "time waiting for memory", not an HBM-stall counter): both ≈ **56% compute / ~40% memory**.
Top ops both paths: `addmm` (linear projections) > `sgemm` > `_fused_rms_norm` (a big memory-bound chunk:
79 ms cached / 367 ms windowed) > `bmm` (attention). So ~40% of GPU time is memory/elementwise-bound;
RMSNorm is the largest single memory-bound cost. Exact HBM memory-stall % would need Nsight Compute.

## Read
The cross-frame KV cache makes continuous rollout throughput **independent of context length** (~900
frames/s at B=32 regardless of N), where the uncached path degrades ~linearly with N (4.8× slower and
~10× more reserved memory at N=64). Cached wins on both speed and memory. The tool (`perf_rollout.py`)
is reusable at other batch/window settings.
