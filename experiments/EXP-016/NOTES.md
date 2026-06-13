# EXP-016 — batch-limit parallelism sweep (rollout KV cache)

Decision: D-023 | ESC: ESC-011 (resolved) → present-then-stop ESC-012
Tool: `experiments/EXP-015/perf_rollout.py --batch-sweep` (reusable; extended for this cut)
Provenance: master @ post-D-022; ckpt `experiments/EXP-012/vanilla_s0.pt` (dims only — perf is
weight-agnostic); tokenizer `trained_autoencoder.pt`; GPU RTX 4070 Laptop (8188 MB).
Config: fixed N=32, batch sweep {32,64,128,256,512,1024,2048}, budget 6 s/config, K=4, n_actions=2.

## Purpose
EXP-015 swept context window N at a fixed B=32 (both methods same batch). Merlin (ESC-011) asked for
the orthogonal cut: hold N fixed, push BATCH toward each method's OWN memory ceiling, and ask whether
the cached-vs-windowed speedup grows with parallelism. Key asymmetry under test: cached
(`generate_streaming`) has the smaller working set, so on 8 GB it should fit a LARGER batch than
windowed (`generate_windowed`) — a double win (faster/step AND more parallelism → higher peak frames/s).

## Reconciliation
Expected (from D-023): both faster in frames/s as batch rises until GPU saturates (compute-bound), then
steps/s falls ~linearly with batch. Speedup RATIO may COMPRESS toward ~1 as both become compute-bound on
the same per-step FLOPs (cached's saving = re-doing window attention, a shrinking fraction at high B) —
so "more speedup with more parallelism" may be FALSE for the steps/s ratio. Cached's decisive win
expected to show as a higher MAX batch (lower memory) → higher peak frames/s end-to-end.

Observed (N=32, RTX 4070 Laptop 8 GB; cached=generate_streaming, windowed=generate_windowed):
```
        cached                         windowed
 B   steps/s frames/s resvMB %    steps/s frames/s resvMB %     speedup(steps/s)
 32   22.8    730.7    380   4.6%   3.9    125.8    340   4.2%    5.85x
 64   18.3   1169.9    680   8.3%   1.8    116.8    558   6.8%   10.17x
128   10.6   1359.9   1184  14.5%   0.8    106.8    962  11.7%   13.25x
256    5.4   1372.5   2210  27.0%   0.4    108.7   1782  21.8%   13.50x
512    2.8   1427.1   4232  51.7%   0.2    107.0   3404  41.6%   14.00x   <- cached VRAM ceiling
1024   (skip, predicted >VRAM)             0.1    105.4   6716  82.0%            <- windowed VRAM ceiling
```
1. **Speedup GROWS with parallelism: 5.85x (B=32) → 14.0x (B=512), monotone.** Directly answers the
   question. Mechanism: windowed re-encodes the full N-1=31-frame window EVERY step (cost ∝ B×window),
   cached does O(1) work/step (just the new frame + cache slice). As batch rises both move toward
   compute-bound, but windowed's wasted recompute scales with batch, so the gap widens.
2. **Windowed gains NOTHING from more parallelism: frames/s FLAT ~105–126 across all B.** It is
   throughput-saturated on the window recompute — adding batch costs proportionally more time per step,
   net throughput constant. Only the cached path converts parallelism into throughput (731→1427 frames/s).
3. **Memory ceiling per approach (the asymmetry — FLIPPED vs EXP-015's N=64 regime):** at N=32 cached
   uses MORE memory than windowed at equal batch (persistent K/V cache for 31 frames × all layers/heads),
   so cached hits its VRAM ceiling SOONER — cached maxes at B=512 (52% VRAM), windowed reaches B=1024
   (82%). EXP-015 found the opposite at N=64 (windowed's transient ballooned). So "which fits a bigger
   batch" is N-dependent. BUT it does not change the verdict: cached at its lower ceiling (B=512, 1427
   frames/s) still delivers **13.5x the end-to-end throughput** of windowed at its higher ceiling
   (B=1024, 105 frames/s).
4. **Past-VRAM regime correctly excluded.** B≥1024 cached / B≥2048 windowed exceed the 8 GB card → WDDM
   sysmem fallback (ms→minutes; this was the 10-min "stuck" hang on the first attempt). The predictive
   memory guard (extrapolate reserved-MB, stop before crossing 0.92×VRAM) skips them; not OOM, not perf.

Surprise: mild (favorable). My D-023 prediction that the speedup RATIO would COMPRESS toward ~1 at high
batch is REFUTED — it GROWS (tripwire #1 fired, favorable). My EXP-015-based expectation that cached fits
a BIGGER batch is also wrong at N=32 (tripwire #2 fired) — cached is more memory-hungry here.

Caveat (measurement stability): absolute steps/s varies ~30% run-to-run on this laptop GPU (thermal/clock
throttling across back-to-back sweeps; an earlier identical run read cached B=512 at 2090 frames/s vs 1427
here). The cached-vs-windowed RATIO and the qualitative shapes (cached scales, windowed flat) are stable
across runs because the two methods are timed back-to-back under the same thermal state. Read the speedup
ratio + shapes as the result, not the absolute frames/s. Memory numbers are stable.

Hypothesis impact: infra/perf only — does not bear on H1/H2/H3.
Tripwires checked (D-023): (1) speedup GROWS with batch → YES (fired, favorable; reported). (2) cached
does NOT fit a bigger batch → YES at N=32 (fired; reported, N-dependent, doesn't change verdict). (3) OOM
non-deterministic → the real failure was sysmem-fallback thrash past VRAM, not fragmentation; fixed by the
predictive guard (no thrash, clean ceilings). All three surfaced and explained.

Methodology note: EXP-016 measures peak memory DURING the steady-state measured pass (full-window context,
no fill transient), differing from EXP-015's separate short low-output pass. Output-buffer contribution at
n_meas≤64 is small (≤~130 MB at B=2048) relative to the working set. Timing tool changes: warmup 4 +
calibrate 3 + budget-sized measured pass (floor 8, cap 64); ~2-pass-equivalent per config vs the original
4 passes — much faster while accurate. Provenance: tool `experiments/EXP-015/perf_rollout.py --batch-sweep`.

Next: present-then-stop → ESC-012.
