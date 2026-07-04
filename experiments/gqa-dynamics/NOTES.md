# GQA dynamics — grouped-query attention, 4× smaller carrying KV cache

**Date:** 2026-07-04. **Ordered by Merlin:** "implement grouped query attention to cut the KV
footprint by 4x; train on gridworld as a test."

## Design (`model.py`, via `--model-module` — no spec-backed file touched)

- `GQAAttention(Attention)`: 16 query heads share **4 KV heads** (groups=4). Separate `q_proj` /
  `kv_proj` replace the fused qkv. Everything else faithful to the base attention IN ORDER:
  QK-RMSNorm → RoPE (fixed table / absolute-`positions`) → cache prepend/commit → per-QUERY-head
  learnable logit scale (grouped view) → tanh soft-cap → block-causal mask → softmax → dropout.
  The grouped matmul broadcasts K/V over the group axis (`unsqueeze(1)`) — repeated K/V is never
  materialized. Cache dicts keep the `{'k','v'}` layout with kv-head-leading shapes, so
  `rollout_init`/`rollout_step`/eviction (pure time-axis slices) work unchanged.
- `DynamicsModelGQA`: swaps every block's attention for GQA + carries the **τ0-anchor objective**
  (p=0.5, copied inline from vanilla-honest-baseline) so the comparison target is
  `dynamics_vanilla_tau0.pt` — identical objective/config/data, **GQA is the single varying
  factor**. Params 6.86M vs 7.75M (smaller kv projections).
- NOTE: GQA checkpoints need `DynamicsModelGQA` to load (different param shapes) — eval scripts
  here load with the class and call `recall()` / probe machinery directly.

## Pre-launch verification (`smoke.py`, local, all PASS)

1. **Causality**: perturbing frame t leaves all outputs < t bit-identical.
2. **Cache equivalence**: committed incremental forwards == one-shot uncached forward within the
   window (maxdiff 1.85e-06, fp32 — the regime V-cache-equiv proved exact for base).
3. **Footprint**: rollout_init cache = 0.25 MB vs base 0.98 MB — **ratio exactly 4.00×**.
4. Loss backward finite; 3-frame generate OK. Plus a 2-epoch local trainer smoke (loss ↓).

## Pre-registered expectations

- val/loss and teacher-forced probe ≈ `dynamics_vanilla_tau0.pt` (GridWorld is small; 4 KV heads
  at head_dim 16 should carry it). Meaningful degradation (probe < 0.9, val ≫ 0.0010) would mean
  the KV bottleneck bites even here — relevant before trying GQA on memmaze (512-dim, W=32),
  where the cache actually matters.
- Footprint claim is already proven mechanically (4.00× by construction + measurement).

## Provenance

- ferranti **job 415214** @ SHA `7ae5d72` (50ep bs256 seed0, 5x data, --hours 4), submitted 2026-07-04 19:51. -> `checkpoints/gridworld/dynamics_gqa_tau0.pt`, W&B `gw-dyn-gqa-tau0`.

## Eval plan (when it lands)

Pull ckpt → `eval.py` (loads DynamicsModelGQA): teacher-forced probe t=2/4/8/15 + free-run,
recall w8 max_k32, val-loss comparison vs `dynamics_vanilla_tau0.pt`; cache-bytes table. Verdict →
EXPERIMENTS.md.

## RESULT (2026-07-04) — PARITY at 4.00x smaller cache

Job 415214 rc=0 (17 min). Head-to-head vs `dynamics_vanilla_tau0.pt` (MHA, same objective/recipe):

| metric | MHA tau0 | **GQA tau0** |
|---|---|---|
| val/loss (default sampler, ep50) | 0.001032 | 0.001058 |
| teacher-forced pos_acc t=2/4/8/15 | 0.84/0.98/1.00/1.00 | 0.86/1.00/1.00/1.00 |
| free-run j=1..12 | 0.98-1.00 | 1.00 flat |
| recall w8 (in-window k<=6 / past) | 1.0 / chance | 1.0/1.0/0.97 / chance |
| rollout KV cache (full window, B=1) | 921.6 KB | **230.4 KB (4.00x)** |
| params | 7.75M | 6.86M |

GQA is behaviorally indistinguishable from full MHA on GridWorld while cutting the carrying
cache exactly 4x (+11% fewer params, smaller attention activations). Green light for trying GQA
where the cache actually hurts: memmaze (512-dim, W=32, 32 latents/frame) and any
eviction-exempt memory-bank design where cache size is the binding constraint (see
tasks/drafts/sparse-memory-tokens.md). Graduation to src/+spec = Merlin's call
(config knob n_kv_heads or gqa_groups, default = n_heads i.e. MHA).
