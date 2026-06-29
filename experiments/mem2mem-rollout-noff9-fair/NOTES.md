# Fair no-FF9 ablation — does the noise-mode rollout flow loss ALONE train memory?

Clean re-run of `experiments/mem2mem-rollout-noff9/` (job 411270, "FF9 NECESSARY"), with its confounds
removed. That run was NOT a clean ablation — it rode the same bootstrap+curriculum+instability+36ep stack
that we showed (in `experiments/mem2mem-rollout-boot-fair/`) produces spurious negatives. See
`tasks/in-progress/fair-noff9-ablation.md` for the full rationale.

## The mechanistic question (and why the old result looked off)
The rollout has two memory-pressure sources: (a) the explicit FF9 sufficiency term, and (b) the 50%
**full-noise** mode, where every latent in the window is pure noise so the new-half flow loss can ONLY be
satisfied by reconstructing the scene from carried memory. 411270 removed (a), kept (b), and got chance
recall. Conceptually (b) alone *should* train memory — IF the gradient flows back through the memory relay
to where the memory was constructed.

**It does.** Verified two ways (NOT a severed-gradient bug):
- `experiments/mem2mem/test_autograd.py` (tiny model, use_ff9=True): frame-0 |grad| 6.5e-3 relay-on, 0.0
  detached.
- `probe_relay_grad.py` (this dir) — REAL DynamicsModelConfig, **use_ff9=False, bootstrap=False, d_min**
  (the clean re-run's exact loss), W=16, T=64, 6 slides, forced full-noise: init-window-only frames (whose
  latents can reach grad ONLY via the memory relay) get |grad| **0.499 relay-on / 0.0 relay-detached**.
  The init/relay frames carry the *dominant* gradient share — the noise-mode loss pushes hard on how
  memory is constructed from the scene. So the noise-mode signal trains memory; a no-FF9 collapse is
  optimization (signal weak/slow/long-horizon) or the 411270 confounds, NOT a broken gradient.

Why FF9 might still help even so (the hypothesis a clean run tests): FF9 is a SHORT-horizon (k=3), DENSE
(every frame), DIRECT (memory→next frames, one scored forward) signal. The noise-mode relay is
LONGER-horizon and INDIRECT — to earn credit, memory must be constructed to encode the scene AND relayed
across slides (new_mem copies old_mem forward, its WRITE gradient arriving only from the NEXT window) AND
be readable to reconstruct ~8 future frames. FF9 may act as a dense scaffold that bootstraps the memory
representation; without it the relay must learn the whole carry from a sparse self-referential signal.

## Relay gradient stability — does the memory-construction gradient vanish or explode? (`probe_relay_decay.py`)
Measured the per-hop backward factor of the relay BPTT chain (loss@slide s → read old_mem → its
construction @ s-1 → …), forced full-noise, no-ff9/no-boot/d_min, tbptt OFF (raw chain), W∈{4,8,16}.
Metric = geometric-mean |grad m_(deeper)| / |grad m_(shallower)| per hop (>1 explodes backward, <1 vanishes).

| | W=16 (4 hops*) | W=8 (8 hops*) | W=4 (16 hops*) |
|---|---|---|---|
| **random init** | 3.00 → 81× | 2.72 → 3.0e3× | 2.31 → **6.8e5×** |
| **trained winner** | 1.34 → 3.2× | 0.94 → 0.6× | 0.85 → 0.1× |

(*hops kept before the trainer's tbptt detach at 2N=32 frames = 2N/half.) **Forward** |m_i| is stable in
both (per-hop ratio ≈1.0) — only the magnitude scale grows with training (|m|~40–80 random → ~350–850 trained).

Findings:
1. **At RANDOM INIT the relay EXPLODES backward** (factor 2.3–3.0/hop), catastrophically for small windows:
   a W=4 relay compounds ~7e5× across its 16 tbptt hops. So early in training the relay gradient is wild,
   and `clip_grad_norm_(…, 1.0)` then rescales the whole step to the deepest-hop blow-up → the early relay
   signal for small windows is effectively noise. NOT vanishing — the opposite.
2. **Training SELF-REGULARIZES the relay to ≈ unit per-hop factor** (0.85–1.34). The converged winner has a
   well-conditioned, near-isometric relay (mild vanish for small W, mild explode for large W) — consistent
   with its flat-to-k=64 recall. So at convergence the chain is stable and no normalizer is required.
3. **Do we have a normalizer?** No DEDICATED per-hop relay normalizer — the carried memory is the RAW
   residual stream (`out_norm` is applied only to the latent output, not to the written memory token).
   What bounds the chain instead: (a) **TBPTT truncation** at 2N=32 frames caps the hop count; (b) **global
   `clip_grad_norm_=1.0`** caps explosion at the param level (but globally, so a blown-up relay drowns other
   signals); (c) **pre-RMSNorm + QK-norm** keep the forward/in-network path stable (forward |m| ratio ≈1.0);
   (d) **training itself** is the real regularizer (→ factor ≈1).

Implication for this ablation: FF9 may help not only as a denser signal but by giving stable, bounded
per-frame gradients that let the memory representation FORM, after which the relay self-regularizes.
Without FF9 the relay must self-organize from the noise-mode signal alone, whose early-training gradient is
the exploding chain above (worst for W=4). **If the clean no-FF9 run is unstable, the fix to try is a
per-hop relay normalizer** — e.g. a backward hook renormalizing the gradient crossing each slide boundary,
RMSNorm on the carried memory before re-injection, dropping W=4 from the n_ctx choices, or a longer LR
warmup. (Not changing training now — flagged for Merlin pending the run's stability.)

## Design — clean isolation (winner config minus FF9)
The rollout-only WINNER (`dynamics_mem2mem_rollout.pt`, job 411133) gets 0.99 recall WITH FF9 using the
`--no-bootstrap` sampler (d_min only, uniform τ, no curriculum). This run = that config + `--no-ff9`, so
the ONLY difference from the winner is the FF9 term.

```
python -u experiments/mem2mem/train_mem2mem.py \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout_noff9_clean.pt \
  --epochs 50 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 \
  --no-bootstrap --no-ff9 \
  --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-noff9-clean
```
- `--no-bootstrap` ⇒ d_min-only, uniform τ, no curriculum (the STABLE winner sampler — 411270's
  instability came from bootstrap+curriculum, which are off here).
- `--no-ff9` ⇒ memory trained ONLY by the rollout flow loss (50% clean / 50% full-noise).
- `--mem2mem-frac 1.0` ⇒ no normal-window batches (expect `train normal: 0.00000`).
- Data: 5× `data/gridworld.npy` + frozen `tokenizer.pt` already on cluster (same as 411133/411221/411502/3).

## Eval plan
Recall @ window=8, max_k=64 (K=4, +K=2/1), overlay vs: winner (with FF9), old confounded no-FF9 (411270,
`recall_dynamics_mem2mem_rollout_noff9.json`), vanilla/copy_last. position_acc mean / tail (k≥14).

## Predictions (pre-registered)
- **near-ceiling (≈ winner)** ⇒ noise-mode relay flow loss ALONE trains memory; FF9 NOT necessary; 411270
  negative was the confounds (Merlin vindicated).
- **still chance** ⇒ FF9 genuinely load-bearing despite the relay gradient flowing; noise-mode signal too
  weak/slow to learn the carry from scratch. Follow-ups: higher noise fraction, longer training, larger M.
- **partial** ⇒ FF9 helps but isn't strictly required; quantify the gap.

## Provenance
- Branch `exp/mem2mem-rollout-only`, SHA `8f54d097a3ed10b15e2c7603e42000f4494f0c01`. Cluster ferranti (H100).
- Job **412506** (`--name noff9clean`, 5h) → `dynamics_mem2mem_rollout_noff9_clean.pt`.
  Log: `runs/noff9clean/slurm-412506.out`.
- Compare against: winner `dynamics_mem2mem_rollout.pt` (411133, with FF9) + old confounded no-FF9 (411270).

## Status
- [2026-06-29] SUBMITTED. Relay gradient verified healthy WITHOUT FF9 (probe_relay_grad.py: init-only
  |grad| 0.499 relay-on / 0.0 detached). Job 412506 on ferranti @ SHA 8f54d09. Eval pending → recall
  overlay vs winner + old 411270.
