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

### Parallel arm 2 — + per-hop relay gradient normalizer (`--relay-grad-clip 0.05`)
Motivated by `probe_relay_decay.py` + `measure_clip_scale.py`: at init the relay gradient EXPLODES backward
(~2–3×/hop; real-data deepest-hop |grad| 0.03 @W=16 but **26 @W=8, 88 @W=4**), self-correcting to ~1 only
once trained. Hypothesis: that early explosion (worst for the small windows the trainer samples), combined
with global grad-clip rescaling the whole step to it, may stop the memory representation from forming
WITHOUT FF9's stable dense scaffold. Arm 2 = arm 1 + a per-hop relay grad normalizer: a backward hook
scales each carried memory tensor's gradient DOWN per sequence to ||grad_b|| ≤ C=0.05 (scale-down only;
near-hop/converged grads ≪ C pass untouched). Training-only — forward & inference identical to arm 1 (no
src/ change, no train/inference mismatch). C=0.05 chosen from the real-data measurement: it barely touches
W=16/near hops, caps the catastrophic W=8/W=4 deep hops, and self-disengages as training stabilizes
(converged grads ~1e-6 ≪ C). Verified: OFF ⇒ byte-identical (relay autograd + newhalf-loss probes pass);
ON ⇒ forward identical, relay grad still flows (reduced not zeroed), clip engages; 2-epoch local smoke
`relay_clip≈0.30` (taming ~30% of hops, loss decreasing). Same command as arm 1 + `--relay-grad-clip 0.05`,
ckpt `dynamics_mem2mem_rollout_noff9_clip.pt`, wandb name `…-noff9-clip`.

## Eval plan
Recall @ window=8, max_k=64 (K=4, +K=2/1), overlay vs: winner (with FF9), old confounded no-FF9 (411270,
`recall_dynamics_mem2mem_rollout_noff9.json`), vanilla/copy_last. position_acc mean / tail (k≥14).

## Predictions (pre-registered) — two arms, A/B on the normalizer
- **arm1 (no norm) near-ceiling (≈ winner)** ⇒ noise-mode relay flow loss ALONE trains memory; FF9 NOT
  necessary; 411270 negative was the confounds (Merlin vindicated). Normalizer then irrelevant (but arm2
  should match — a check it doesn't hurt).
- **arm1 chance BUT arm2 (norm) near-ceiling** ⇒ the early relay-gradient EXPLOSION was the blocker; the
  per-hop normalizer rescues no-FF9 (the normalizer hypothesis confirmed). The cleanest possible outcome.
- **both chance** ⇒ FF9 is genuinely load-bearing as a dense short-horizon scaffold; conditioning the relay
  gradient isn't enough. Follow-ups: higher noise fraction, longer training, larger M.
- **both near-ceiling** ⇒ FF9 not necessary and the normalizer is harmless; prefer the simpler arm1.
- **partial** ⇒ quantify the gap; report where each lands on the recall curve.

## Provenance
- Branch `exp/mem2mem-rollout-only`. Cluster ferranti (H100).
- **Arm 1 (no normalizer)**: SHA `8f54d09`, job **412506** (`noff9clean`, 5h) →
  `dynamics_mem2mem_rollout_noff9_clean.pt`. Log `runs/noff9clean/slurm-412506.out`.
- **Arm 2 (+ per-hop relay grad-norm, C=0.05)**: SHA `e266bea`, job **412510** (`noff9clip`, 5h) →
  `dynamics_mem2mem_rollout_noff9_clip.pt`. Log `runs/noff9clip/slurm-412510.out`. (Arm-1's job is
  unaffected by the e266bea re-sync: the normalizer is default-OFF and byte-identical.)
- Compare against: winner `dynamics_mem2mem_rollout.pt` (411133, with FF9) + old confounded no-FF9 (411270).

## Result — ARM 1 (no-FF9, NO normalizer): FF9 is NOT necessary; the old 411270 negative was the confounds
Job 412506 completed clean (50ep, rc=0, STABLE — val 0.0051, ff9 0.0000, train normal 0.0, d_unlocked 1/8;
none of 411270's instability because bootstrap+curriculum are OFF here). Recall w8 max_k64 position_acc:

| model | K=4 mean | tail(k≥14) | K=2 | K=1 |
|---|---:|---:|---:|---:|
| **Arm 1 — no-FF9, no normalizer (412506)** | **0.989** | 0.988 | 0.999 | 0.999 |
| winner — rollout-only WITH FF9 (411133) | 0.992 | 0.988 | — | — |
| old confounded no-FF9 (411270) | 0.044 | 0.041 | — | — |
| vanilla | 0.042 | 0.035 | — | — |

⇒ Near-ceiling, flat to k=64, matching the FF9 winner. **The 50% full-noise rollout mode alone trains
memory to carry hidden state — FF9 is redundant on this task.** 411270's chance recall was the
bootstrap+curriculum+instability+36ep confounds (exactly like the discredited boot run), NOT a missing FF9.
Merlin's intuition vindicated. (Also: the init relay explosion the probes measured did NOT block learning
in this stable d_min-only config — so the normalizer wasn't even needed for success; arm 2 checks if it's
neutral/helpful.)

## Result — ARM 2 (no-FF9 + per-hop relay grad-clip 0.05) + FINAL VERDICT
Job 412510 completed clean (50ep, rc=0, val 0.0055). The normalizer behaved exactly as designed: **clip
fraction 0.133 in epoch 1, then 0.000 every epoch after** — it tamed the init relay explosion, then the
relay self-regularized below the cap and the clip disengaged (matches the trained-model factor ≈1 from
probe_relay_decay.py).

Final recall (w8, max_k64), position_acc:

| model | K=4 mean | tail (k≥14) | k=64 | K=2 | K=1 | colour@k64 |
|---|---:|---:|---:|---:|---:|---:|
| **Arm 1 — no-FF9, no normalizer** (412506) | **0.989** | 0.988 | 0.984 | 0.999 | 0.999 | ~0.84 |
| **Arm 2 — no-FF9 + relay-clip 0.05** (412510) | **0.985** | 0.980 | 0.953 | 0.996 | 1.000 | ~0.78 |
| winner — rollout-only WITH FF9 (411133) | 0.992 | 0.988 | 1.000 | — | — | ~0.95 |
| old confounded no-FF9 (411270) | 0.044 | 0.041 | 0.031 | — | — | ~0.19 |
| vanilla (no memory) | 0.042 | 0.035 | 0.031 | — | — | ~0.19 |

**Verdict:**
1. **FF9 is NOT necessary on GridWorld.** Clean no-FF9 (arm 1) is near-ceiling and flat to k=64 (0.989),
   matching the FF9 winner (0.992). The 411270 "FF9 necessary → chance" was the CONFOUNDS
   (bootstrap+curriculum+instability+36ep), NOT a missing FF9. The 50% full-noise rollout mode alone
   teaches memory to carry hidden position; the relay gradient flows (probes) and learning succeeds.
   Merlin's intuition vindicated.
2. **The per-hop relay normalizer is ~neutral here** (arm 2 ≈ arm 1, 0.985 vs 0.989; within eval noise).
   The init relay explosion (probe_relay_decay.py: ~3×/hop, 88 @W=4) is a transient that the stable
   d_min-only config rides out via global grad-clip; the normalizer engaged only in epoch 1 then idled.
   It's harmless → keep it as an OFF-by-default flag for harder/longer-relay envs (e.g. Memory Maze),
   where the init explosion may actually bite.
3. **Residual FF9 edge:** on the long-horizon STATIC attribute (ball colour past the window) FF9 keeps
   ~0.95@k64 vs no-FF9's ~0.78–0.84 — a small advantage. Position (the dynamic state) is matched without it.

Visuals: `compare_w8_k64_noff9.png` (3-panel 5-way overlay); occlusion sheets
`sheets_{arm1_noff9clean,arm2_noff9clip,ref_vanilla}/sheet_occlusion.png` — both memory arms' belief
(bottom) tracks the true square (top) cell-for-cell through 16 occluded frames; vanilla's belief collapses
(bg/colour/position scramble) once the window evicts.

## Status
- [2026-06-29] DONE. Both arms completed clean on ferranti (412506, 412510) @ SHAs 8f54d09 / e266bea.
  Checkpoints pulled, recall (K=4/2/1) + overlay + occlusion sheets produced. Verdict above: FF9 not
  necessary on GridWorld (411270 was confounds); relay normalizer neutral-but-harmless. Task → done.
