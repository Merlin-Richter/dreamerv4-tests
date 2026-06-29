# Fair bootstrap A/B — does the shortcut bootstrap ladder ACTUALLY hurt retention?

Re-run of `experiments/mem2mem-rollout-boot/` (job 411221, NEGATIVE) with its confounds removed. That
run conflated the bootstrap with three other changes vs the rollout-only winner; this isolates the
bootstrap gradient. See `tasks/in-progress/fair-bootstrap-ablation.md` for the full rationale.

## The confounds we remove
1. **FF9 normalizer dilution** → fixed with `--ff9-norm-flow`: normalize the FF9 term by the pure d_min
   FLOW magnitude (`flow_norm`), not the mixed flow+bootstrap diffusion mean. The mixed mean is dragged
   down by the smaller bootstrap self-distillation term, which silently down-weights memory. (Local
   smoke: `flow 0.113` mixed vs `flow_norm 0.266` pure ⇒ the old code under-weighted FF9 ~2.4×.)
2. **τ distribution** → held IDENTICAL across both arms by using the curriculum d-sampling in BOTH (the
   d-snapped grid that bootstrap requires). It cannot be matched to the *winner* (which used uniform τ),
   because bootstrap's two-half-step target is only well-defined on the d-grid — so the τ-shift is
   intrinsic to bootstrap and is measured separately (Arm A vs winner) rather than removed.
3. **Epochs** → both arms 50ep (matches the winner; the old boot ran only 36).

## Design — 2-arm factorial, 50ep, ferranti H100
- **Arm A (control):** `--boot-loss-off --ff9-norm-flow`. Snapped-τ + curriculum, bootstrap LOSS off
  (coarse-d tokens get flow MSE). Everything the boot run has EXCEPT the bootstrap gradient.
- **Arm B (fair boot):** `--ff9-norm-flow` (bootstrap + curriculum ON by default).
- **A vs B** → pure bootstrap-gradient effect (τ, FF9-weight, epochs all identical).
- **A vs winner** (`dynamics_mem2mem_rollout.pt`, uniform-τ pure-flow 50ep) → the τ-shift effect.

## Commands
```
# Arm B (fair boot)
python -u experiments/mem2mem/train_mem2mem.py \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout_boot_fair.pt \
  --epochs 50 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 \
  --ff9-norm-flow \
  --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-boot-fair

# Arm A (control)
python -u experiments/mem2mem/train_mem2mem.py \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout_bootctrl.pt \
  --epochs 50 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 \
  --boot-loss-off --ff9-norm-flow \
  --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-bootctrl
```

## Eval plan
Recall @ window=8, max_k=64 (+ K=2/1), overlay 4-way: Arm A, Arm B, winner (no-boot rollout-only),
vanilla/copy_last baselines. position_acc mean / tail (k≥14) / k=64.

## Predictions (pre-registered)
- If **B ≈ A ≈ winner**: the bootstrap is free; the old negative was purely the confounds (normalizer +
  epochs). Intuition vindicated — shortcut forcing does not hurt retention.
- If **B < A**: the bootstrap gradient itself hurts memory on this task (would be surprising).
- If **A < winner**: the τ-shift (snapped grid, ~25% τ=0) is the culprit, independent of bootstrap.

## Provenance
- Branch `exp/mem2mem-rollout-only`, SHA `851a7ab8d8f9d88827e3d9588c4ecd4d0ed742d1`. Cluster ferranti (H100).
- **Arm B (fair boot):** job **411502** → `dynamics_mem2mem_rollout_boot_fair.pt` (--ff9-norm-flow, 50ep, 5h).
- **Arm A (control):** job **411503** → `dynamics_mem2mem_rollout_bootctrl.pt` (--boot-loss-off --ff9-norm-flow, 50ep, 4h).
- Data: 5× `data/gridworld.npy` + frozen `tokenizer.pt` already on cluster (same as jobs 411133/411221).
- Compare against: winner `dynamics_mem2mem_rollout.pt` (job 411133) + old unfair boot (411221).

## Result (2026-06-29) — the old "bootstrap halves retention" was the CONFOUNDS, not the gradient
Both jobs COMPLETED clean (`rc=0`, full 50ep, stable — no instability). Training already showed the
normalizer confound fixed: Arm A `flow==flow_norm` every epoch (boot off ⇒ diffusion mean IS pure flow);
Arm B `flow 0.0010 < flow_norm 0.0029` (boot drags the mixed mean ~3× lower, and `--ff9-norm-flow`
correctly used the larger pure-flow basis). **Final FF9 loss: Arm B 0.013 / Arm A 0.0105 — both ~4×
better than the old unfair boot's 0.054.** That 5×-worse FF9 was the mechanism behind the old collapse,
and it's gone. But training is not the result — the recall eval decides.

Recall @ window=8, max_k=64, K∈{4,2,1}, n_rollouts=64 (same scorer @ recall.py `786e6ce`; winner +
baselines reused). position_acc:

| model | K=4 mean | K=4 tail(k≥14) | K=2 mean | K=1 mean |
|---|---:|---:|---:|---:|
| **Arm A — control (boot OFF)** | **0.998** | 0.996 | 1.000 | 1.000 |
| **Arm B — fair boot (boot ON)** | **0.968** | 0.973 | 0.980 | 0.999 |
| winner (rollout-only no-boot, 411133) | 0.992 | 0.988 | — | — |
| old UNFAIR boot (411221) | 0.472 | 0.424 | — | — |
| vanilla / FF9-50/50 (ref) | 0.042 / 0.387 | 0.035 / 0.135 | — | — |

Verdict against the pre-registered predictions:
1. **B ≈ A ≈ winner — CONFIRMED.** All three are flat near-ceiling to k=64; the old 0.472 collapse is
   GONE. So the bootstrap does NOT halve retention — the 411221 negative was the FF9-normalizer dilution
   (dominant) + epochs + τ-shift, exactly the confounds. Merlin's intuition (shortcut forcing shouldn't
   hurt) is vindicated.
2. **B < A — small, consistent, NOT catastrophic.** Arm B sits a hair below Arm A at K=4 (0.968 vs
   0.998) and K=2 (0.980 vs 1.000), equal at K=1 (0.999 vs 1.000). The bootstrap gradient carries a tiny
   residual cost (~3 pts) with no offsetting gain.
3. **A < winner — FALSE (good direction).** Arm A (0.998) ≥ winner (0.992): the τ-shift (snapped grid,
   ~25% τ=0) is benign once FF9 is normalized to pure flow. If anything the snapped-τ+curriculum control
   is marginally the cleanest model of the set.

**Few-step (the bootstrap's actual motivation): no upside.** Arm A (boot OFF) is ALREADY perfect at K=2
(1.000) and K=1 (1.000) — there is nothing for the shortcut ladder to fix. Confirms the old run's point
#1, even more strongly. Adding the diffusion-step / bootstrap loss is now PROVEN SAFE for retention but
POINTLESS on GridWorld: the pure finest-step x-prediction flow + FF9 already nails single-step inference.

**Decision (GridWorld only): keep the simple rollout-only + FF9 + x-prediction.** The bootstrap is free,
not harmful — but it buys nothing *on this env* and costs a few points, so don't ship it *for GridWorld*.
What changed vs the old claim is WHY (confound, not a real hurt). Overlay: `compare_w8_k64_4way.png`.
Per-K JSONs: `outputs/recall/recall_{bootfair,bootctrl}_K{4,2,1}.json`.

**SCOPE — do NOT generalize this to "drop shortcut forcing."** All three configs (winner / Arm A / Arm B)
keep the FULL diffusion-forcing loss; the A/B toggles only the *shortcut bootstrap self-distillation* on
coarse step sizes, NOT diffusion-on-vs-off. Arm A's "boot off" trains coarse steps with flow/x-prediction
MSE (predict clean in one big step). The reason there's no upside here is that GridWorld is DETERMINISTIC:
the next-latent distribution is a delta, so single-step x-prediction (the conditional MEAN) is exactly
right → K=1 nails it and the few-step shortcut has nothing to buy. On a STOCHASTIC/multimodal env a
one-step x-prediction returns the blurred conditional mean (mode-collapse); you then need multiple
diffusion steps to sample a sharp transition, and the bootstrap's self-consistency is the mechanism that
keeps few-step sampling on the correct trajectory (Arm A's big-step flow MSE would itself regress to the
mean). GridWorld can neither show that upside nor test the diffusion sampler's real job. Validating the
bootstrap/diffusion benefit requires a stochastic env (cf. `tasks/drafts/harder-grid-env.md`).

## Status
- [2026-06-29] DONE. Jobs 411502 (B) + 411503 (A) completed clean on ferranti @ SHA `851a7ab`. Both
  checkpoints pulled (`dynamics_mem2mem_rollout_boot_fair.pt`, `dynamics_mem2mem_rollout_bootctrl.pt`).
  Recall A/B run locally (4070). Result above; task → done.
- [2026-06-27] SUBMITTED. Code verified (probe |diff|=0, autograd relay intact, both arms smoke clean @ 4070).
