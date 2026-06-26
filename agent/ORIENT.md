# ORIENT.md

Rewritten: 2026-06-26.

## What we're doing right now and why
Rebuild merged + spec→code sync done. Two campaigns just completed — both WINS:
1. **Retrain r2 (DONE):** dynamics with 5× data (5000 eps) + fixed LR schedule (warmup→flat→cosine
   80-100%). Fixed the r1 "position null" — FF9 position recall went from chance to near-perfect to
   k≈12 (then decays). val/loss 0.0058→0.0016. (NB: a PEP-604 `str|None` in the --model-module seam
   crashed on cluster py<3.10; fixed with `from __future__ import annotations`, 73b1c65.)
2. **mem→mem training (DONE, `experiments/mem2mem/`, `tasks/done/test-new-memory-training.md`):** new
   training signal teaching memory tokens to be built from prior memory tokens. Autograd check passes
   (relay grad 3.25e-3, 0.0 when detached). **Result: mem→mem holds position recall ~0.96 FLAT to k=20**
   where FF9 decays to 0.14 — long-horizon tail (k≥14) pos_acc vanilla 0.03 / FF9 0.20 / mem→mem 0.96.
   It carries hidden state indefinitely past the window — the core project goal demonstrated on GridWorld.

## Checkpoints (in `checkpoints/gridworld/`, all SHA-1688818-era, frozen tokenizer)
`tokenizer.pt`, `dynamics_vanilla.pt` (chance recall, no memory), `dynamics_ff9.pt` (recall to k≈12),
`dynamics_mem2mem.pt` (recall flat to k=20). Recall driver: `experiments/recall-ab/run.py` (3-way).

## Background: the spec→code sync (`f91e2a0`)
temporal cadence every-4th→every-3rd (`3×[spatial,temporal,spatial]`, depth 8→9), dynamics attention
gained the learnable per-head `logit_scale`, tokenizer decoder gained a sigmoid output bound, FF9
terminal frame gets a sampled τ, trainers moved to required `--frames/--tokenizer/--checkpoint`. **All
pre-sync checkpoints are architecture-incompatible — retrained.**

## Just finished (this session)
- **Cache-equivalence verification** (task → `tasks/done/verify-cached-vs-uncached-rollout-identical.md`).
  New gate test `src/tests/test_dynamics_cache.py` (green) + `experiments/verify-cache-equiv/`. Result,
  independently re-verified (critical-claim-verifier, fp64): the carrying KV-cached rollout `==` an
  uncached current-window recompute **bit-exact within the window**, but **diverges materially (O(1), not
  fp) once the sliding window evicts** — because ≥2 stacked temporal layers freeze each committed frame's
  deep-temporal K/V at its commit-time receptive field. Clean dichotomy: 1 temporal layer → exact;
  ≥2 → diverges. So the cache is a correct optimization *within* a window but **not** past it.
  EXPERIMENTS.md: `V-cache-equiv`. HOWTO updated (`rope_kv_cache_caveat.md`).

## Open decisions flagged for Merlin (do not silently resolve)
1. ~~Cache train/inference semantics gap~~ — **RESOLVED 2026-06-26**: the post-eviction divergence IS
   the intended information-preservation mechanism (Merlin). Recall correctly measures the carried path.
2. **Recall `k`↔tick alignment** (documented atop `src/evals/gridworld/recall.py`) — still needs sign-off
   before recall numbers are trusted. **This now gates the recall A/B.**

## Models ready (retrain DONE, `tasks/done/retrain-models.md`)
Trained on ferranti @ SHA `0a0e070`, pulled to `checkpoints/gridworld/`: `tokenizer.pt` (fg-weight 2.0,
val fg_mse 1.7e-5, no collapse), `dynamics_vanilla.pt` (n_memory=0), `dynamics_ff9.pt` (n_memory=4,
ff9_k=3). Both dynamics action-conditioned (n_actions=2). EXPERIMENTS: `R-gridworld-retrain`.

## Recall A/B history (`experiments/recall-ab/`)
r1 (under-trained): position null. r2 (5× data + LR fix): FF9 position recall near-perfect to k≈12 then
decays. 3-way (+mem→mem): mem→mem flat ~0.96 to k=20. See `NOTES.md` + `results_*.json`. EXPERIMENTS:
`recall-AB`, `R-gridworld-retrain2`, `mem2mem-train`.

## NEXT (for Merlin to steer)
1. **Recall `k`↔tick alignment** sign-off (decision #2) — still needed before *absolute* k numbers are
   trusted (the model-vs-baseline *comparisons* above are convention-robust).
2. **Graduate mem→mem?** It's the campaign's headline win. If keeping, fold it into `src/`+spec (it lives
   in `experiments/mem2mem/` now). Possible ablations: mem2mem-only vs 50/50; n_ctx schedule; longer
   max_k to find where it breaks; segmented-backward TBPTT (footprint); a harder env than GridWorld.
3. Cluster: ferranti UP, idle (all jobs COMPLETED). **galvani socket DOWN** (needs `open_master.sh
   --cluster galvani` from Merlin).

## Background — the memory research
Frontier question: do per-timestep **memory tokens** let a Dreamer-4 world model retain hidden/off-screen
state past the short latent window? The clean code keeps the two ideas worth keeping: the **FF9
sufficiency loss** (memory must reconstruct future frames from memory alone) and the **carrying KV-cached
inference** (read old memory tokens' cached K/V, write new ones each step). The recall eval is the
result-defining spine — changing it silently redefines results.

## Parked
- Discrete/VQ memory idea (only if the carrying relay still drifts on GridWorld after retraining).
