# ORIENT.md

Rewritten: 2026-06-26.

## What we're doing right now and why
The clean spec-driven rebuild is **merged to master** (`a51a78e`). On top of it a **spec→code sync** is
committed (`f91e2a0`): temporal cadence every-4th→every-3rd (`3×[spatial,temporal,spatial]`, depth 8→9),
dynamics attention gained the learnable per-head `logit_scale` (was a fixed `1/√d`), tokenizer decoder
gained a sigmoid output bound, FF9 terminal frame gets a sampled τ, and the trainers moved to required
`--frames/--tokenizer/--checkpoint` + `--fresh`→`--resume`. **Consequence: all existing checkpoints are
architecture-incompatible — everything must be retrained.**

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

## Recall A/B — first read DONE (provisional), `experiments/recall-ab/`, EXPERIMENTS `recall-AB`
- **STATIC attribute = memory WIN:** FF9 `color_acc` 1.000 at every k incl. k=20 (past the 16-window);
  vanilla decays 0.56→0.22. The memory token carries the static color past the window; vanilla forgets.
- **DYNAMIC = null:** `position_acc` near chance for both (FF9 0.03–0.09, vanilla 0.0–0.17), below
  copy_last, FF9 **not** > vanilla. Moving-square position is not retained.
- Caveats: env bounce period ~10 spikes copy_last to 1.0 at k=10/20 (confound); 50ep/7.75M/1000ep may
  under-train position; provisional on k-alignment. oracle self-test 1.0.

## NEXT (for Merlin to steer)
1. Sign off / adjust the recall **k↔tick alignment** (decision #2) so the absolute k-axis is trusted.
2. Decide whether to chase the **position null** (more train time / bigger model / longer schedule) or
   bank the color win for now. Consider the env-periodicity confound before over-reading position.
3. Cluster: ferranti up, idle (all retrain jobs COMPLETED). galvani socket down.

## Cluster / env
ferranti master socket UP, no jobs running. **galvani socket DOWN** (needs `open_master.sh --cluster
galvani` from Merlin if we want it). Local dev = Windows/Git-Bash; cluster orchestration = WSL wrappers.

## Background — the memory research
Frontier question: do per-timestep **memory tokens** let a Dreamer-4 world model retain hidden/off-screen
state past the short latent window? The clean code keeps the two ideas worth keeping: the **FF9
sufficiency loss** (memory must reconstruct future frames from memory alone) and the **carrying KV-cached
inference** (read old memory tokens' cached K/V, write new ones each step). The recall eval is the
result-defining spine — changing it silently redefines results.

## Parked
- Discrete/VQ memory idea (only if the carrying relay still drifts on GridWorld after retraining).
