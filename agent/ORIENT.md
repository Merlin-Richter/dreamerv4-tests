# ORIENT.md

Rewritten: 2026-06-25 (clean spec-driven rebuild complete).

## What we're doing right now and why
**Clean rebuild of `src/` from the specs is DONE** (task `tasks/done/fill-src-from-specs.md`, branch
`rebuild/src-from-specs`, NOT yet merged to master). The pre-rebuild code had accreted many dead memory
experiments (FF7, multistep, ff9-rollout, snapshot/streaming inference). We rebuilt `src/` to exactly the
keep-list the spec map defines: a ~5× smaller, spec-faithful codebase that does only what the specs say.

## State of the rebuild (on branch `rebuild/src-from-specs`)
- `src/` = keep-list only: `wlog`, `envs/{base,gridworld}`, `models/{tokenizer,dynamics_model}`,
  `datagen/generate_gridworld`, `evals/gridworld/{readout,recall}`, `training/{train_dynamics,
  train_tokenizer}`, `interactive/play_dynamics`, `tests/{test_gridworld,test_gridworld_eval,test_dynamics}`.
- The two WRITE files were verified independently (critical-claim-verifier): **FAITHFUL, no BUG**.
  `dynamics_model.forward` is bit-identical to the old code for `n_memory=0`; the NEW carrying KV-cached
  rollout (read-old/write-new memory relay, 5th-pass commit, RoPE-by-absolute-index, window eviction) and
  the merged single-rollout `recall.py` scorer pass their probes. All 3 test files green.
- **Open decision flagged for Merlin:** the recall `k`↔tick alignment (documented at the top of
  `recall.py` + in the done-task). Needs sign-off before any recall numbers are trusted.

## NEXT (awaiting Merlin)
1. Merlin reviews the rebuild (esp. `recall.py` alignment + `dynamics_model` carrying inference) → merge
   `rebuild/src-from-specs` to master.
2. Train the GridWorld pipeline on this clean code: `generate_gridworld` → `train_tokenizer` (frozen) →
   `train_dynamics` vanilla + `--ff9 K` memory. Then `recall` A/B (memory vs vanilla) — the real test of
   whether the carrying relay retains hidden state past the latent window.
3. Cluster free; nothing running.

## Background — the memory research (history; the rebuild supersedes the dead code)
Frontier question: do per-timestep **memory tokens** let a Dreamer-4 world model retain hidden/off-screen
state past the short latent window? The clean code keeps the two ideas worth keeping: the **FF9 sufficiency
loss** (memory must reconstruct future frames from memory alone) and the **carrying KV-cached inference**
(read old memory tokens' cached K/V, write new ones each step). Prior campaign verdict (pre-rebuild,
archived): an FF9-no-rollout memory model retained STATIC hidden state past the window; the FF9
rollout-training relay (op-3) was a NEGATIVE result under the correct windowed inference. That whole
rollout-training line is intentionally DROPPED from the rebuild. The recall eval is the result-defining
spine — changing it silently redefines results.

## Parked
- Discrete/VQ memory idea (only if the carrying relay still drifts on GridWorld after retraining).
