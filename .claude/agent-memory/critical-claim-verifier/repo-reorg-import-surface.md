---
name: repo-reorg-import-surface
description: T-019 repo-reorg review — the real import blast radius, the two patterns the plan misses, and where the BaseEnv/Eval interfaces are hand-wavy
metadata:
  type: project
---

T-019 (`tasks/T-019-repo-reorg-plan.md`) proposes src/{models,training,envs,data,evals,interactive}/. Reviewed 2026-06-16; verdict SOUND-WITH-FIXES.

**Why:** plan is structurally good (wrap-don't-move frozen spine = option A is correct) but under-counts import surface and over-claims interface fit.

**How to apply** when re-checking or executing the move:

Two import patterns the §5.2 recipe grep list MISSES (it lists `from probe_env`, `from dynamics_model`, etc. but not these):
1. **Package-style** `from probe.revisit_probe import ...` / `from probe.position_consistency import ...` — used by `experiments/EXP-013/coherence_eval.py`, `experiments/EXP-015/perf_rollout.py`. Works because `src/probe/__init__.py` exists (EMPTY, 1 line) and `src` is on sys.path. Relocating probe breaks these even under option A unless a `probe` shim package stays.
2. **`from data_generators.occluded_bouncing import OccludedBouncingEnv`** (package path, not bare) — `src/probe/probe_env.py:39`, `position_consistency.py:222`, `verify-T011-scorer/{scorer_probe,c4_markov_check}.py`. The FROZEN spine itself imports the env via this path. Moving env to `src/envs/` requires a back-compat `data_generators.occluded_bouncing` alias or the spine's import breaks — this is THE option-A reproducibility hole.

**BaseEnv (§2) fit is asymmetric, not "mostly declaring the ABC":**
- `OccludedBouncingEnv` fits cleanly: `reset(seed)->self`, `step(action)->(frame,state[5])`, `.color`. Real.
- bouncing sim has NO env class at all — pure procedural inside `generate_episode` (`bouncing_objects.py:94`), returns states `(T,4)=[x,y,vx,vy]`, n_actions=0, no per-step action. "Mechanical refactor" understates it; it's a real extraction + the only env exercising the n_actions=0 path.
- State dim differs (4 vs 5); `hidden_state()->[x,y,vx,vy,curtain]` contract is occluded-specific. ABC needs variable-width state, not a fixed signature.

**Eval interface (§3) `score(tok,dyn,*,device,budget)`** — the live-model args fit, BUT every existing eval also needs `K` (inference_steps), `tok_win`, window_N, episodes/n_occ grid (see `run_condition(tok,dyn,episodes,device,K,tok_win,...)`, `open_loop_curve(tok,dyn,episodes,device,K,H)`). These are not captured by `budget=` alone. `load()` returns `(tok,dyn,dcfg,tok_win)` — dcfg.inference_steps and tok_win are load-bearing and must flow into score(). Signature as written drops them.

Reusable probe pattern: grep BOTH bare (`from X import`) AND package (`from pkg.X import`) forms + `sys.path.insert` to get true blast radius; a recipe that only lists bare imports under-counts.
