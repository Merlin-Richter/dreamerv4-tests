# ORIENT.md

Rewritten: 2026-06-16 (after the T-019 repo reorg, with Merlin steering live).

## What we are doing and why
Just completed a **full repo restructure (T-019, D-030/D-031)** at Merlin's request — reframe the
codebase around concerns so it scales to "many envs and evals." The research thread (C1 / motion,
exposure-bias / open-loop compounding) is **paused mid-flight** and resumes next.

## Repo structure NOW (src/, reorganized — see REPO_MAP.md)
`models/` `training/` `envs/` (BaseEnv ABC + the 2 sims) `datagen/` (dataset writers) `evals/`
(common Eval interface: `base.py` REGISTRY, FROZEN spine under `revisit/`+`position_consistency/`+
`probe_env.py`, working `motion/`+`rollout_view/`) `tests/` (5 gate tests) `interactive/` `wlog.py`.
- **Eval interface:** `import evals; evals.discover()` → REGISTRY {motion, revisit}; each has
  `score(tok,dyn,cfg)` (cheap scalars, mid-run-able) + optional `report(...)`; `compatible_envs` tags.
- **FROZEN spine** moved byte-identical-except-imports (gated by old→new diff; commit 5503e75 logic intact).
- **Historical experiment scripts are FROZEN to their commit (D-031)** — they import old paths and are
  NOT rewired. To rerun an old EXP: `git checkout` its commit. NEW work imports from `src/evals/`.

## Reorg status: DONE (4 phases, all committed + gated)
- P1 env/data split · P2 evals move + frozen-spine relocation + Eval interface · P3 models/training/
  tests/interactive split · P4 docs (CLAUDE.md + REPO_MAP). Gates each phase: 5 gate tests green,
  spine code-identity + smoke, BouncingEnv byte-identical, Eval adapters run on ff7_k3.

## Research thread — PAUSED, resumes next (was the overnight C1/motion work)
- **EXP-021 (full-data C1 training)** left a usable checkpoint `experiments/EXP-021/c1_full_s0.pt`
  (~epoch 10/40; the overnight job stopped). Purpose (D-029): confound-free open-loop compounding test
  — C1-full vs the COMPETENT reference set (vanilla_s0 / ff7_k3 / ff9v2) at matched TF-map quality.
- **Pending NEXT ACTION (research):** probe `c1_full_s0.pt` with open-loop + TF motion curves AND the
  per-model `context_signal` sweep (EXP-022 made the s-sweep a standard axis): does C1's TRAINED
  robustness beat ff7 + the tuned inference knob? **Do this via the NEW evals interface** (MotionEval /
  `evals.motion.motion`), NOT the frozen EXP-021/eval.py. Reconcile → view → escalate to Merlin (§5).
- Key prior findings still standing: EXP-020 C1 SUPPORTED (250-ep, weak-control caveat); EXP-022
  context_signal lever HALVES ff7 open-loop compounding, C1 already optimal at 0.9.

## Current worries
1. EXP-021 checkpoint is only ~ep10 — TF map may not yet be competent enough for a clean compounding
   comparison; if so, retrain longer before concluding.
2. Single seed throughout the C1 thread — positive deltas need a seed before any standalone claim.

## Parked
- **ESC-014 (OPEN):** op-3 relay gradient design (dynamic-state occluded memory). Resume after motion.
