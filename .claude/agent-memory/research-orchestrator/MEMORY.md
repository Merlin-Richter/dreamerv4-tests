# Research Orchestrator — Memory Index

- [Milestone = reevaluate, not execute](feedback_milestone_reevaluate.md) — at milestones Merlin wants a critical re-read of the trajectory, not mechanical continuation of the backlog
- [Measurement validity is first-class](feedback_measurement_validity.md) — Merlin scrutinizes whether a metric can actually be measured cleanly (e.g. color recall needs ball localization); design for it
- [No privileged data constraint](project_no_privileged_data.md) — H3: model never gets privileged hidden state; methods must generalize across envs; eval-only access is OK
- [GridWorld metric semantics](project_gridworld_metric_semantics.md) — colour=static memory (still failable), position=memory+reasoning; both are memory tests, not just position
- [Ground ML claims in code](feedback_ground_claims_in_code.md) — verify architecture/gradient claims against the module (cite file:line) before asserting; don't reason from priors
- [Causal prefix is in-distribution](feedback_causal_prefix_in_distribution.md) — shorter inference context than train length is IN-distribution for causally-masked transformers; don't call it OOD
- [Evals should use the env directly](feedback_evals_use_env_directly.md) — generate eval data from the env (controlled scenarios), not the fixed dataset/val set
- [Sequence subobjectives, keep the end-goal visible](feedback_sequence_subobjectives.md) — finish one subobjective in isolation before the next, but don't scrub the big-picture goal when narrowing scope
- [Use idle overnight GPU](feedback_overnight_gpu.md) — when wrapping up late, launch a relevant low-risk overnight run (even a known-imperfect method); don't rush a bug-prone build
- [Cluster wrappers run in WSL](feedback_cluster_use_wsl.md) — scripts/ verbs + ssh master socket MUST run in WSL (shared socket namespace), not Git Bash/PowerShell; invoke via wsl.exe
- [Never clobber cluster.env](feedback_never_clobber_cluster_env.md) — scripts/cluster.env is Merlin's gitignored secret config; back up before any test that writes scripts/, never rm/overwrite it
