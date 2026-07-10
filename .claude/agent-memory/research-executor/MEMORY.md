# Research Executor — Memory Index

- [GridWorld determinism vs diffusion steps](feedback-gridworld-determinism-diffusion.md) — don't conclude "drop diffusion/shortcut forcing" from GridWorld; it's deterministic and can't test stochastic sampling
- [Spec-edit delegation (memmaze campaign)](feedback-spec-edit-delegation.md) — Merlin granted spec edits + autonomous task management for the memmaze dynamics campaign (2026-07-03); campaign-scoped, don't generalize
- [Autoresearch generality rule](project-autoresearch-generality-rule.md) — editable-layer changes must be env-GENERAL (novelty-weighting ok, phase/period-specific weighting = cheating); Merlin 2026-07-07
- [Investigate "broken" claims](feedback-investigate-broken-claims.md) — when user says a wait feels wrong, verify ground truth before re-asserting "known flakiness"; caught a real bug this way 2026-07-10
- [Bash timeout on long waits](feedback-bash-timeout-long-waits.md) — run_in_background still kills at timeout (max 600s, rc=124); launch detached + Monitor-poll cluster waits, never hold them in one call
