# T-001 report — Context-noise rollout diagnostic harness

Built and run inline by the orchestrator (not delegated to a cold worker): the task
is ~140 lines reusing the existing `run_test_checkpoint` load/decode path, single-
threaded, and the orchestrator was blocked on its result regardless — so the
context-economy/parallelism rationale for spawning a worker did not apply, and the
Agent tool's "don't spawn unless asked / it's the expensive path" guardrail did.
(Flagged to Merlin as a harness/process note in ESC-002.)

Artifact: `experiments/EXP-008/diagnose_context_noise.py` (headless, deterministic).

Acceptance criteria:
1. `--n-episodes 2` runs headless to completion, exit 0, in well under 5 min. PASS
   (a few seconds on the 4070).
2. Writes `experiments/EXP-008/results.json` with a numeric `gen_mse_mean` per
   tau_ctx. PASS.
3. Writes one `images/ep{E}_s{S}_tau{TAU}.png` per (episode, tau_ctx), each a 2-row
   GT/rollout strip with the context boundary marked, plus `_sheet_tau{TAU}.png`
   contact sheets. PASS.
4. Re-run with same seed → identical numbers. PASS (deterministic seeding;
   full 6-ep run is the recorded artifact).

Full sweep command (recorded experiment EXP-008):
    python experiments/EXP-008/diagnose_context_noise.py --n-episodes 6

No source file had to be modified (the diagnostic only mutates
`model.config.context_noise` at runtime, which `_denoise_next` reads at call time).

Interpretation is in `experiments/EXP-008/NOTES.md` + ESC-002, done by the
orchestrator (not in this report, per the present-then-stop protocol).
