# OPERATING — how the AI works in this repo (task-driven)

Replaces the old orchestrator protocol. The AI is an executor, not a decision-maker.

- **Work only on tasks Merlin specifies** — files in `tasks/backlog/`, or a direct instruction.
  No autonomous research decisions, no escalations, no present-then-stop gates.
- **Tasks are files** (`tasks/`, one file per task; filename = short description, contents = details).
  State = which folder. Check status by printing the tree (`find tasks -type f`), not by opening files.
- **Task flow:** `git mv tasks/backlog/<task>.md tasks/in-progress/` when starting → do it →
  `git mv` to `tasks/done/` and append a one-line result. If blocked or ambiguous, ask — don't improvise.
- **Keep `agent/ORIENT.md` current** (one page: what's being worked on now, anything in flight, next).
- **Experiments:** put artifacts in `experiments/EXP-NNN/`; add ONE short line to `agent/EXPERIMENTS.md`.
- **Code is spec-driven:** code is generated from `specs/<file>.md`; Merlin owns the specs, the AI
  translates and verifies (against tests + `src_old`). Don't redesign code outside its spec.
- **Reversible by default:** work on branches, archive instead of delete, commit with clear messages.
