---
name: feedback-spec-edit-delegation
description: Merlin delegated spec editing + autonomous task creation/management to the agent for the memmaze dynamics campaign (2026-07-03)
metadata:
  type: feedback
---

For the memmaze dynamics campaign (latent cache + vanilla/mem2mem training, started 2026-07-03),
Merlin explicitly granted: "You can edit specs. Please create tasks yourself into backlog and manage
them while you work. I'm always available for questions."

**Why:** the campaign is compute-heavy and Merlin wants velocity; he stays in the loop via questions
rather than pre-approving each spec diff.

**How to apply:** for THIS campaign, edit `specs/**` directly when the work needs it (keep spec<->code
in sync as always) and create/move task files without waiting for Merlin. This relaxes OPERATING.md §3
("Merlin owns the specs") for the campaign only — still flag design decisions and surprising spec
changes in the task/ORIENT notes, and still ask when a decision is genuinely his (model size, compute
budget, research direction). Do not generalize this grant to other work without re-confirmation.
