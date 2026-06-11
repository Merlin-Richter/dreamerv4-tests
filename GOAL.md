# GOAL.md — Research goal and hypotheses

> **Owner: Merlin.** The orchestrator may propose amendments but applies them only
> after a milestone conversation in which he agreed.
>
> Backfilled 2026-06-11 from journal, git history, and W&B. The H1–H3 framing was
> approved by Merlin in the protocol-adoption exchange (see ESC-001). Success
> criteria for H2/H3 are still to be pre-registered (task T-004) — they MUST be
> written before the experiments that decide them.

## Research idea

World models learned from video (DreamerV4 lineage) compress each frame through a
tokenizer trained with reconstruction-only objectives. Such latents only need to
represent what is *visible in that frame*; nothing forces them to retain
currently-hidden state — inventory contents, off-screen objects, anything behind an
occluder. A dynamics model living in that latent space therefore cannot recall
hidden information, however good its next-frame loss looks (the "look north, turn
around, everything is hallucinated" failure).

We investigate alternative encoding objectives that force the latent state to
retain all information that, under the right actions (open the inventory, lift the
curtain), affects future frames.

Testbed: CurtainsEnv (`src/data_generators/occluded_bouncing.py`) — a bouncing
ball with per-episode fixed color and background, plus an absolute curtain action
that hides the entire frame. A model with ≥1 visible context frame should recall
ball/background color; with ≥2, also position (velocity derivable).

---

## H1: A DreamerV4-style tokenizer + dynamics pipeline can be reproduced at small scale on synthetic video

Status: **partially supported** — tokenizer supported, dynamics open
Success criteria (written retroactively, hence weak): tokenizer reconstructions
visually faithful without latent collapse; dynamics rollouts preserve per-episode
constants (ball color, background) and approximate ball position over ≥10 frames.
Evidence: EXP-002, EXP-003 (negative→fixed), EXP-004, EXP-005, EXP-006 (tokenizer:
supported), EXP-007 (dynamics: **not yet** — rollouts randomize ball color and
position; under diagnosis, D-009)

## H2: Reconstruction-only frame latents + a short temporal window cannot retain occluded state

Status: **open**
Success criteria: TO BE PRE-REGISTERED (T-004) before the deciding experiment.
Proposed shape: on the frozen revisit-consistency probe suite (observe ≥2 visible
frames → occlusion of k frames → reveal), the unmodified baseline pipeline's
re-reveal predictions show near-chance recall of ball color and position as a
function of k. Requires the probe suite (T-002) built and frozen first.
Evidence: EXP-007 (suggestive — rollouts lose ball identity immediately — but
confounded: dynamics model may simply be broken/undertrained; see D-009)

## H3: An encoding objective that forces retention of action-revealable hidden state improves recall-after-occlusion

Status: **open, not started**
Success criteria: TO BE PRE-REGISTERED (T-004) before the first method experiment.
Must be measured on the frozen probe suite, ≥2 seeds, against the H2 baseline under
identical provenance discipline. Reconstruction/next-frame loss alone never decides
this hypothesis.
Evidence: —
Blocked on: EXP-007 diagnosis resolved (D-009), probe suite frozen (T-002), H2
baseline measured.
