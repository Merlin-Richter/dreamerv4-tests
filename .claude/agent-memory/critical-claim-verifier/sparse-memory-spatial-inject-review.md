---
name: sparse-memory-spatial-inject-review
description: Design review of tasks/drafts/sparse-memory-spatial-inject.md — its compounding premise is CONTRADICTED by 411133 (dense relay flat to k=64); it is Design B + age-embedding; correct falsifier is "measure compounding on memmaze first".
metadata:
  type: project
---

**Finding (design review, 2026-07-04): the sparse-memory-spatial-inject proposal's CENTRAL PREMISE
("per-frame rewriting compounds errors; a chain of T rewrites gives T opportunities to compound") is
UNSUPPORTED and directly CONTRADICTED by the repo's own strongest result.**

**The killer number (verified from `outputs/recall/recall_dynamics_mem2mem_rollout.json` = job 411133,
window=8 native=16 n_mem=4):** the DENSE per-frame relay holds position_acc 0.97–1.00 FLAT from k=2 to
k=64 under sustained occlusion — i.e. 56+ consecutive per-frame lossy rewrites past the evicted window
with ZERO visible decay. `recall_noff9clean_K4.json` (dense no-FF9) same: 0.98–1.00 flat. copy_last sits
at chance except bounce-period spikes (k=10/40). So on GridWorld per-frame rewriting does NOT compound on
the dynamic state (position). The proposal admits this in its §5 but its §2 motivation asserts compounding
anyway — internal inconsistency.

**Why the premise's error-model is wrong (load-bearing):** compounding assumes each rewrite injects
INDEPENDENT noise → random-walk error ~√T, so ÷N rewrites help ~√N. But the TRAINED relay is a learned,
near-isometric, error-CORRECTING map (probe_relay_decay.py: trained per-hop FORWARD factor ≈1.0; forward
|m| ratio ≈1.0 at init AND trained). It is a stable fixed-point, not a noise accumulator. The proposal
cites the BACKWARD gradient explosion (~2–3×/hop at init) as evidence the "write path amplifies per hop" —
that is a MISATTRIBUTION: it's a backward/init transient that self-regularizes to ≈1 once trained
(noff9-fair NOTES), and forward error never compounds. Gradient-explosion-at-init ≠ forward-error-
compounding-at-convergence.

**Redundancy verdict (vs `tasks/drafts/sparse-memory-tokens.md`):** the new proposal IS that draft's
**Design B** (sparse write + re-inject last written set, piecewise-constant, zero-surgery `--model-module`
subclass) PLUS (1) a learned age/staleness embedding and (2) a bigger-set axis. The age embedding is a
GENUINE, arguably necessary fix (without it, full-occlusion reconstruction faces identical-input/different-
target → mode-averaged blur). The "SPATIAL read" framing is REBRANDING, not new mechanism: the current
model ALREADY reads memory only spatially (temporal attn is strictly slot-wise; memory→latent transfer is
within-frame spatial mixing at every layer — dynamics_model.py forward). Injecting the stale written set as
`memory_in` and letting spatial attn read it is exactly Design B. NOT a regression; NOT purely cosmetic
(age embed is real); the architectural-novelty claim is overstated.

**Internal tension the doc hides:** it wants Design B's "zero architecture change" (keep memory slots in
every frame → their temporal K/V are STILL committed and carry the constant M(t_w)) AND Design A's
"temporal channel freed to check what changed." Can't have both without the surgery. The temporal memory
channel is still there carrying near-constant content; "temporal checks what changed" is aspirational, not
mechanical. Worse, under the target regime (SUSTAINED OCCLUSION — the recall eval's regime, whole frame =
gray curtain), the latent "check what changed" channel is INERT (occluded latents carry no square motion),
so the design degenerates to PURE N-step extrapolation from a stale set — exactly the burden §4 concedes,
but the "what changed" narrative doesn't apply where it matters.

**Cheapest correct falsifier (NOT the flattering §6.1 A/B):** measure the PREMISE before building the fix.
On GridWorld the premise is ALREADY falsified by 411133 (no new run needed). The design's justification
rests entirely on memmaze, where NO recall/probe eval exists yet (memmaze-dyn NOTES: "memory CLAIMS wait on
the recall/probe eval"). So the correct first move is: **build the memmaze recall eval and measure whether
the DENSE relay compounds on memmaze.** If dense is also flat → premise dead, design unmotivated. Only if
dense decays on memmaze does sparse have a problem to solve. You cannot show a fix reduces compounding until
you've measured compounding exists.

**Reusable pattern:** when a design's motivation is "X compounds," find the longest-horizon existing recall
curve and check for the trend X predicts BEFORE funding the fix. Here one `json.load` of an existing artifact
settled it. See [[detached-carry-relay-drift]] (drift shows past trained horizon) and
[[anchored-selfrollout-vs-relay]] (anchored ≠ contraction) for the relay-dynamics priors.
