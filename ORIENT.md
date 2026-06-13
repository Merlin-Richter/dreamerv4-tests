# ORIENT.md

Rewritten: 2026-06-13 ~08:00 (EXP-011 done & reconciled; ESC-007 open — present-then-stop)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75; T-004 H3 bar = color ΔRGB < ~63 at n_occ {12,16,24}.
- **H3 — FF7 v1 supports COLOR (EXP-010).** Position was the open worry → diagnosed in EXP-011.
- **EXP-011 reframes the position worry:**
  1. Position is fully encoded in the tokenizer latents (linear probe R²=0.96) → deficit is in
     the dynamics D, NOT the encoder C (D-015 tripwire did not fire).
  2. my_dynamics is a weak motion model — 1-step teacher-forced 4.5px, WORSE than copy-last 3.2px
     (≈ failure (a)). The old baseline is undertrained at motion.
  3. FF7 tracks motion WELL: 1-step ~1.0px, open-loop to 14.8px@h12 (k3) before chaos → failure
     (b), not inability. EXP-010's "position at chance" was a reveal-frame snapshot in the
     saturated regime; horizon-resolved, FF7 clearly tracks position.
  → Occluded position-at-chance = dead-reckoning a bouncing ball through occlusion (chaotic),
    a measurement issue, NOT a base-capability wall. Position-memory is NOT doomed.

## In flight
**NOTHING running. 4070 idle. present-then-stop gate (ESC-007) — awaiting Merlin's verdict.**
Per §5 the §3 prep allowance does NOT apply: do not start a baseline retrain, metric change, or
any FF7 variant until he answers.

## Open methodological issue (raised in ESC-007)
EXP-009 baseline (my_dynamics) and EXP-010 (FF7, fresh 100-ep) are **NOT training-matched**.
Can't attribute FF7's better dynamics to the loss vs just more training. Color-memory conclusion
survives (sliding-window cliff is architectural), but H3 wants a **budget-matched vanilla
baseline**; my_dynamics likely retired. This is the first overnight/cluster-worthy run candidate.

## NEXT ACTION
Wait for ESC-007 verdict. His three questions: (1) agree with the reframing? (2) train a
budget-matched vanilla baseline (my rec: yes — needed regardless)? (3) position path —
closed-loop/distributional metric vs color-only-and-move-on vs FF7 position variant (my lean:
baseline first, then closed-loop position metric, then judge position). On his answer: write the
next decision, then act.

## Access points for his review (ESC-007)
- `experiments/EXP-011/headline.png` (open-loop pos_err vs horizon; killer numbers in title)
- `experiments/EXP-011/results.json`; full reconciliation `experiments/EXP-011/NOTES.md`

## Current worries
1. The FF7-loss-vs-more-training confound is real and blocks a clean H3 attribution until a
   budget-matched vanilla baseline exists.
2. Don't over-claim FF7 "solved motion" — it's better than my_dynamics, but open-loop still
   decays to chance by h~16; the win is 1-step + medium-horizon tracking.
3. Position-memory through occlusion still genuinely unproven (chaos makes the open-loop metric
   uninformative) — needs a metric that measures memory, not trajectory chaos.
