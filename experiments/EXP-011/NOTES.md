# EXP-011 — position-deficit diagnostic (D-015). NO TRAINING.

Decision: D-015. Spawned by ESC-006 (Merlin: position is at chance even in open rollout → base
model never learned to track motion; predates FF7). Reuses frozen probe env+detector (5503e75).

## Question
The model's ball-position error hits chance even in an OPEN rollout (curtain up). Two failures
to disambiguate, plus a localization:
- (a) never learned motion (predicts static/incoherent ball), vs
- (b) learned motion but open-loop rollout chaotically desyncs from the *specific* GT trajectory.
- Localization: deficit in tokenizer C (position not encoded) vs dynamics D (encoded, not propagated).

## Setup
- Models: my_dynamics.pt (the base — the one that matters for (a)/(b)), plus ff7_k1/k3 for context.
- Frozen tokenizer trained_autoencoder.pt. N=8, P=3. Curtain-up episodes (k=0, R=24), 32 eps.
  Latent probe: 64 eps. Detector = the frozen probe's detect_ball (gate already validated).
- Components: (1) GT kinematics; (2) open-loop pos_err vs horizon + copy-last + chance;
  (3) teacher-forced 1-step pos_err vs GT per-step displacement; (4) linear probe latents→(x,y).

## Pre-registered reads (written BEFORE results — interpretation branches)
Key comparisons and what each implies:
- **Ball speed (comp 1).** If ~tiny (≪1px/step), copy-last is near-perfect and "position at
  chance" would partly be a metric artifact. [Smoke already showed ~3.4px/step — NOT tiny, so
  copy-last is a real, beatable baseline. Good: the test is meaningful.]
- **Open-loop model vs copy-last (comp 2).** Model BELOW copy-last at all horizons ⇒ it predicts
  motion better than freezing ⇒ motion learned. Model ≈ or ABOVE copy-last ⇒ it is ~freezing the
  ball ⇒ leans (a).
- **Teacher-forced 1-step vs GT displacement (comp 3) — the (a)/(b) discriminator.**
  - model 1-step ≪ GT displacement, and stays low across the trajectory (incl. bounces) ⇒ the
    1-step dynamics are GOOD; open-loop chance is compounding/chaos ⇒ **(b)**. Open-loop
    GT-matched position is then the wrong yardstick; the FF7 color result stands and position
    "memory" needs a closed-loop or distributional metric, not a base-dynamics fix.
  - model 1-step ≈ GT displacement (i.e. ≈ copy-last) ⇒ even one clean step it cannot beat
    freezing ⇒ **(a)** genuine motion deficit ⇒ fixing base dynamics (training/architecture/
    motion-aware loss) gates further H3 position work.
- **Latent probe (comp 4) — C vs D.** Low px err / high R² (position linearly decodable from
  frozen latents) ⇒ info is present; any deficit is in D's propagation. High err / low R² ⇒ the
  TOKENIZER bottlenecks position; no D-side method (incl. FF7) can carry it → reframes H3 position
  work onto C. [TRIPWIRE from D-015: not-decodable is the result that changes the most.]

My stated prior (D-015): lean (b) + position decodable from latents (deficit, if any, in D).
Run the copy-last/teacher-forced comparisons honestly — if the model only matches copy-last, that
is (a) and it overturns my prior.

## Provenance
- Code: master @ <SHA at run>. No training. Local 4070. Checkpoints as above.

## Observed
<fill after run>

## Reconciliation
<fill — Expected / Observed / Surprise / which branch (a)/(b) + C/D / Next>
