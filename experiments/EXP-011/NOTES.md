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
- Code: master @ 9050a80-ish (EXP-011 scaffold commit). No training. Ran on CPU (cuda not
  visible to the background shell) — immaterial: FF7 numbers match EXP-010's cuda runs, and
  chance/ceiling reconcile with EXP-009 (below).

## Observed
- **GT kinematics:** ball speed mean **3.26 px/step** (median 3.20, p90 3.75). NOT tiny →
  copy-last ("freeze the ball") is a real, beatable baseline. chance pos_err **23.1 px**
  (matches EXP-009 chance 23.2).
- **Latent linear-probe (frozen tokenizer → x,y):** median **2.67 px**, R² **0.962**
  (n_test 346). **Position is cleanly encoded in the tokenizer latents.** The tokenizer C is
  NOT the bottleneck.
- **Teacher-forced 1-step pos_err** (perfect GT context, predict next frame) vs GT step 3.19px:
  - **my_dynamics: 4.52 px (median 4.20) — WORSE than copy-last (3.19).** Even with perfect
    context it predicts the next ball position worse than freezing it. (Consistent with EXP-009
    ceiling pos 5.58 px — no harness bug; the "1.1px" I'd remembered was the *detector* gate
    0.65px on GT frames, not the model.)
  - **ff7_k1: 1.02 px ; ff7_k3: 0.99 px** — ~3× better than copy-last, ~4.5× better than
    my_dynamics. Motion IS learnable on this data; these checkpoints do it well.
- **Open-loop pos_err vs horizon** (model | copy-last | chance 23.1):
  - my_dynamics: h1 4.5 | h4 10.1 | h8 17.3 | h12 22.4 | h16 25.8 | h24 23.5 — reaches (and
    briefly exceeds) chance by h≈13; weak.
  - ff7_k1: h1 1.0 | h4 6.9 | h8 15.6 | h12 19.8 | h24 20.5 — tracks better, saturates ~20.
  - **ff7_k3: h1 0.95 | h4 4.6 | h8 9.95 | h12 14.8 | h16 19.5 | h24 20.3** — tracks the moving
    ball well out to h≈12, then gradual open-loop compounding toward chance.
  - All models BEAT copy-last from h≈3+ (copy-last diverges as the ball moves away).
  - ball_lost_rate 0.00 everywhere (no frozen/blank-ball failure).

## Reconciliation
**Expected (pre-registered branches above):** disambiguate (a) never-learned-motion vs (b)
open-loop chaos, and localize C vs D. My prior: lean (b) + position decodable (deficit in D).

**Observed → verdict is split by model, and it's informative:**
- **Localization: deficit is in D, not C.** Position is linearly decodable from the frozen
  tokenizer latents (R²=0.96). The D-015 tripwire (not-decodable → tokenizer bottleneck →
  reframe H3 onto C) did **NOT** fire. Good: FF7-on-D (registers) can in principle carry
  position — the info reaches D.
- **my_dynamics ≈ failure (a):** its 1-step dynamics are genuinely weak (4.5px > copy-last
  3.2px even teacher-forced). The "position at chance" worry traces partly to the **baseline
  being an undertrained/weak motion model.**
- **FF7 (esp. k3) ≈ failure (b):** excellent 1-step (~1px), tracks position open-loop for ~12
  steps, then degrades to chance by gradual compounding. This is open-loop chaos, NOT inability
  to model motion.

**Surprise: HIGH (and reframing).** Three surprises:
1. **The FF7 checkpoints have far better motion dynamics than my_dynamics** (1-step 1.0 vs
   4.5px). The EXP-010 "position at chance" was misleading: it was measured only at the reveal
   frame, which for the drift control sits at horizon=n_occ — i.e. for n_occ 12/16/24 always
   deep in the saturated/chaos regime. The horizon-resolved view shows FF7 *does* track the
   moving ball; EXP-010's single-point snapshot hid it.
2. **The occluded position-at-chance is largely expected (b), not a memory defect:** dead-
   reckoning a *bouncing* ball through occlusion with zero visual feedback is chaotic — one
   bounce-timing error desyncs the exact GT trajectory. Color (static) survives; exact position
   (chaotic) cannot be expected to under an open-loop GT-matched metric. So that metric measures
   chaos, not memory, at long horizon.
3. **my_dynamics is a weak baseline relative to the FF7-era training** — which surfaces a
   methodological problem (below).

**CONFOUND I cannot resolve from this diagnostic (must flag):** is FF7's superior dynamics due
to the FF7 *loss*, or just that the FF7 runs were trained 100 epochs while my_dynamics (the old
baseline) was trained less/differently? Provenance of my_dynamics is older/approximate. A
budget-matched vanilla retrain is needed to attribute it. **Methodological consequence:**
EXP-009 (H2 baseline = my_dynamics) and EXP-010 (FF7, fresh 100-ep) are **not training-matched**.
The *color-memory* conclusion survives (the sliding-window cliff is architectural — a better-
trained vanilla model still can't see beyond its window), but a clean H3 comparison wants a
budget-matched vanilla baseline; my_dynamics should likely be retired as the baseline.

**Hypothesis impact:** Relieves the H3 position blocker substantially. Position is in the
latents (C fine); a trained model tracks motion well 1-step and ~12 steps open-loop. Whether
the *memory* approach can carry position through occlusion is still open, but it is NOT doomed
by an inability to model motion — the residual is open-loop/dead-reckoning chaos, which is a
metric/measurement question (closed-loop or distributional position), not a base-capability wall.

**Next: ESCALATE — present-then-stop (§5, EXP-011 is high-surprise + methodological).** → ESC-007.
No next decision (retrain baseline? closed-loop position metric? FF7 position variant?) until
Merlin's verdict.
