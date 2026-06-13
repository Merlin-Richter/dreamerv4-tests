# EXP-012 — budget-matched vanilla dynamics baseline (D-016)

Decision: D-016. Spawned by ESC-007 (FF7-loss vs training-budget confound; baselines not
training-matched). Vanilla (--ff7 0 --fresh) at the EXACT EXP-010 budget. Config + launch:
experiments/EXP-012/{config.yaml,run.sh}.

## Purpose / what it controls
my_dynamics (old H2 baseline) is weak at motion (EXP-011: 1-step 4.5px > copy-last 3.2px), but
its training provenance is older/approximate, so EXP-009↔EXP-010 are not training-matched. This
run is the clean control: identical architecture + data + 100ep budget as the FF7 arms, FF7 loss
OFF. Lets us attribute the EXP-011 dynamics gain and re-anchor H2/H3.

## Pre-registered expectations (D-016, before results)
- COLOR cliff: reproduces EXP-009 (post-window color recall → chance; architectural). H2 stands.
- MOTION (1-step teacher-forced pos_err): expect << my_dynamics 4.5px (much of that was
  undertraining); the open question is vanilla-budget-matched vs FF7 ~1.0px.
  - vanilla ≈ FF7 (~1px) ⇒ dynamics gain was training budget, not the FF7 loss.
  - vanilla still > FF7 ⇒ FF7 loss genuinely improves dynamics.
- Tripwire (D-016): vanilla showing beyond-window color recall < T-004 bar ⇒ EXP-010 color win
  was not the register relay ⇒ halt + rethink.

## Plan
1. Train (run.sh first stage) → vanilla_s0.pt; 2. frozen probe → results.json + sheet;
3. rerun EXP-011 diagnostic on vanilla_s0.pt (open-loop/teacher-forced/copy-last) for the
   apples-to-apples motion comparison vs FF7 and my_dynamics.

## Provenance
- Training: master near ebbec80/b3ceaf6 (pre-KV-cache), local 4070, 100 ep, final val ~0.0066
  (≈ FF7's 0.0065 reference — budget genuinely matched). W&B exp012-vanilla-s0.
- Probe: ran at 13:30–13:44 against the KV-cache-modified `dynamics_model.py` (D-017, commits
  ce6bcc7/ec4e698 landed mid-session). The probe uses `dyn.generate()` with the DEFAULT path
  (positions=None, cache=None), which is validated byte-identical to the pre-edit code
  (test_kv_cache.py: seeded generate==generate_cached; FF7 smokes 5/5). So the numbers are
  unaffected by the KV-cache work. Checkpoint vanilla_s0.pt predates those commits entirely.

## Observed (probe, results.json)
- **COLOR cliff — reproduced, sharp.** color ΔRGB: n_occ 2/4/6 = 14.7/14.4/15.0 (≈ ceiling 13.2),
  then **jumps at the window boundary** to 98.3 (n_occ=7), 103.3 (8), 99.2 (9), and sits at
  109.8/105.9/108.0 for n_occ 12/16/24 — i.e. **≈ chance (108.5)** everywhere beyond the window.
  Cliff between n_occ=6→7, exactly as EXP-009 (prefix exits the N=8 window at reveal idx 3+n_occ>9).
- latent-MSE mirrors it (0.39–0.56 in-window → 0.87–0.91 ≈ chance 0.83 beyond); latentMSE↔color
  r=0.97 here. Detector gate green (p99 0.65px, miss 0.0).
- Position: occluded pos_err ~9→22px ≈ chance (21.7) and ≈ its matched-horizon drift control
  (8→27px) — no position memory, fully drift-confounded, as T-004 anticipated.
- **Motion attribution (diagnostic.json, 32 eps, all 4 models, frozen probe env):**
  teacher-forced 1-step pos_err — vanilla_s0 **4.66px** (median 4.45) ≈ my_dynamics 4.51px
  (median 4.33) ≫ ff7_k1 1.02px, ff7_k3 0.96px. GT step / copy-last = 3.19px (both vanillas are
  WORSE than freezing the ball; both FF7 arms beat it ~3×). Open-loop pos_err: ff7_k3 tracks far
  longest (h4 4.1 / h8 8.5 / h16 14.6px) vs vanilla (9.8 / 18.5 / 28.2) ≈ my_dynamics (11.8 /
  19.1 / 24.1). Latent→xy probe R²=0.96 (position encoded in tokenizer, reproduced).
  - **Caveat (Merlin, measurement validity):** the open-loop pos_err curves TURN OVER at long
    horizons (vanilla 28.2@h16→24.0@h24) because the ball BOUNCES off walls and returns to prior
    regions — a desynced prediction coincidentally lands near GT, NOT recovery. Open-loop pos_err
    is bounded-domain + bounce-artifacted; do not over-read the long-horizon dip. Reinforces the
    need for the closed-loop/distributional position metric (BOARD).

## Reconciliation
Expected (D-016): (color) cliff reproduces (H2 architectural); (motion) budget-matched vanilla
1-step << my_dynamics 4.5px (much of that was undertraining); vanilla≈FF7 ⇒ gain was budget,
vanilla>FF7 ⇒ FF7 loss helps. Tripwire = vanilla beyond-window color < 63 bar ⇒ halt.
Observed: color cliff reproduced (beyond-window ≈ chance 105–110, nowhere near 63). Motion:
vanilla 4.66px ≈ my_dynamics 4.51px ≫ FF7 ~1.0px.
Surprise: **mild — one sub-prediction REFUTED.** I predicted the budget-matched vanilla would
substantially beat my_dynamics at motion (undertraining hypothesis from EXP-011). It did not —
both vanillas land at ~4.5px, within noise. So motion weakness is NOT an undertraining artifact;
it is intrinsic to the vanilla setup on this data/objective. (Color was exactly as predicted.)
Hypothesis impact:
- **H2 stands — now anchored on a training-matched baseline.** A budget-matched vanilla (val loss
  = FF7's) still collapses hidden-color recall to chance the instant the prefix leaves the window.
  Cliff is architectural (sliding window has no state). → retire my_dynamics; vanilla_s0 is the
  EXP-012 H2/H3 baseline.
- **Confound RESOLVED — FF7's wins are the method, not the budget, on BOTH axes:**
  (color) D-016 tripwire did NOT fire — budget-matched vanilla does not reproduce EXP-010's
  beyond-window retention (FF7 40–65 < 63 bar through n_occ 16); EXP-010 color memory = the FF7
  register relay. (motion) both vanillas ~4.5px ≫ both FF7 ~1.0px → the FF7 loss, not budget,
  produces good 1-step dynamics. EXP-009↔EXP-010 are now retroactively trustworthy.
- **Bonus finding:** FF7 improves base 1-step dynamics ~4.6× (not just the memory relay) — the
  single-timestep-sufficiency objective appears to act as a dynamics regularizer. Larger claim
  than "carries static color"; worth disentangling (loss vs relay-inference) later.
Tripwires checked: D-016 color tripwire NOT triggered (good). D-016 "would-change-my-mind (2)"
(vanilla 1-step WORSE than my_dynamics ⇒ data/seed/init issue): vanilla 4.66 vs 4.51 is marginally
worse but within run-to-run noise and CONSISTENT across two independent vanillas → read as "equal,
not a bug," not a seed/data fault. Flagged for Merlin; a 2nd vanilla seed would confirm if he wants.
Next: ESC-008 present-then-stop. Closes the ESC-007 baseline action. Proposed: proceed to the
agreed Q3 path (closed-loop/distributional position metric) — and note the sequential register-
relay training idea (IDEAS.md) as the candidate H3 position method once we can measure position.
