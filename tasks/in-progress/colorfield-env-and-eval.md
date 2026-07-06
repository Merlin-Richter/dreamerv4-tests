# ColorField — frozen env + eval layer for the autoresearch harness

Build the self-contained "prepare layer" of the autoresearch harness (Karpathy-autoresearch-style,
discussion with Merlin 2026-07-06): a memory env between GridWorld and Memory Maze, its dataset,
a closed-form readout, and THE result-defining comeback eval. Everything lives under
`autoresearch/frozen/` — self-contained (no imports from `src/`, no specs link), gate-tested,
then FROZEN by hash after Merlin's sign-off.

Motivation (the identified gap): memmaze mem2mem simulates physics/on-screen persistence well but
fails to remember what was there after looking away. Mechanism hypothesis: the gradient that forces
consolidation into carried memory tokens is proportional to how often off-screen info becomes
relevant again in the data. GridWorld = dense pressure (reveal any tick) → flat recall; memmaze =
sparse/delayed/conditional pressure → info dropped across relay hops before ever queried.
ColorField makes time-to-relevance and revisit frequency explicit, tunable knobs.

Related: memory tokens are NOT trained to be per-step Markov-sufficient — the trained invariant is
"the union of memory tokens across the live window suffices" (consistent with FF9 proving
unnecessary, and with registers acting as a temporal side-channel; gwv2 campaign findings).

## Env design v1 (Merlin's, 2026-07-06/b — updated after second design round)

- **Map**: **15×15 cells**, cell = 12px → 180×180px world. Each cell iid uniform over a **5-color
  palette**; outside the map = a fixed **6th color**. Fresh map per episode. (~225·log2(5) ≈ 522
  bits of episode state.)
- **View**: egocentric 64×64 **RGB** uint8 window (autoresearch/ is self-contained — RGB
  end-to-end inside it; convert only at cv2 display boundaries). View **center** lives on the
  **90×90 sub-cell lattice** (2px pitch = 1/6 cell) ⇒ at map corners up to 32px (~2.7 cells) of
  color-6 band is visible — the only absolute-position landmark.
- **Actions (LOCKED)**: up/down/left/right = one lattice step (2px) + **stay** (n_actions=5).
  **Invalid-action semantics (Merlin)**: an outward move at the lattice edge is NOT a valid
  action — not clamping/collision; it cannot even be tried. Env raises on invalid action;
  datagen policies sample only from the valid set; consequently push-at-border NEVER appears in
  training data (⇒ eval must never feed one either — see eval v2 closed-loop policies).
- **Dynamics**: fully deterministic given (map, start, action stream). State =
  (center_col, center_row) on the 90-lattice + the 225-cell map. `hidden_state()` /map access
  are measurement-only (never a model input) — BaseEnv contract carried over.
- **NOT in v1**: walls, collision, goals, curtain, grid lines (flat color cells only).
  Pixel-aligned rendering only — no subpixel, no aliasing.
- Difficulty dials (documented, not varied in v1): map N, palette size, view size, step size.

## Datagen — diverse behaviour-policy zoo

All policies sample only VALID actions. Episode: fresh map + random start, **T = 1024** (Merlin)
— long episodes = long-range revisit structure = the memory-pressure knob. ~5000 train + held-out
val episodes, mix ≈ uniform over policies; policy id (+ params) logged per episode so
revisit-pressure vs recall stays studyable later.

**Storage is PROCEDURAL (proposed)**: 5000×1024 raw frames ≈ 60 GB — instead store only
(map, start, actions) per episode (few KB; rendering = a crop of the 180×180 world image) and
render frames on the fly in the dataloaders (tokenizer training included). The fp16 latent cache
(~2.6 GB) is still materialized once for dynamics training. Dataset hash = hash of the sidecar
arrays.

- **P1 goal-seek (Merlin's)**: sample a random goal on the 72×72 lattice; each step move on x
  with p = d_x/(d_x+d_y) toward the goal, else y; with prob ε take a uniform random action
  (ε per episode ∈ [0.05, 0.3]); resample goal on arrival.
- **P2 momentum walk**: repeat last action w.p. p ∈ [0.6, 0.95], else uniform.
- **P3 lawnmower sweep**: boustrophedon coverage, random orientation + lane spacing.
- **P4 box patrol**: random rectangle circuit, ≥2 laps.
- **P5 dwell-and-dart**: jitter near a point, then dart to a far point, repeat.
- **P6 border-hug**: lap along the map edge (maximal color-6 exposure).
- **P7 uniform random**.
- **P8 out-and-back oscillator**: random axis + amplitude (10–60 steps), repeated.

## The comeback eval — v2 (imagination-mode; second design round 2026-07-06, OPEN items marked)

Merlin's steer: no read-only-branch-vs-GT bookkeeping (information enters view too slowly; would
need roll-forth-and-back logic). Instead run the eval **in imagination** with a **cell tracker**
on top and score what the imagination comes up with. Agent's guards added because the
autoresearch loop is an OPTIMIZER pointed at this number: pure self-consistency has a degenerate
optimum (an all-one-color imagined world is perfectly consistent) — so half-anchor to ground
truth and gate on prerequisites.

**Structure per eval episode** (seeded, deterministic given seed + model):
1. **Real prefix** (~128–256 steps, OPEN): teacher-forced commits of TRUE frames while following
   the eval policy; start near a corner so a border is OBSERVED (pins the imagined lattice).
2. **Imagination phase** (~512–1024 steps, OPEN): pure carried rollout. Driver feeds actions from
   a seeded **closed-loop eval policy** that consults the READOUT OF THE IMAGINED FRAME and never
   pushes into an imagined border (color-6 band ≥ 32px on a side ⇒ that direction forbidden).
   Required, not cosmetic: invalid actions cannot be tried and never occur in training data, so a
   fixed script hitting an imagined-early border would feed an out-of-distribution input
   (undefined behavior). Policies are structured patterns (out-and-back-N, box loop, sweep,
   spiral, idiot-walk) parameterized to cover the age bins; 10–20 policies × 8–16 seeds.

**Cell tracker** (over imagined frames; registration = path-integral of taken actions):
records each tile's color whenever visible (read at MAX visibility within a visit, majority vote
across the visit's frames), with provenance:
- **real-observed** (seen during the prefix) → comeback events scored against **GROUND TRUTH** —
  ungameable retention, the direct descendant of the GridWorld recall eval.
- **imagination-born** (first seen during imagination) → comeback events scored for
  **SELF-CONSISTENCY** vs the previous visit's recorded color (Merlin's metric; unlimited horizon;
  drift-inclusive by design).
Comeback definitions unchanged: *on-screen* = viewport∩cell overlap ≥ 6px in x AND y ⟺ center in
view; *fully left* = zero pixel overlap; event = on-screen → fully left → center back; one score
per (cell, event); never-seen cells never scored.

**Age standardization (Merlin's requirement, made structural)**: every event logs an age; all
headline numbers are means over FIXED age bins averaged with EQUAL bin weight (min-events-per-bin
enforced, occupancy reported) ⇒ distribution shifts (e.g., early imagined borders → younger
recalled info) move bin populations, NOT the score. The age-vs-accuracy curve is a first-class
output (successor of the recall-vs-k curve).

**Age definition (v2.1, red-team-hardened)**: age = the LONGEST contiguous zero-overlap absence
the cell survived since its previous visit. Weaker definitions fell to constructed exploits:
"since last on-screen" is inflatable by partial-visibility hovering (1–5px slivers keep a cell
alive in any context window); "since first zero-overlap" is inflatable by chaining short absences
via partial refreshes (multi-gap bridging). A window-W model cannot bridge a single absence > W,
so beyond-window bins are chance by construction. "Since last on-screen" kept as `age_onscreen`
diagnostic.

**Scoring (v2.1, revised per the adversarial red-team — Merlin sign-off PENDING)**:
- Per-bin CHANCE CORRECTION over in-map events: acc_cc = max(0,(acc−1/5)/(1−1/5)). Kills the
  0.2-floor padding that let "win near bins, chance far bins" models score 0.62.
- **Border (OUT) tiles are EXCLUDED from the scored accuracy** — reported as a separate
  border_recall diagnostic (overall + per-bin). DEVIATES from the original 0.1 weight
  (Merlin's instruction): measured, border events are ~78% of far bins on loop policies and are
  pure geometry (a zero-content-memory model gets them right at any age), so ANY nonzero weight
  lets them set the long-range score.
- **composite = real_cc × (0.7 + 0.3 × consistency)** — MULTIPLICATIVE: consistency can only
  amplify GT-anchored retention, never substitute (the additive form let a zero-retention
  "consistent liar" (0.43) outrank honest 16-frame memory (0.25)).
- Adapter factories are SANDBOXED: candidate models receive None (baselines opt into
  privileged=True) — handing out env was an instant-oracle hole.
Post-fix reference curve (small config): liar 0.01 / W=16 0.14 / W=32 0.32 / W=64 0.55 /
W=128 0.79 / full-memory 1.0 — monotone in genuine retention horizon, score ≈ fraction of the
age spectrum with demonstrated retention. Regression fence:
tests/test_eval.py::test_bounded_window_monotone_and_capped.

**Hard gates (score := floor if failed)** — prerequisite competences + Goodhart guards:
1. **Action fidelity**: per-step imagined scroll (frame-to-frame cross-correlation) vs the
   commanded 2px — catches "actions do nothing" models (cf. memmaze vanilla) whose consistency
   numbers would otherwise be meaningless.
2. **Color-marginal entropy**: imagination-born first-seen colors ≈ uniform over the 5 palette
   colors (KL threshold) — catches collapse-to-one-color.
3. **Oracle self-test**: tracker fed GT frames must score exactly 1.0 on BOTH provenances (gate test).

**Headline scalar** (the loop's number; weights OPEN — Merlin): age-standardized composite
≈ **0.7·real-anchored accuracy + 0.3·consistency**, out-of-map (color-6) tiles weighted **0.1**
inside each term (their values are mutually determined once the border is placed).
Also reported (diagnostics, not optimized): unweighted / in-map-only variants, per-policy
breakdown, in-window (k ≤ W) vs beyond, border-position drift (imagined vs true border distance —
Merlin's early-border worry becomes a measurement), action-fidelity stats, entropy, bin occupancy,
mean age of recalled info. Eval-once-to-JSON, plot-many.

**Readout**: closed-form, pure numpy — per-cell average of visible pixels, nearest of the 6
palette colors; registration from the action path-integral (prefix: true state — identical).

**Baselines through the identical eval**: oracle, random-guess chance (empirical), copy-last-frame,
trained no-memory reference (tau0-anchor vanilla analog). Expected event count ≈ thousands of
binary scores ⇒ σ well under 1% — verified during harness calibration.

## Freezing + integrity

- Layout: `autoresearch/frozen/{env.py, policies.py, datagen.py, readout.py, scripts.py,
  eval_comeback.py, train_tokenizer.py (vendored), tests/}`; dataset + tokenizer ckpt gitignored
  but their SHA-256 recorded.
- After sign-off: record SHA-256 of every frozen file in `autoresearch/frozen/MANIFEST.json`.
  The driver re-hashes before every scoring call; any mismatch → score := chance, run flagged
  `tampered`. The driver computes all scores itself — the loop agent never self-reports.
- Gate tests (CPU OK): geometry + readout exactness (oracle 1.0 on every eval policy × seed,
  both provenances); comeback + age bookkeeping vs a brute-force per-frame reference; env raises
  on invalid action + no policy ever emits one (fuzzed); closed-loop eval policy never pushes
  into a border (fuzzed on synthetic imagined frames incl. early/wrong borders); policy smoke
  (coverage/revisit stats per policy sane); determinism (same seed ⇒ identical episode);
  procedural render == materialized frames (spot-check).

## One-time prep (after freeze)

1. Generate dataset (train + val) locally; record hashes.
2. Train the tokenizer (vendored gridworld arch; start 4 latents / 64 bottleneck) on colorfield
   frames — local 4070 or one short cluster job. VERIFY recon is readout-exact on val (the gwv2
   check); if not, adjust before freezing the ckpt. Freeze.
3. Build the latent cache for dynamics training.

## STATUS (2026-07-06, session log)

- Design signed off by Merlin (incl. eval v2 imagination-mode, 0.7/0.3 anchoring, corner-start
  border anchoring). Remaining detail choices delegated: palette RGB values, P=192/I=768,
  bin edges (1,17,33,65,129,257,inf) — all now concrete in code, flag if objectionable.
- IMPLEMENTED @ e4fc77d: `autoresearch/frozen/{env,readout,policies,eval_policies,eval_comeback,
  adapters,datagen}.py` + 5 gate-test suites — ALL GREEN. Highlights: oracle composite == 1.0
  exactly; tracker == independent brute-force event reference; perfect_imaginary ("consistent
  liar") scores consistency 1.0 / composite ~0.44 (demonstrates the 0.7 GT-anchoring);
  constant_color / noise_cells / copy_last all gated to 0.0 (fidelity/entropy gates work —
  NB copy_last DOES produce comeback reads under the moving registration; the fidelity gate is
  what kills it, i.e. the gate is load-bearing).
- Fix during build: perfect_imaginary baseline needed grid-PHASE alignment with the tracker
  registration (real models get it free from the prefix; the privileged baseline peeks env.pos).
- ADVERSARIAL REVIEW DONE (3 independent background agents, Merlin's order), all reports in
  experiments/colorfield-{geometry-audit,bookkeeping-audit,redteam}/REPORT.md:
  * geometry/OOD audit: 5/6 CONFIRMED incl. structural proof + 3.46M-action fuzz of the band
    guard; 1 finding (estimate_shift tie-break on texture-free frames — kept strict-fail by
    design, degenerate imagination SHOULD fail fidelity; documented).
  * bookkeeping audit: 7/7 CONFIRMED vs independent reimplementations (oracle re-verified 1.0
    at FULL frozen config, 51k events). Caveats N1 (min age structural), N2 (equal-weight
    protection conditional on qualified-bin set).
  * red-team: EXPLOITABLE at v2.0 → SCORING v2.1 fixes applied + regression fence (see the
    eval section above + EXPERIMENTS row V-colorfield-redteam). All 5 suites re-green.
- AWAITING MERLIN SIGN-OFF on 3 semantic changes: border-tile exclusion (vs his 0.1 weight),
  multiplicative composite, max-gap age definition.
- NOT DONE YET: dataset generation (procedural, ~minutes), tokenizer training + latent cache,
  MANIFEST hash freeze (last, after sign-off).

## Done when

Env + policies + eval + gate tests green; tokenizer frozen + readout-exact; MANIFEST hashes
recorded; Merlin has signed off the remaining OPEN items: eval-v2 design overall + composite
weights (0.7/0.3 proposed), prefix/imagination lengths, age-bin edges, concrete palette RGB
values, procedural dataset storage. No EXPERIMENTS.md line until something produces a result.
