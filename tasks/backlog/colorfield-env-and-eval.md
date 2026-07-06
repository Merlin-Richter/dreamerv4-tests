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

## Env design v1 (Merlin's, 2026-07-06 — items marked OPEN need his sign-off)

- **Map**: 12×12 cells, cell = 12px → 144×144px world. Each cell iid uniform over a **5-color
  palette**; outside the map = a fixed **6th color**. Fresh map per episode.
- **View**: egocentric 64×64 BGR uint8 window. View **center** lives on the **72×72 sub-cell
  lattice** (2px pitch = 1/6 cell). Center clamped to the lattice ⇒ at map corners up to
  32px (~2.7 cells) of color-6 band is visible — the only absolute-position landmark.
- **Actions**: up/down/left/right = one lattice step (2px), clamped at lattice edge; plus
  **stay** (OPEN: include stay? recommend yes, gwv2 precedent, n_actions=5).
- **Dynamics**: fully deterministic given (map, start, action stream). State =
  (center_col, center_row) on the 72-lattice + the 144-cell map. `hidden_state()` /map access
  are measurement-only (never a model input) — BaseEnv contract carried over.
- **NOT in v1**: walls, collision, goals, curtain, grid lines (flat color cells only; OPEN:
  confirm no grid lines). Pixel-aligned rendering only — no subpixel, no aliasing.
- Difficulty dials (documented, not varied in v1): map N, palette size, view size, step size.
- BGR end-to-end, [0,255] uint8 frames like the rest of the repo.

## Datagen — diverse behaviour-policy zoo

All policies emit (frames, actions, states); per-episode sidecars: policy id (+ params), map.
Episode: fresh map + random start, T = 192 frames (OPEN: T). ~5000 train + held-out val episodes,
mix ≈ uniform over policies. Policy id logged so revisit-pressure vs recall stays studyable later.

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

## The comeback eval (result-defining scalar — the harness's val_bpb)

Scripted episodes, exact bookkeeping, one weighted accuracy number.

- **Definitions** (cells = 12px tiles, grid extended outside the map for color-6 tiles):
  - *on-screen*: viewport∩cell overlap ≥ 6px in x AND in y ⟺ cell center inside the view.
  - *fully left*: zero pixel overlap with the viewport.
  - *comeback event*: on-screen → fully left → center on-screen again. Scored ONCE per
    (cell, event), at the FIRST re-entry frame. Never-seen cells are never scored (iid ⇒
    unpredictable). A cell can produce multiple events per episode.
- **Mechanics [OPEN — needs Merlin's explicit sign-off, changes what is measured]**.
  Recommended: **teacher-forced carry + read-only branch** — each step commits the TRUE frame
  into the carrying rollout (memory written from real observations); at steps whose next frame
  contains ≥1 comeback re-entry, take a read-only branch prediction (rollout_step commit=False
  analog), decode, score those cells. Direct analog of GridWorld recall's branch-reveal;
  isolates relay retention from free-rollout drift; branch forwards only on event frames (cheap).
  Alternative (free rollout after a real observation leg) becomes a DIAGNOSTIC variant, not the
  scalar.
- **Readout**: closed-form, pure numpy — registration from TRUE state (known center at the scored
  frame); per scored cell average its visible pixels, nearest of the 6 palette colors. Diagnostic
  columns (not the scalar): best-shift score via cross-correlating the predicted frame against the
  GT map + the shift error, to separate "forgot the color" from "lost registration".
- **Scripts**: 10–20 hand-written action sequences × 8–16 map seeds each. Start positions chosen
  per script so NO action ever clamps (asserted at eval build time). Ladder of difficulty, e.g.:
  out-and-back (40L,40R) and shorter/longer variants; box loop (40R,40U,40L,40D,40R); zigzag
  sweep-and-return; spiral-out-return; long-away-return (~60 steps away); comb pattern; multi-loop.
  Away-durations must span well past W (in-window events are the easy tail — intended, gradual eval).
- **Score**: weighted accuracy over all (cell, event) pairs — in-map weight 1.0, **out-of-map
  (color-6) tiles weight 0.1** (their values are mutually determined, near-free once the border is
  placed). Also reported: unweighted, in-map-only, per-k slices (k = steps since last on-screen),
  in-window (k ≤ W) vs beyond-window, per-script breakdown. Eval-once-to-JSON, plot-many.
- **Baselines through the identical eval**: oracle (render GT — MUST score exactly 1.0, gate
  test), random-guess chance (empirical), copy-last-frame, and a trained no-memory reference
  (tau0-anchor vanilla analog). Expected event count ≈ thousands of binary scores ⇒ σ well
  under 1% — verified during harness calibration.

## Freezing + integrity

- Layout: `autoresearch/frozen/{env.py, policies.py, datagen.py, readout.py, scripts.py,
  eval_comeback.py, train_tokenizer.py (vendored), tests/}`; dataset + tokenizer ckpt gitignored
  but their SHA-256 recorded.
- After sign-off: record SHA-256 of every frozen file in `autoresearch/frozen/MANIFEST.json`.
  The driver re-hashes before every scoring call; any mismatch → score := chance, run flagged
  `tampered`. The driver computes all scores itself — the loop agent never self-reports.
- Gate tests (CPU OK): geometry + readout exactness (oracle 1.0 on every script × seed);
  comeback bookkeeping vs a brute-force per-frame reference; no-clamp assertions; policy smoke
  (coverage/revisit stats per policy sane); determinism (same seed ⇒ identical episode).

## One-time prep (after freeze)

1. Generate dataset (train + val) locally; record hashes.
2. Train the tokenizer (vendored gridworld arch; start 4 latents / 64 bottleneck) on colorfield
   frames — local 4070 or one short cluster job. VERIFY recon is readout-exact on val (the gwv2
   check); if not, adjust before freezing the ckpt. Freeze.
3. Build the latent cache for dynamics training.

## Done when

Env + policies + eval + gate tests green; tokenizer frozen + readout-exact; MANIFEST hashes
recorded; Merlin has signed off every OPEN item (eval mechanics, stay action, grid lines, T,
concrete palette BGR values). No EXPERIMENTS.md line until something produces a result.
