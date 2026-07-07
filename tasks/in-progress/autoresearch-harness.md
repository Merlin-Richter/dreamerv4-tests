# Autoresearch harness + calibration (ColorField)

Karpathy-autoresearch-style loop for this repo (discussion 2026-07-06): fixed-budget from-scratch
training experiments on the ColorField env, scored by the frozen comeback eval, agent iterating on
a single editable file. Depends on: `tasks/backlog/colorfield-env-and-eval.md` (frozen layer).

## Layout — self-contained subdir, no specs↔src link

```
autoresearch/
  frozen/        # env, eval, datagen, tokenizer, MANIFEST.json  (built by the other task; FROZEN)
  editable/
    train.py     # THE one file the loop edits: model + objective + hyperparams, from-scratch each run
  driver/        # deterministic harness code (NOT editable by the loop)
  runs/          # per-experiment artifacts + experiments.jsonl + leaderboard.md  (gitignored)
  program.md    # instructions for the loop agent (the "research org code", Merlin iterates on this)
  README.md
```

- `editable/train.py` seeded from a vendored copy of the mem2mem-rollout recipe (the current best
  memory training) sized down to budget. Self-contained — vendor whatever model code it needs;
  no imports from `src/`.
- Loop write surface = `editable/` (whole dir). `frozen/` + `driver/` are integrity-checked.

## BUILD PLAN (concretized 2026-07-06 after the frozen layer sealed; sources located)

Write surface refinement: `editable/` is a small package, not one file (merging 1200 lines into
one file adds refactor risk for no control benefit; program.md scopes the loop to the dir):
- `editable/model.py` — vendor `src/models/dynamics_model.py` @ HEAD verbatim (663 lines;
  DynamicsModel + carrying rollout primitives). n_actions=5, n_memory/ff9 per config.
- `editable/rollout.py` — vendor `experiments/mem2mem/rollout.py` (286 lines, mem2mem sliding
  rollout loss; drop the src-path bootstrap).
- `editable/train.py` — adapted from `experiments/mem2mem/train_mem2mem.py` (260 lines): consume
  the LATENT CACHE (below) + actions from the procedural dataset; from-scratch; STOP on wall-clock
  budget (`--budget-s`, checked per step, save final ckpt at expiry); winner defaults =
  rollout-only mem2mem (`mem2mem_frac=1.0, no bootstrap`; FF9 optional — noFF9 proved sufficient
  on GridWorld); model dims sized in calibration (start 7-8M like GridWorld); **W=16 pinned**.
- `editable/adapter.py` — the eval bridge, CONTRACT: must export
  `make_adapter(ckpt_path, tokenizer, device) -> factory` whose adapter implements
  begin(prefix_frames, prefix_actions)/step(action)->frame(uint8 RGB): encode prefix frames to
  latents (frozen tokenizer), rollout_init (long-context prefill exists for T_ctx>W), then
  rollout_step per action + decode. The loop MAY edit this (its model may need a new inference
  path) — the frozen eval + window probe keep it honest.

Driver (`driver/`, NOT loop-editable; manifest.py already done):
- `latent_cache.py`: encode both datasets once with the frozen tokenizer (fp16, ~2.6 GB;
  16-frame chunks). PREREQ CHECK: window-invariance probe on colorfield (was verified for
  gridworld/memmaze; re-verify here — arbitrary-offset slicing must be safe) + cache hash
  recorded in runs/ metadata (not MANIFEST — cache is derived, regenerable).
- `window_probe.py`: the WINDOW PIN verifier (perturb a frame > W back; committed prediction
  must be bit-identical; run on the trained ckpt before scoring; violation -> score 0, flagged).
- `run_experiment.py`: manifest --check -> train under budget -> window probe -> frozen
  run_eval(make_adapter(...), privileged=False) -> append runs/experiments.jsonl
  {tag, git-diff of editable/, score+breakdown, gates, wall, seed} -> regenerate
  runs/leaderboard.md. Keep/discard rule (2σ from calibration) applied by the LOOP AGENT reading
  the log, enforced advisorily by the driver (flag `below_threshold`).

Calibration (go/no-go, per the main task body): budget sizing (~10-12 min local incl. eval;
H100 measured too), reference arms (mem2mem vs vanilla-tau0-style no-memory vs frozen baselines),
seed-noise floor x5-8 -> σ; REQUIRE dense mem2mem beats no-memory beyond bin 1 and doesn't
saturate; else difficulty dials (but frozen layer is SEALED — dials mean a v2 freeze, avoid).

Order: latent_cache (+window-invariance probe) -> editable/ vendoring -> smoke train (tiny budget,
4070) -> adapter + window probe -> run_experiment end-to-end with frozen baselines -> calibration
runs -> program.md -> report calibration numbers to Merlin (first overnight loop = separate go).

## STATUS (2026-07-06/07 session log)
- driver/manifest.py DONE (freeze recorded; caught its own first violation — cache_job.sh
  placed in frozen/ tripped the unrecorded-file check; moved to driver/).
- driver/latent_cache.py DONE; window-invariance probe PASSED (cos 0.9975; window-delta recon
  MSE 9.4x train / 12.6x val below recon error vs the 6x GridWorld acceptance precedent).
- Local cache build killed (laptop unplugged) -> CLUSTER JOB **416225** @ 686d1ed
  (colorfield-cache: idempotent datagen + probe + encode both sets + sha256). Pull latents back
  via pull_file.sh (direct copy, not git; ~2.6GB train + 130MB val), filenames
  latents-bd8f18857d71.npy (tokenizer-hash-keyed).
- editable/ COMPLETE @ 1839fbc: model.py + rollout.py (vendored) + train.py (budgeted
  from-scratch mem2mem trainer; W_PIN=16 module constant not a flag; n_actions=5; rollout-only
  winner defaults; --n-memory 0 = vanilla reference arm; BUDGET_STOP contract; --sched-steps for
  budget-sized LR horizon — NB default --epochs 50 dies inside warmup at 10-min budgets, size it
  at calibration) + adapter.py (make_adapter(ckpt, tokenizer, device); chunked prefix encode,
  rollout_init long-context prefill, rollout_step+T=1 decode). Built by delegated agent,
  verified: fake-cache budget smoke (54 steps, clean BUDGET_STOP, reloadable ckpt) + one real
  frozen-eval episode privileged=False (5.7s, 50 events). BUILD_NOTES.md has the transcript.
- CACHES LOCAL + VERIFIED: 416225 completed on ferranti (queued ~75min, ran <1h); both latents
  pulled via pull_file (direct rsync), sha256 == job log (train f305d8c9…, val 988ab78d…);
  probe numbers identical to local. (Redundant local build killed; partials deleted.)
- REAL SMOKE DONE (4070): train.py --budget-s 120 on the real cache → 7.75M params, trains,
  clean BUDGET_STOP. **CALIBRATION ANCHORS**: mem2mem step ≈ **14s/step local** (clip64, relay
  backward is the cost, as on memmaze) → a 10-min local budget ≈ 40 steps (likely too few);
  adapter+eval episode (prefix192+imag256) ≈ 26s local → FULL eval suite ≈ 100 min local —
  loop needs a REDUCED eval config (fewer policies×seeds, σ re-measured) and/or H100 backend
  (~4-6× both). Fidelity gate correctly reads the 9-step model as garbage (0.078).
- BUG (minor): train.py prints val(normal) nan under mem2mem_frac=1.0 (val path uses the
  normal loss it never trains) — fix to a mem2mem val or skip.
- REMAINING: fix val-nan -> speed levers (batch-size/clip_len profile; maybe torch.compile) ->
  driver/window_probe.py (corrected memory-pinned design) -> driver/run_experiment.py ->
  program.md -> calibration proper (budget sizing / reference arms / reduced-suite σ →
  keep-rule) -> go/no-go report with the 4070-vs-H100 trade-off numbers.

## CALIBRATION LOG (2026-07-06/07 night — "is 20 min enough?" per Merlin)
- driver/sheets.py: snake-prefix(192) + revisit(96) GT-vs-imagination strips, per-column
  on-screen cell acc (chance 0.2), map seeds 5/6, deterministic — comparable across runs.
- Arm 1 (7.75M, bs64, mixed n_ctx, 121 steps/20min): COMPLETELY LACKING — gray-pink mush,
  no appearance prior, no scroll, acc ~chance flat. runs/cal20/.
- bs128 @ 7.75M-small mixed n_ctx OOM'd on 8GB: a drawn n_ctx=4 batch holds 31 slide graphs
  for one backward -> added --fixed-n-ctx (always W_PIN=16: 7 slides, bounded VRAM, fewer
  fatter forwards; GPU util was 47% at bs64 small — Merlin's catch; 100% after bs128).
- Arm 2 (1.32M = 128/6/8, bs128, fixed n_ctx, 306 steps/20min, 3.93s/step): still lacking but
  DIFFERENT failure — collapsed toward dark/OUT mode; early-imagination acc 0.40-0.47 decaying
  to 0.0 (below chance = systematic OUT-overpainting). Loss still descending both arms.
  runs/cal20small/.
- VERDICT so far: 20 min on the 4070 is out of range at both sizes; need the steps-vs-quality
  KNEE. -> OVERNIGHT LOCAL RUN: 1.32M bs128 fixed-n-ctx, budget 8h, sched 7000 (~7300 steps
  expected), snapshots at 250/500/1000/2000/4000/6000 -> runs/calcurve/dynamics_stepN.pt.
  First launch (session-tied background task) was killed at ~step 500; RELAUNCHED ~00:50
  DETACHED via PowerShell Start-Process (survives session end; log:
  autoresearch/runs/calcurve/train.log, needs laptop on + plugged + logged in).
  MORNING: read train.log -> sheets + (reduced) comeback eval per snapshot -> the curve ->
  budget/backend decision with Merlin.
- **WINDOW PIN (red-team consequence; design CORRECTED 2026-07-06 late)**: the driver pins the
  model's temporal attention window (W=16) because the comeback scalar gives a window-W model
  ≈ the fraction of age bins ≤ W — without a pin the loop's cheapest move is "grow the window".
  **CORRECTION**: the original probe ("perturb a frame > W back ⇒ output must be bit-identical")
  is WRONG for memory models — the memory relay is SUPPOSED to carry far-frame information past
  the window; that is the research question. A good mem2mem model MUST react to far
  perturbations. Only the ATTENTION path must not reach past W. Corrected enforcement, three
  layers:
  1. Config check (hard): ckpt config.max_temporal_length ≤ 16, else score 0.
  2. Memory-pinned perturbation probe (behavioral, for the seeded architecture): run two prefix
     streams differing in ONE frame > W back; at every commit INJECT stream A's written-memory
     K/V into stream B's cache (forcing the memory channel identical); any remaining output
     divergence flows through non-memory attention to far frames = true window violation ⇒
     score 0. If cache introspection fails (loop restructured internals) ⇒ flag
     `window_probe: manual-review` on the leaderboard instead of silently passing.
  3. Legibility backstop: leaderboard always shows the per-age-bin curve — window-shaped gains
     (near bins up, far bins ~0) are visually distinct from memory-shaped gains; kept diffs get
     human review (the ultimate backstop, same as Karpathy's setup).

## Driver (deterministic code, not agentic)

`run_experiment(tag)`:
1. Re-hash `frozen/` + `driver/` against MANIFEST → mismatch ⇒ score := chance, flag `tampered`.
2. Train `editable/train.py` from scratch under a fixed WALL-CLOCK budget B (Karpathy-style: a
   change that slows the model trains fewer steps in its budget — compute cost auto-priced in).
3. Run the frozen comeback eval; the DRIVER computes the score (agent never self-reports).
4. Append to `runs/experiments.jsonl`: {tag, train.py diff vs current-best, score + breakdown
   (weighted / in-map / per-k / per-script), wall time, seed, budget}; update `runs/leaderboard.md`.

Keep/discard: keep iff score > best + 2σ_seed (σ from calibration below). Baseline re-run every
~15 experiments to detect drift/nonstationarity. Append-only log; kept versions of train.py
snapshotted under `runs/`.

## Backends

- **Local 4070** (default): loop runs overnight, owns the GPU exclusively.
- **Cluster H100** (Merlin leans this way): ONE long `submit_job.sh` job that runs many budgeted
  experiments inside it (the loop agent stays local; per-experiment sbatch round-trips are queue-
  latency-bound — measure before choosing). Reference point: GridWorld 50ep bs256 cached-latents
  ≈ 17 min on H100 ⇒ budget B ≈ 5–10 min plausible there.
- Decide backend + B during calibration; driver keeps a `run_backend` abstraction so both work.

## Calibration phase (REQUIRED before any loop runs; this is the go/no-go)

1. **Budget sizing**: scale the reference config (epochs/model dims/data subset) until
   train + eval ≈ 10–12 min on the 4070 (measure H100 too if a socket is up).
2. **Reference arms under budget**: dense mem2mem-rollout vs no-memory (tau0-anchor vanilla).
   Require HEADROOM (dense comfortably < 1.0 on the comeback scalar) and DISCRIMINATION
   (dense ≫ vanilla ≫ chance). If dense saturates → raise the env difficulty dials (map N /
   palette / view) BEFORE the frozen layer is hashed — this is why calibration precedes freezing.
3. **Seed-noise floor**: identical config × 5–8 seeds → σ of the scalar; keep-threshold := 2σ.
   If σ swamps plausible effect sizes, STOP and report to Merlin (the loop is not viable as
   configured — do not run it anyway).
4. Eval cost check: branch-forwards only on event frames; confirm eval ≤ ~2–3 min at budget scale.

Record everything in `autoresearch/runs/calibration/NOTES.md` + one EXPERIMENTS.md line.

## program.md (v1 sketch — Merlin owns/iterates this file)

- Edit ONLY `editable/train.py`; one idea per experiment; from-scratch each run (no checkpoints).
- Read `runs/experiments.jsonl` + leaderboard before proposing; build on kept changes.
- Never touch `frozen/`, `driver/`, the dataset, or the scorer (hash-enforced anyway).
- Negative results get one log line too; no re-running until a change is made.

## Done when

Calibration numbers recorded (budget config, σ, reference-arm scores, backend choice) and
reported to Merlin. The FIRST overnight loop is a separate explicit go from Merlin — not part
of this task.
