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
- **WINDOW PIN (red-team consequence)**: the driver contract pins the model's temporal context
  window (e.g. W=16 frames) and VERIFIES it (probe: perturb a frame at distance > W from the
  target; the prediction must be bit-identical — if changing out-of-window input changes the
  output, the window claim is violated → score 0). Rationale: the comeback scalar gives a
  window-W model ≈ the fraction of age bins ≤ W (measured curve in the colorfield task), so
  without a pin the loop's cheapest score move is "grow the window" — the opposite of the
  memory-token research question. With the pin, every bin beyond W must come from a carried
  memory mechanism.

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
