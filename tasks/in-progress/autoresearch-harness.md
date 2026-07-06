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
- Loop write surface = `editable/train.py` ONLY. `frozen/` + `driver/` are integrity-checked.
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
