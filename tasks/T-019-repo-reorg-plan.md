# T-019 — Repo reorg plan (DEFERRED, needs Merlin approval)

Status: **PROPOSED, not executed.** D-028 did the safe subset (extract reused evals → `src/eval/`,
remove scratch, write `REPO_MAP.md`). The deeper model/training/env relocation below is high-fanout
and was deliberately NOT done unsupervised overnight ("don't break anything"). This is the ready-to-
execute plan for when Merlin signs off.

## Why defer the big move
`src/{A,B,C,D}_*` mix model + training per pipeline stage. The cleaner target separates concerns, but
`dynamics_model.py` / `video_auto_encoder.py` / `train_*.py` are imported by:
- the 5 gate tests (`src/D_dynamics_model/test_*.py`),
- `play_dynamics_checkpoint.py`, `latent_explorer/run.py`,
- ~10 experiment scripts via hardcoded `sys.path.insert(".../src/D_dynamics_model")` + `from dynamics_model import ...`,
- `CLAUDE.md`'s entire "Key Files" section.
A subtle broken import could sit undetected until Merlin reruns an old experiment. High blast radius.

## Proposed target layout
```
src/
  models/      dynamics_model.py            (← D_dynamics_model)
               tokenizer.py                 (← C_multi_image_auto_encoder/video_auto_encoder.py)
               single_image_ae.py           (← B_single_image_auto_encoder/video_auto_encoder.py)
               lm.py                        (← A_LM/model.py)
  training/    train_dynamics.py            (← D_dynamics_model/train_dynamics_model.py)
               train_tokenizer.py           (← C_.../train_autoencoder_bouncing.py)
               train_single_image_ae.py     (← B_.../train_autoencoder_bouncing.py)
               train_lm.py                  (← A_LM/train.py)
  envs/        bouncing_objects.py, occluded_bouncing.py, load_data.py   (← data_generators/)
  probe/       (UNCHANGED — frozen spine)
  eval/        motion.py (+ future shared A/B drivers)                   (done in D-028)
  interactive/ play_dynamics.py             (← D_.../play_dynamics_checkpoint.py)
               latent_explorer/             (← src/test/latent_explorer/)
  tests/       test_*.py                    (← D_dynamics_model/test_*.py)
  wlog.py
data/          *.npy        (optional; requires updating train-script path defaults)
checkpoints/   shared *.pt  (optional; same caveat)
```

## Safe migration recipe (per moved module)
1. `git mv` the file to its new home; rename only where it removes ambiguity (the two
   `video_auto_encoder.py` differ → `tokenizer.py` vs `single_image_ae.py`).
2. Update every importer's `sys.path` + `from X import` (grep first: `from dynamics_model`,
   `from video_auto_encoder`, `from train_dynamics_model`, path inserts).
3. Run ALL gate tests (`test_kv_cache/stream_cache/ff7_smoke/ff9_smoke/multistep_smoke`) — must stay green.
4. CPU-smoke one representative experiment driver per touched module (e.g. `probe_multistep`, `ab_eval`,
   an EXP-017 eval) — must reproduce prior numbers.
5. Update `CLAUDE.md` Key-Files table + `REPO_MAP.md` in the same commit.
6. Commit per module (small, revertible steps) — never one big-bang commit.

## Decisions for Merlin
- Keep the meaningful A/B/C/D pipeline-stage naming, or go to models/training split? (The stage
  letters encode the pipeline order; the split encodes concern. Can keep stage names as a comment.)
- Move datasets/checkpoints into `data/` + `checkpoints/` (cleaner root, but touches every script's
  path defaults), or leave at root (less churn)? Recommend: leave for now; lowest value/risk.
- Provide back-compat shims at old paths (`from src.models.dynamics_model import *`) to avoid touching
  done-experiment scripts, or update all callers cleanly? Shims = less churn but more clutter.
