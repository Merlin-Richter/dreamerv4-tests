# REPO_MAP — where things live

Orientation map for the codebase. Pairs with `CLAUDE.md` (architecture/commands) and the protocol
state files (`ORIENT/GOAL/DECISIONS/EXPERIMENTS/BOARD/ESCALATIONS`). Updated 2026-06-16 (T-019 reorg,
D-030/D-031).

## Top level
| Path | What |
|------|------|
| `GOAL.md` | Hypotheses (H1–H3) + success criteria. Owned by Merlin. |
| `DECISIONS.md` | Append-only decision log (D-NNN). |
| `EXPERIMENTS.md` | One-line index per experiment; details in `experiments/EXP-NNN/`. |
| `ORIENT.md` / `BOARD.md` / `ESCALATIONS.md` | Live situation / task board / open questions. |
| `IDEAS.md` | Method ideas backlog (FF7/FF9/C1 designs, etc.). |
| `HOWTO/` | Stable ops knowledge (cluster quirks, RoPE/KV caveats, venv/CUDA note). |
| `tasks/` | Per-task plans/specs/verdicts (T-NNN). |
| `CLAUDE.md` | Codebase guide for Claude Code (architecture, commands, conventions). |
| `requirements.txt` | Python deps (Python 3.11+; CUDA wheels via the venv). |
| `src/` | All source code (see below). |
| `experiments/` | Per-experiment configs, run scripts, NOTES, results, readouts. |

**Local-only artifacts (gitignored — NOT in version control):**
- Datasets: `bouncing.npy`, `occluded.npy`, `gridworld.npy` (+ `_actions`/`_states`, gridworld also `_colors`) at repo root.
- Shared checkpoints under **`checkpoints/<env>/`** (D-032 — env is explicit in the path; a model does NOT transfer across envs): `checkpoints/occluded/tokenizer.pt` (frozen tokenizer, was `trained_autoencoder.pt`) + `dynamics_vanilla.pt` (retired, was `my_dynamics.pt`); `checkpoints/bouncing/dynamics.pt` (was `dynamics_bouncing.pt`) + `tokenizer.pt`; `checkpoints/gridworld/` (in progress).
- Per-experiment checkpoints under `experiments/EXP-NNN/*.pt`. `venv/`, `wandb/`.
- Because datasets/checkpoints are gitignored, training scripts reference them by path; **do not relocate
  them without updating the `--frames/--actions/--tokenizer/--checkpoint` defaults + every caller.**

## `src/` — current layout (reorganized by concern, T-019 / D-030, 2026-06-16)
Imports use packages off `src/` on `sys.path` (`from models.X import …`, `from evals.X import …`,
`from envs.X import …`). A file under `src/<dir>/` bootstraps with `_SRC = parents[1]` (or `parents[2]`
one level deeper) and inserts `_SRC`.

| Path | Role | Notes |
|------|------|-------|
| `src/models/` | **Model architectures** | `dynamics_model.py` (active research), `tokenizer.py` (temporal tokenizer = the frozen `checkpoints/occluded/tokenizer.pt`), `single_image_ae.py` (baseline), `lm.py` (standalone char-LM) |
| `src/training/` | **Training scripts** | `train_dynamics.py`, `train_tokenizer.py`, `train_single_image_ae.py`, `train_lm.py` |
| `src/envs/` | **Environments** (steppable sims behind `BaseEnv`) | `base.py` (the ABC), `occluded_bouncing.py` (`OccludedBouncingEnv`), `bouncing.py` (`BouncingEnv`), `gridworld.py` (`GridWorldEnv` — discrete 8×8 grid memory env, D-032) |
| `src/datagen/` | **Dataset generation + inspection** (drives envs → `.npy`) | `generate_occluded.py`, `generate_bouncing.py`, `generate_gridworld.py`, `example_read.py`. (Named `datagen`, NOT `data/`, to dodge the `.gitignore` `data/` artifact rule.) |
| `src/evals/` | **Evaluation toolbox** (common Eval interface) | `base.py` (Eval ABC + REGISTRY + `load()`), `probe_env.py` (FROZEN episode builder), `revisit/` (FROZEN spine `probe.py` @ 5503e75 + RevisitEval), `position_consistency/` (FROZEN `consistency.py`), `motion/` (working curves + MotionEval), `rollout_view/` (`ab_view.py`) |
| `src/tests/` | **Gate tests** (the dynamics-model safety net) | `test_kv_cache/stream_cache/ff7_smoke/ff9_smoke/multistep_smoke.py`, `test_gridworld.py` (GridWorld env, D-032) |
| `src/interactive/` | **Interactive viewers** | `play_dynamics.py`, `latent_explorer/`, `lm_inference.py` |
| `src/wlog.py` | Lightweight W&B logger (no-op unless `--wandb`) | |

**FROZEN spine** (revisit/position-consistency measurement logic): `src/evals/probe_env.py`,
`src/evals/revisit/probe.py`, `src/evals/position_consistency/consistency.py` — frozen @ commit
5503e75; any change is a logged decision (GOAL §8). The T-019 move kept them byte-identical except
import/bootstrap lines (verified by old→new diff).

### Gate tests
`src/tests/test_{kv_cache,stream_cache,ff7_smoke,ff9_smoke,multistep_smoke}.py` — run (CPU OK) after
any change to `models/dynamics_model.py`.

## `experiments/EXP-NNN/`
Each holds: `config.txt`/`config.yaml`, `run.sh` (provenance — the exact command), `NOTES.md`
(purpose + reconciliation), `results.json`, and readout artifacts (`*.png`, `*.html`). NEW experiment
scripts import shared logic from `src/evals/` rather than redefining it. **Historical experiment
scripts are FROZEN to the commit they ran at (D-031)** — they import old paths and are not rewired;
to rerun one, `git checkout` its commit.

## Concept → location quick index
- **Models** → `src/models/*` (`dynamics_model.py`, `tokenizer.py`, `single_image_ae.py`, `lm.py`)
- **Training** → `src/training/train_*.py`
- **Environments** → `src/envs/` (subclass `BaseEnv`)
- **Dataset generation** → `src/datagen/generate_*.py`
- **Data (datasets)** → repo-root `*.npy` (gitignored, local)
- **Evals** → `src/evals/` (registry: `import evals; evals.discover()`; frozen spine under `revisit/` + `position_consistency/`)
- **Readouts** → `experiments/EXP-NNN/` (PNG/HTML views + NOTES.md decisive read)
- **Interactive** → `src/interactive/{play_dynamics.py,latent_explorer/,lm_inference.py}`

## Adding a new env or eval (the structure's purpose)
- **New env:** subclass `envs.base.BaseEnv` (implement `reset`/`step`; expose hidden state via
  `hidden_state()` for measurement only). Add a `datagen/generate_*.py` if it needs a dataset.
- **New eval:** add `src/evals/<name>/` with an `Eval` subclass (`score()` cheap scalars,
  optional `report()` rich artifacts; declare `compatible_envs`); `register()` it and add to
  `discover()`. Mid-run-cheap evals pass `midrun=True`.
