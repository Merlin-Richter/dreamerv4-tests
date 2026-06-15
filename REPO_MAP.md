# REPO_MAP — where things live

Orientation map for the codebase. Pairs with `CLAUDE.md` (architecture/commands) and the protocol
state files (`ORIENT/GOAL/DECISIONS/EXPERIMENTS/BOARD/ESCALATIONS`). Updated 2026-06-16 (D-028).

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
- Datasets: `bouncing.npy`, `occluded.npy`, `occluded_actions.npy`, `occluded_states.npy` (repo root, ~2.3 GB each).
- Shared checkpoints: `trained_autoencoder.pt` (frozen tokenizer), `my_dynamics.pt` (retired), `dynamics_bouncing.pt`.
- Per-experiment checkpoints under `experiments/EXP-NNN/*.pt`. `venv/`, `wandb/`.
- Because datasets/checkpoints are gitignored, training scripts reference them by path; **do not relocate
  them without updating the `--frames/--actions/--tokenizer/--checkpoint` defaults + every caller.**

## `src/` — current layout
| Path | Role | Notes |
|------|------|-------|
| `src/A_LM/` | Char-level LM (standalone, **not** in the video pipeline) | `model.py`, `train.py`, `inference.py` |
| `src/B_single_image_auto_encoder/` | Single-frame AE (baseline) | `video_auto_encoder.py`, train script |
| `src/C_multi_image_auto_encoder/` | **Temporal tokenizer** (the frozen `trained_autoencoder.pt`) | `video_auto_encoder.py`, train script |
| `src/D_dynamics_model/` | **Dynamics model** (the active research code) | `dynamics_model.py`, `train_dynamics_model.py`, `play_dynamics_checkpoint.py`, `test_*.py` gate tests |
| `src/data_generators/` | **Environments / data generation** | `bouncing_objects.py`, `occluded_bouncing.py`, `load_data.py` |
| `src/probe/` | **FROZEN probe spine** (revisit/position consistency) | `revisit_probe.py`, `probe_env.py`, `position_consistency.py` — frozen @ 5503e75; changes are logged decisions (GOAL §8) |
| `src/eval/` | **Working eval toolbox** (non-frozen) | `motion.py` — shared motion curves/A-B helpers extracted from experiments (D-028) |
| `src/test/latent_explorer/` | Interactive tokenizer-latent browser UI | `run.py` |
| `src/wlog.py` | Lightweight W&B logger (no-op unless `--wandb`) | |

### Gate tests (the safety net for `src/D_dynamics_model`)
`test_kv_cache.py`, `test_stream_cache.py`, `test_ff7_smoke.py`, `test_ff9_smoke.py`,
`test_multistep_smoke.py` — run these (CPU OK) after any change to `dynamics_model.py`.

## `experiments/EXP-NNN/`
Each holds: `config.txt`/`config.yaml`, `run.sh` (provenance — the exact command), `NOTES.md`
(purpose + reconciliation), `results.json`, and readout artifacts (`*.png`, `*.html`). Experiment
scripts should import shared logic from `src/eval/` and `src/probe/` rather than redefining it.

## Concept → location quick index (what Merlin asked for)
- **Models** → `src/{D_dynamics_model,C_multi_image_auto_encoder,B_single_image_auto_encoder,A_LM}/*` (model.py / video_auto_encoder.py / dynamics_model.py)
- **Training** → `src/<component>/train_*.py`
- **Environments / data generation** → `src/data_generators/`
- **Data (datasets)** → repo-root `*.npy` (gitignored, local)
- **Evals** → `src/probe/` (frozen spine) + `src/eval/` (working toolbox)
- **Readouts** → `experiments/EXP-NNN/` (PNG/HTML views + NOTES.md decisive read)
- **Interactive** → `src/D_dynamics_model/play_dynamics_checkpoint.py`, `src/test/latent_explorer/`

## Planned cleanup (staged — see `tasks/T-019-repo-reorg-plan.md`)
The `src/{A,B,C,D}_*` directories mix *model* + *training* per pipeline stage. A clearer target
(`src/models/`, `src/training/`, `src/envs/`, plus the existing `src/probe` + `src/eval`) is drafted
in T-019 but **deferred for Merlin's approval** — it's a high-fanout move (every experiment script's
`sys.path` + `CLAUDE.md` + gate tests) and not safe to do unsupervised. D-028 did the low-risk part
(extract reused evals → `src/eval/`, hygiene, this map).
