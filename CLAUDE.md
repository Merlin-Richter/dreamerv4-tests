# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research implementation of a video dynamics model in the **Dreamer-4** lineage, used to study one
question: can a world model **retain hidden / off-screen state past its short latent window**? Our
extension over vanilla Dreamer-4 is per-timestep **memory tokens** that should carry that state.

The codebase is **spec-driven**: every source file under `src/` has a one-to-one spec in `specs/<same
path>.md`, and **the specs are authoritative** (Merlin owns them). Code is generated/maintained to match
its spec — `specs/` is the canonical map of what exists and how it must behave. The agent harness lives
under `agent/` (`OPERATING.md`, `ORIENT.md`, `EXPERIMENTS.md`) and `tasks/` (folder-per-state task files).

The active pipeline is **GridWorld → frozen Tokenizer → Dynamics model**:

- **Tokenizer** (`models/tokenizer.py`): frozen temporal video autoencoder; compresses frames to latents.
- **Dynamics model** (`models/dynamics_model.py`): causal transformer that predicts the next frame's
  clean latents via shortcut forcing, with optional memory tokens.

> Note: this `src/` was rebuilt clean from the specs (a ~5× shrink). Earlier experiment lines — FF7
> register-memory, multistep DAgger, FF9 rollout-training, snapshot/streaming inference, the
> occluded-bouncing and DVD-bouncing envs, the LM and single-image-AE baselines, the motion/revisit/
> position-consistency evals — were intentionally dropped. Their history is in `agent/EXPERIMENTS.md`
> and `git`. If you find a reference to them in code, it is a leftover bug.

## Architecture & Data Flow

```
GridWorld frames (B, T, 64, 64, 3) uint8 BGR
  ↓
[models.tokenizer]  (FROZEN)
  • patchify 8×8 → spatial + temporal (3×[spatial,temporal,spatial], causal) attention
  • MAE patch-dropout (train only); restricted cross-attention bottleneck
  • → latents (B, T, n_latents=4, bottleneck_dim=64)
  ↓
[models.dynamics_model]  (TRAINABLE)
  • per-frame latent denoising via block-causal attention, shortcut forcing
  • optional memory tokens carry hidden state past the window
  • → predicts clean next-frame latents; autoregressive carrying rollout at inference
```

### Tokenizer (`models/tokenizer.py`) — spec: `specs/models/tokenizer.md`
- Patchifies frames into non-overlapping `patch_size=8` patches + learned position embeddings.
- Spatial layers: full self-attention within a frame over patches + `n_latents` learned latent tokens.
  Temporal layers (every 3rd, the middle of each [spatial,temporal,spatial] triple): causal across time with RoPE.
- **Restricted cross-attention IS the bottleneck**: in the encoder latent tokens attend to patches but
  patches do not attend to latents; in the decoder patches attend to latents but not vice-versa.
- **MAE patch-dropout** (train only) forces the decoder to use the latents, preventing mean-image collapse.
- **Learnable per-head attention temperature** (`logit_scale`, init log(4), clamp log(100)) — with QK-norm,
  the textbook 1/√d scale collapses latents to the mean image. Load-bearing; do not drop.
- Frozen after training. BGR in/out, [0,1]. `n_latents=4, bottleneck_dim=64, patch_size=8`.

### Dynamics model (`models/dynamics_model.py`) — spec: `specs/models/dynamics_model.md`
**Token layout per frame** (sequence axis): `[action | latents | registers | (memory) | shortcut]`.
Registers are plain learned scratch tokens; memory tokens are present only when `n_memory>0`.

**Transformer**: 2-D, separate space/time attention, pre-RMSNorm, RoPE (time only), SwiGLU, QK-norm,
attention-logit soft-capping, learnable per-head logit scale. Spatial = unmasked within a frame;
temporal = causal in time, **position-wise per token slot** (each slot, incl. each memory slot, is its
own causal channel across time), applied every 3rd layer (the middle of each [spatial,temporal,spatial] triple).

**Shortcut forcing loss** (`loss`): diffusion forcing (per-frame signal level τ) + shortcut models
(step size d=1/K). **x-prediction** (predicts clean `ẑ₁`, not velocity). Finest step → flow MSE; coarser
steps → bootstrap loss distilling two d/2 steps with stop-grad, scaled to x-space by (1−τ)². Ramp weight
`w(τ)=(1−ramp_min)τ+ramp_min`.

**Memory (FF9 sufficiency loss)**: when `n_memory>0` and `config.ff9_k>0`, for each frame the path
latents are set to τ=0 (pure noise — no scene in the latents) and the **written** memory token is
injected; the model must reconstruct the next 1..ff9_k frames **from memory alone**. The FF9 term is
normalized to the diffusion magnitude by a gradient-detached scaler, and its gradient flows back through
the mechanism that constructs the memory tokens.

**Inference (carrying rollout)** — a single `generate(context, n_generate, K=None, action_idx=None)`:
- Autoregressive over a sliding window of `max_temporal_length`, each frame denoised from noise in K=4
  shortcut steps, context held near-clean at `context_signal` (≈0.9).
- **KV cache across TIME**: a frame's K diffusion steps only *read* the cache of committed past frames; a
  dedicated **5th forward pass** commits the frame, re-presented near-clean with its **written memory
  token**, into the cache. RoPE is computed at the **absolute rollout index** so cached K/V is never
  re-rotated; window eviction is a pure slice (see `HOWTO/rope_kv_cache_caveat.md`).
- **Memory relay** = read the old memory tokens' cached K/V and write new ones each step (NOT threading a
  frozen activation forward). `n_memory=0` ⇒ vanilla (plain rollout, no carry).
- `rollout_init` / `rollout_step` are the underlying primitives. `rollout_step(..., commit=False)` is a
  **read-only branch** that predicts a frame without mutating the carried cache — this is what the recall
  eval uses to peek a reveal while the occluded rollout continues.

### GridWorld env (`envs/gridworld.py`) — spec: `specs/envs/gridworld.md`
Discrete **6×6** memory env: a solid-color background + a single square (4 palette colors), the square
steps 1 cell/tick in one of 8 directions with wall reflection. A per-frame **curtain** action occludes
the whole frame (flat gray) — physics runs behind it. Geometry: `3px border + 6×8px cells + 5×2px lines
= 64`; cell `i` interior at `3+10·i`. The 10px cell stride is deliberately NOT a multiple of the
tokenizer's 8px patch (anti-overfit). Recall is exact: 6×6 cell (chance 1/36) + 4-way color. State =
`[col, row, dcol, drow, curtain]`; `.color`/`hidden_state()` are measurement-only (never a model input).
Channel order is **BGR end-to-end** (RGB only for on-screen display).

### GridWorldV2 env (`envs/gridworldv2.py`) — spec: `specs/envs/gridworldv2.md` (DRAFT)
Action-driven successor: **7 actions** (0=reveal, 1=hide → curtain LATCH, square doesn't move on
toggle ticks; 2..5 = up/down/left/right CLAMPED at walls; 6=stay), no autonomous physics — under
occlusion the hidden position is a nonlinear function of the action stream (memory must integrate
actions, not extrapolate ballistics). Geometry/rendering imported from v1 ⇒ the v1 readout is
exact on v2 frames, and the **frozen v1 tokenizer works unchanged** (verified readout-exact on
recon). State `[col,row,curtain]`. Datagen: `datagen/generate_gridworldv2.py` (alternating
revealed/occluded runs, shared movement-run policy) → `data/gridworldv2*.npy` (n_actions=7
auto-detected). Recall: `evals/gridworldv2/recall.py` (branch-after-commit alignment: k = occluded
MOVEMENT actions integrated; oracle via measurement-only `render_revealed()`). Gate:
`tests/test_gridworldv2.py`. All v2 specs are DRAFT (Merlin sign-off pending).

### Recall eval (`evals/gridworld/{recall,readout}.py`) — specs under `specs/evals/gridworld/`
- `readout.read_square(frame)`: closed-form, exact readout of (col, row, color, bg) from one frame —
  background = median cell color, square = farthest cell, colors = nearest of 4 palette. `is_occluded` =
  **no black-grid-line pixel** (no pixel with all channels < 25). Pure numpy, identical on true & predicted
  frames.
- `recall.recall(model, tokenizer, *, n_ctx, max_k, n_rollouts=64, K=4, device)`: the env-based memory
  scorer. Per seed: show `n_ctx` revealed context frames, then one long OCCLUDED rollout; at each scored
  `k` branch a **read-only** reveal, decode it, and score the square against the env's independently-
  advancing true state. Baselines through the same readout: `oracle` (ceiling, must read 1.0), `copy_last`
  (no-memory reference), `chance`. **This eval is the result-defining spine — memory claims must show here,
  not on reconstruction loss.**
- `sheets.py` (`occlusion_sheet`/`normal_sheet`/`save_sheet`): the QUALITATIVE companion to `recall` —
  cv2 filmstrip PNGs (TOP = ground truth / true underlying square, BOTTOM = model rollout / belief) you
  eyeball. The occlusion sheet uses the same carried `rollout_init`/`rollout_step(commit=False)` branching
  rollout `recall` scores (so it carries memory for FF9 models). Illustrates; never decides — `recall` does.

### Shared components
- `wlog.py`: lightweight W&B logger, no-op unless `--wandb`.
- `envs/base.py`: `BaseEnv` interface (`reset`/`step`/measurement-only `hidden_state`); BGR + privileged-
  state contracts.
- `datagen/generate_gridworld.py`: dataset writer + cv2 viewer.
- Attention building blocks (tokenizer & dynamics): QK-norm + RMSNorm, learnable per-head logit scale,
  soft-cap `tanh(logits/cap)·cap`, RoPE on temporal axes, SwiGLU MLP.

## Development Commands

Run from the repo root (scripts bootstrap `src` onto the path). Datasets live under `data/` (gitignored);
checkpoints under `checkpoints/<env>/` (gitignored). Almost always pass `-u` so output isn't buffered.

> **Local GPU:** use the repo venv `venv/Scripts/python.exe` — it has a CUDA torch build
> (`torch 2.12.0+cu126`) and sees the RTX 4070 Laptop GPU, so evals/training run on the GPU. The bare
> system `python` is a CPU-only torch build (`+cpu`, `cuda.is_available()==False`) — using it silently
> falls back to CPU. All code already does `device = "cuda" if torch.cuda.is_available() else "cpu"`, so
> just invoke it via the venv python to get the 4070.

```bash
# 1. Generate the GridWorld dataset  ->  data/gridworld.npy (+ _actions/_states/_colors)
python -u src/datagen/generate_gridworld.py --n_episodes 1000
python -u src/datagen/generate_gridworld.py --play     # interactive (you drive the curtain)
python -u src/datagen/generate_gridworld.py --debug    # preview one scheduled episode

# 2. Train the tokenizer (then it is FROZEN for the dynamics model)
python -u src/training/train_tokenizer.py --epochs 20 --batch-size 16
python -u src/training/train_tokenizer.py --lpips                 # optional LPIPS perceptual loss
python -u src/training/train_tokenizer.py --test-checkpoint --checkpoint checkpoints/gridworld/tokenizer.pt

# 3. Train the dynamics model on the frozen tokenizer.
# --frames and --tokenizer are REQUIRED (no defaults); --checkpoint defaults to none (train from
# scratch, no save) — pass it to save/resume.
python -u src/training/train_dynamics.py --epochs 50 --batch-size 32 --context-length 16 \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics.pt
# Memory model: --ff9 K enables the FF9 sufficiency loss; --n-memory M (default 4) sets memory tokens.
python -u src/training/train_dynamics.py --ff9 3 --n-memory 4 --context-length 16 --seed 0 \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics_ff9.pt
# Try a NEW loss/training-flow without editing the spec-backed model: pass an experiment-local
# DynamicsModel subclass via --model-module (see experiments/README.md for the workflow).
#   --model-module experiments/EXP-NNN/model.py:DynamicsModelEXP_NNN
# Interactive rollout viz (--checkpoint + --frames required):
python -u src/training/train_dynamics.py --test-checkpoint \
  --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworld/dynamics.pt

# 4. Inspect interactively (keys: 0=reveal, 1=occlude, r=reset, q=quit)
python -u src/interactive/play_dynamics.py \
  --checkpoint checkpoints/gridworld/dynamics.pt --tokenizer checkpoints/gridworld/tokenizer.pt \
  --frames data/gridworld.npy

# Memory Maze, playable INSIDE the world model — the pygame twin of the real playable maze
# (external/memory-maze/gui/run_gui.py, see common_commands.txt): same keymap (arrows drive,
# space=pause, backspace=reset, tab=speedup, esc=quit), but every frame is the dynamics model's
# carrying rollout in tokenizer-latent space on the local GPU. Reset replays a full context window
# of REAL held-out episode frames (green border, committed via rollout_init), then you play.
# ~9 fps on the 4070 (> the 6 fps target). --selftest N = headless smoke mode. Needs pygame
# (installed in the repo venv). Spec: specs/interactive/play_memmaze.md.
venv/Scripts/python.exe -u src/interactive/play_memmaze.py \
  --checkpoint checkpoints/memmaze/dynamics_vanilla.pt --tokenizer checkpoints/memmaze/tokenizer.pt

# Qualitative rollout sheets -> outputs/sheets/{sheet_occlusion,sheet_normal}.png (cheap, CPU OK; the
# default --out-dir outputs/sheets/ is gitignored). Occlusion sheet needs no dataset (built from the env);
# --frames is only used for the normal sheet. Pass --out-dir experiments/EXP-NNN to keep a run's sheets.
python -u src/evals/gridworld/sheets.py --kind both \
  --checkpoint checkpoints/gridworld/dynamics_ff9.pt --tokenizer checkpoints/gridworld/tokenizer.pt \
  --frames data/gridworld.npy

# Recall curves: eval each checkpoint -> JSON (outputs/recall/), then overlay any set of runs into a
# compare figure. Eval-once-to-JSON, plot/compare-many (no re-eval to re-plot). --max-k may exceed the
# window (the rollout slides/evicts); --window forces a shorter sliding window than training (e.g. 8).
python -u src/evals/gridworld/recall.py --max-k 32 [--window 8] \
  --checkpoint checkpoints/gridworld/dynamics_ff9.pt --tokenizer checkpoints/gridworld/tokenizer.pt
python -u src/evals/gridworld/plot_recall.py \
  --series "vanilla|outputs/recall/recall_dynamics_vanilla.json|tab:red" \
  --series "FF9 (carry)|outputs/recall/recall_dynamics_ff9.json|tab:green"

# GridWorldV2 (action-driven; same tokenizer as v1)
python -u src/datagen/generate_gridworldv2.py --n_episodes 5000       # -> data/gridworldv2.npy
python -u src/evals/gridworldv2/recall.py --max-k 32 [--window 8] \
  --checkpoint checkpoints/gridworldv2/dynamics.pt --tokenizer checkpoints/gridworld/tokenizer.pt

# Gate tests (CPU OK)
python -u src/tests/test_gridworld.py          # env geometry / physics / curtain schedule
python -u src/tests/test_gridworldv2.py        # v2 semantics / readout compat / recall instrument
python -u src/tests/test_gridworld_eval.py     # readout exact + recall instrument (oracle == 1.0)
python -u src/tests/test_dynamics.py           # forward / loss+FF9 grad / carrying generate / read-only branch
```

W&B is optional on every trainer via `wlog` flags (`--wandb --wandb-project … --wandb-entity … --wandb-name
… --wandb-tags …`), or `$WANDB_ENTITY`/`$WANDB_PROJECT` env defaults. Checkpoints stay local.

### Cluster / remote runs
Big training/eval runs go to the GPU cluster (ferranti H100s / galvani A100s). **All** cluster access goes
through the `scripts/` wrappers (`sync_code.sh`, `submit_job.sh`, `pull_results.sh`, …) — never raw
ssh/scp/rsync/sbatch (protocol §6). Asymmetric transport: **code goes up only via GitHub** (`sync_code.sh`
does a remote `git fetch`+`checkout`, so commit+push first), while **results/checkpoints come straight back
to local** via `pull_results.sh` (rsync; `*.pt` only with `--what checkpoints|all`). The wrappers **must run
in WSL** (the ssh ControlMaster socket is WSL-namespaced) and need a master socket Merlin opens
interactively first. Local 4070 training stays in Windows/Git-Bash; only cluster orchestration is WSL.
**Read more:** `scripts/README.md` (verbs + error contract) and `HOWTO/cluster.md`.

## Config Dataclasses

**AutoEncoderConfig** (`models/tokenizer.py`): `embedding_dim, depth, n_heads, mlp_ratio, patch_size,
img_input_H/W, n_latents (4), bottleneck_dim (64), max_temporal_length, mae_min/max_mask` + stability/
soft-cap fields.

**DynamicsModelConfig** (`models/dynamics_model.py`):
- Must match the tokenizer: `bottleneck_dim`, `n_latents`.
- Transformer: `embedding_dim, depth, n_heads, gqa_groups (1 = plain MHA; >1 = grouped-query
  attention, KV cache shrinks by this factor), mlp_ratio, max_temporal_length, n_registers`.
- Shortcut forcing: `max_sampling_steps` (K_max, power of two), `inference_steps` (K, typically 4),
  `context_signal` (τ_ctx ≈ 0.9), `ramp_min`.
- Conditioning / memory: `n_actions` (0 = unlabeled), `n_memory` (0 = vanilla), `ff9_k` (FF9 lookahead).

## Key Files

| File | Role |
|------|------|
| `src/models/tokenizer.py` | Frozen temporal autoencoder (latent space) |
| `src/models/dynamics_model.py` | Dynamics transformer + shortcut/FF9 loss + carrying rollout |
| `src/training/train_tokenizer.py` | Tokenizer training (stability: AdamW β2≈0.95, grad-spike skip, best-by-recon ckpt) |
| `src/training/train_dynamics.py` | Dynamics training on frozen latents (`--ff9`/`--n-memory` for memory; `--model-module` for experiment subclasses) |
| `src/interactive/play_dynamics.py` | Interactive single-frame viewer (single `model.generate` path) |
| `src/interactive/play_memmaze.py` | Playable Memory Maze rendered by the dynamics rollout (pygame, local GPU) |
| `src/envs/{base,gridworld}.py` | Env interface + GridWorld memory env |
| `src/datagen/generate_gridworld.py` | GridWorld dataset writer + viewer |
| `src/evals/gridworld/{readout,recall}.py` | Closed-form frame readout + the memory recall scorer |
| `src/evals/gridworld/sheets.py` | Qualitative rollout-sheet PNGs (occlusion belief / free-run), cv2-only |
| `src/evals/gridworld/plot_recall.py` | Overlay recall-curve JSONs into one 2×2 compare figure (matplotlib) |
| `src/wlog.py` | W&B logger (no-op unless `--wandb`) |
| `specs/**` | Authoritative one-spec-per-source-file map (Merlin owns these) |
| `experiments/**` | The lab — speculative, NOT spec-backed; try ideas here without touching `src/` (see `experiments/README.md`) |

## Notes on Design Choices
- **MAE dropout (tokenizer)**: stops the decoder from ignoring patches / reconstructing the mean image.
- **Shortcut forcing (dynamics)**: efficient inference (K=4 steps, not 128) while training a per-frame
  diffusion signal; x-prediction keeps long rollouts stable.
- **Frozen tokenizer**: a stable latent space; the dynamics model only learns transitions. Dynamics dims
  must match the tokenizer's `n_latents`/`bottleneck_dim`.
- **Memory tokens**: the carrier for hidden state past the latent window — relayed by read-old/write-new
  through the carrying KV cache, trained to be sufficient by the FF9 loss. Registers are plain scratch.
- **RoPE on time only + absolute-index rotation**: lets rollouts exceed the cos/sin table and slide the
  KV window by pure slicing.

A trained GridWorld tokenizer lives at `checkpoints/gridworld/tokenizer.pt`. (`checkpoints/occluded/`
holds legacy models from the deleted occluded-bouncing env — not used by the current code.) Authoritative
experiment checkpoints stay under `experiments/EXP-NNN/`.
