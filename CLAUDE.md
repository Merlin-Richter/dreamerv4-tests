# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of a video dynamics model inspired by Dreamer v4. Code is organized
by concern under `src/` (`models/`, `training/`, `envs/`, `datagen/`, `evals/`, `tests/`,
`interactive/` — see `REPO_MAP.md`). The four pipeline-stage models (historically the A/B/C/D dirs,
now modules in `src/models/`) are:

- **LM** (`models/lm.py`): Character-level language model (standalone, not part of main pipeline)
- **Single-image AE** (`models/single_image_ae.py`): Frame-only autoencoder (baseline)
- **Tokenizer** (`models/tokenizer.py`): Temporal video autoencoder with frozen bottleneck (= the frozen `checkpoints/occluded/tokenizer.pt`)
- **Dynamics model** (`models/dynamics_model.py`): Causal transformer that predicts latent frame dynamics

The pipeline compresses video into learned latent representations and trains a generative model to predict future frames autoregressively.

## Architecture & Data Flow

### Main Training Pipeline

```
Raw Video Frames (B, T, H, W, 3) [uint8]
  ↓
[models.tokenizer] - Frozen tokenizer
  • Patchifies frames (8×8 patches by default)
  • Encodes via spatial attention layers + temporal attention layers (alternating)
  • Uses MAE (Masked Autoencoder) dropout to prevent latent collapse
  • Outputs: (B, T, n_latents=4, bottleneck_dim=64) clean latents z1
  ↓
[models.dynamics_model] - Trainable
  • Per-frame latent denoising via causal (block-temporal) attention
  • Trained with shortcut forcing: diffusion-based loss + bootstrap distillation
  • Supports discrete action conditioning (optional)
  • Generates: predicts z1 from noised latent + causal history
```

### Tokenizer Architecture (models/tokenizer.py)

**Encoder:**
- Patchifies input into non-overlapping patches (default 8×8)
- Projects patches to `embedding_dim=256` via learned position embeddings
- Spatial layers: full self-attention within each frame over patch tokens + 4 learned latent tokens
- Temporal layers: causal attention across frames (every 4th layer)
- MAE dropout: during training, randomly masks patch tokens (0–90% per frame)
- Outputs 4 learned latent tokens per frame, projected to `bottleneck_dim=64`

**Decoder:**
- Inverse operation: learned patch tokens + bottleneck latents → spatial/temporal layers → patches → image
- Uses sigmoid activation to produce [0,1] RGB output

**Key Design Choice**: Learned latent tokens have restricted cross-attention with image patches (encoder) and patches cannot attend back (decoder), forcing an information bottleneck.

### Dynamics Model (models/dynamics_model.py)

**Token layout per frame** (along spatial axis):
```
[action_token(s) | latent_tokens(n_latents=4) | register_tokens(n_registers=4) | shortcut_token]
```

**Shortcut Forcing Loss:**
- Samples per-frame (τ, d) pairs where τ ∈ [0, 1] is signal level and d = 1/K is step size
- Noises latent: z̃ = (1−τ)z₀ + τz₁
- Predicts clean latent z₁ (x-prediction, not velocity)
- Finest step (d=1/K_max): pure flow loss `||z₁ − ẑ₁||²`
- Coarser steps: bootstrap loss distilling two d/2 steps (Eq. 7 from paper)
- Ramp weight: w(τ) = 0.9τ + 0.1 focuses capacity on high-signal levels

**Action Conditioning:**
- `n_actions=0`: unlabeled video (only learned action embedding)
- `n_actions>0`: discrete action table maps each action ID to per-frame features added to embedding

**Inference (generate):**
- Autoregressive rollout: each frame uses K shortcut steps (default K=4)
- Context is held near-clean at signal level `context_signal` (default 0.9; high = near-clean)
  to prevent error accumulation (see EXP-008/D-010 — the old "τ_ctx=0.1" was the inference bug)
- `generate_cached()` (T-008/D-017): KV-cached drop-in for `generate()`, **bit-for-bit identical**
  (same RNG → same draws), ~2× faster at probe scale. Caches each frame's context K/V across the
  K shortcut substeps (rebuilt per frame — NOT cross-frame). RoPE is computed at **absolute
  positions** on the fly (temporal Attention, `positions=` arg) so cached K/V is never re-rotated
  and long rollouts exceed the cos/sin table — see `HOWTO/rope_kv_cache_caveat.md`. Training/default
  forward (`positions=None`) is unchanged.
- `generate_streaming()` + `stream_rollout_init`/`stream_rollout_step` (T-012/D-020): **cross-frame
  sliding-window KV eviction cache** for **efficient open-ended / continuous rollouts**. Persists
  each finalized frame's K/V across rollout steps and evicts the oldest time-column when the window
  (N−1) overflows; since cached K/V are pre-rotated at absolute positions, eviction is a pure slice
  (no re-rotation). So an arbitrarily long rollout costs O(1) attention per step instead of
  re-encoding the whole window — the init/step primitives drive open-ended generation (e.g.
  interactive play, long video). One deliberate semantic deviation from `generate()`: each frame's
  context-noise is drawn **once** at commit instead of redrawn every step (a frame's committed
  representation is fixed once generated). So NOT bit-identical to `generate()`, but bit-identical to
  `generate_windowed` (the **uncached twin**, below), and the residual deviation from `generate()` is
  within its own seed-to-seed noise on a trained model (smaller, in fact). Inference-only
  (`@torch.no_grad()`). FF7 register-memory path is window-1 already → dispatched to `generate_memory`
  unchanged.
- `generate_windowed()` (T-012): the **uncached twin** of `generate_streaming` — same frozen
  per-frame context-noise semantics by full windowed recompute (no persistent cache), independent
  stepping logic. Both take `noise_seed`: a deterministic per-frame noise source keyed on the
  **absolute frame id** (not RNG call order), so a shared seed gives the two paths identical noise
  and `generate_streaming == generate_windowed` bit-for-bit — the test that isolates the cache (any
  divergence is a cache/eviction/RoPE bug, not a noise mismatch). `noise_seed` also makes any rollout
  reproducible across global-RNG state.

### Supporting Components

**Attention Modules (both tokenizers & dynamics):**
- QK-norm + RMSNorm for stability
- Learnable per-head logit scaling (initialized to log(4.0), clamped to log(100))
- Soft-cap activation: `tanh(logits/cap) × cap` to bound attention logits
- RoPE on temporal axes for causal layers

**MLP Variant:**
- SwiGLU: `(Linear₁ · ReLU(Linear₂)) → dropout → Linear₃ → dropout`

## Development Commands

### Environment Setup

```bash
# Python 3.11+ required
python -m venv venv
venv\Scripts\activate  # Windows PowerShell
source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
# Key dependencies: torch, numpy, opencv-python, wandb (optional), lpips (optional)
```

### Generate/Prepare Data

```bash
# Generate bouncing object dataset (DVD-style physics simulation)
python src/datagen/generate_bouncing.py --n_episodes 1000 --out bouncing.npy

# Debug: preview a single episode
python src/datagen/generate_bouncing.py --debug --shape star
```

### Training

Run training scripts from the repo root (each bootstraps `src` onto the path — no `cd` needed).

**Single-Image Autoencoder (B):**
```bash
python -u src/training/train_single_image_ae.py --epochs 5 --batch-size 32 --lr 3e-4
```

**Temporal Autoencoder (C) - Tokenizer:**
```bash
python -u src/training/train_tokenizer.py --epochs 10 --batch-size 16 --lr 3e-4
# With W&B logging:
python -u src/training/train_tokenizer.py --wandb --wandb-project my-project --epochs 10
# Enable LPIPS perceptual loss (additional metric, slower):
python -u src/training/train_tokenizer.py --lpips
```

**Dynamics Model (D):**
```bash
python -u src/training/train_dynamics.py --epochs 20 --batch-size 32 --context-length 16
# With W&B:
python -u src/training/train_dynamics.py --wandb --wandb-project my-project
# FF7 register-memory training (D-014): adds the single-timestep-sufficiency loss with
# lookahead K. At inference, use_register_memory=True checkpoints carry register state across
# the window via generate_memory, which is a thin loop over the reusable primitives
# memory_rollout_init / memory_rollout_step (also driven by the interactive viewer):
python -u src/training/train_dynamics.py --ff7 1 --lambda-ff7 1.0 --seed 0
# FF9 ROLLOUT-TRAINING (D-048, op-3 memory->memory relay): trains the cross-window memory relay the
# FF9 sufficiency loss (_ff9_loss) leaves un-gradiented (it fills intermediate frames with the
# learned-init placeholder). Rolls h differentiable memory hops (TBPTT-k via --ff9-rollout-tbptt),
# carrying the WRITTEN memory each hop; per-step source latents are hidden (--ff9-rollout-hide-mode
# tail=contiguous occlusion-mirror | iid). --rollout-clip-len decouples the dataset clip from the
# model window so the relay trains DEEPER than the window (P1/EXP-029: dynamic-state recall horizon ~
# training rollout depth -> train to the eval depth). Combine with --ff9 (the sufficiency term that
# makes memory CONTAIN state) + --ff9-rollout-warmup (contain-then-propagate). Eval with the
# UPDATING-memory inference: model.generate_updating_memory (op-3/B2, memory re-written each step) via
# the recall adapter inference='relay' (vs 'snapshot'=frozen B1, 'windowed'=plain sliding window).
python -u src/training/train_dynamics.py --ff9 3 --ff9-rollout 24 --ff9-rollout-tbptt 12 \
  --ff9-rollout-hide-mode tail --ff9-rollout-warmup 20 --context-length 16 --rollout-clip-len 28 --seed 0
# Gate tests (run from src/tests/, CPU OK) — FF7 paths, KV cache, streaming cache, FF9 rollout:
python -u src/tests/test_ff7_smoke.py
python -u src/tests/test_kv_cache.py
python -u src/tests/test_stream_cache.py
python -u src/tests/test_ff9_rollout.py
```

**Language Model (A):**
```bash
python -u src/training/train_lm.py --epochs 10 --batch-size 64 --seq-len 64 --lr 3e-4
python -u src/interactive/lm_inference.py --checkpoint checkpoint.pt --prompt "ROMEO:\n" --max-new-tokens 500
```

### Visualization & Testing

```bash
# Inspect autoencoder reconstruction (interactive OpenCV window)
python src/training/train_single_image_ae.py \
  --test-checkpoint --checkpoint checkpoints/bouncing/single_image_ae.pt

python src/training/train_tokenizer.py \
  --test-checkpoint --checkpoint checkpoints/occluded/tokenizer.pt

# Inspect dynamics rollout (interactive)
python src/training/train_dynamics.py --test-checkpoint
python src/interactive/play_dynamics.py  # Single-frame interactive
# Auto-detects the checkpoint's inference mode (same dispatch order as generate()):
#   FF7 (config.use_register_memory) -> register-carry relay (memory_rollout_init/step);
#   FF9 v2 (config.use_full_state_memory & n_memory>0) -> full-state-memory rollout
#     (full_state_rollout_init/step, A1+B1) — WRITEs a frozen snapshot from a deeper prefix
#     (up to max_temporal_length-1 frames) then carries it, so static hidden state survives
#     indefinitely past the latent window (the exact inference evaluated in EXP-017);
#   otherwise the vanilla sliding-window path. The on-screen "mode=" line shows which is active.
# To inspect an FF7 model on the occluded env:
python src/interactive/play_dynamics.py \
  --checkpoint experiments/EXP-010/k3/ff7_k3_s0.pt --tokenizer checkpoints/occluded/tokenizer.pt \
  --frames occluded.npy --actions occluded_actions.npy
# To inspect the FF9 v2 model (EXP-017) on the occluded env:
python src/interactive/play_dynamics.py \
  --checkpoint experiments/EXP-017/ff9v2_s0.pt --tokenizer checkpoints/occluded/tokenizer.pt \
  --frames occluded.npy --actions occluded_actions.npy
```

### Wandb Integration

All training scripts support optional W&B logging via `wlog.py` (no-op by default):

```bash
python -u src/training/train_tokenizer.py \
  --wandb \
  --wandb-entity YOUR_TEAM \
  --wandb-project transformer-C-tokenizer \
  --wandb-name run-v1 \
  --wandb-tags experiment,v1
```

Set environment variables for defaults:
```bash
export WANDB_ENTITY=your-entity
export WANDB_PROJECT=transformer
```

## Key Files & Responsibilities

> See `REPO_MAP.md` for the full directory map (concept→location index, the Eval/BaseEnv interfaces,
> how to add an env/eval). All evals live under `src/evals/` (T-019 reorg, D-030): the **FROZEN** spine
> is `evals/probe_env.py` + `evals/revisit/probe.py` + `evals/position_consistency/consistency.py`
> (frozen @ 5503e75 — changes are logged decisions); the **working** toolbox is `evals/motion/` (motion
> curves) + `evals/rollout_view/` (A/B headline). Common interface in `evals/base.py`
> (`import evals; evals.discover()` → REGISTRY). NEW experiment scripts import these; historical ones
> are frozen to their commit (D-031).

### Core Models

| File | Role |
|------|------|
| `src/models/lm.py` | Transformer LM architecture |
| `src/models/single_image_ae.py` | Single-frame AE (baseline) |
| `src/models/tokenizer.py` | Temporal AE (tokenizer) |
| `src/models/dynamics_model.py` | Dreamer-style dynamics transformer |

### Training Scripts

| File | Purpose |
|------|---------|
| `src/training/train_lm.py` | Shakespeare LM training |
| `src/interactive/lm_inference.py` | LM text generation |
| `src/training/train_single_image_ae.py` | Single-frame AE training + testing |
| `src/training/train_tokenizer.py` | Temporal AE training + testing |
| `src/training/train_dynamics.py` | Dynamics model training + rollout visualization |
| `src/interactive/play_dynamics.py` | Interactive single-frame dynamics |

### Envs, Data & Evals

| File | Purpose |
|------|---------|
| `src/envs/base.py` | `BaseEnv` interface (reset/step + measurement-only `hidden_state`) |
| `src/envs/occluded_bouncing.py` | `OccludedBouncingEnv` (action-conditioned memory env) |
| `src/envs/gridworld.py` | `GridWorldEnv` (D-032; geometry D-038): discrete **6×6** grid memory env — solid bg + square (4 colors), square steps 1 cell/tick in 8 directions w/ wall reflection, same curtain occlusion. **Geometry: 3px border + 6×8px cells + 5×2px lines = 64; cell i interior at 3+10·i. The 10px cell stride is deliberately NOT a multiple of the tokenizer's 8px patch (anti-overfit, D-038).** Recall = 6×6 cell (chance 1/36) + 4-way color (crisp, no ΔRGB fuzz) |
| `src/envs/bouncing.py` | `BouncingEnv` (unconditioned DVD-style sim) |
| `src/datagen/generate_bouncing.py` | Bouncing dataset writer + viewer |
| `src/datagen/generate_occluded.py` | Occluded (curtain) dataset writer + viewer |
| `src/datagen/generate_gridworld.py` | GridWorld dataset writer + viewer. Curtain schedule (D-032): per block 90% one random action / 5% 8-revealed run / 5% 8-occluded run. Writes frames/actions/states + `_colors.npy` (per-ep [bg_idx, square_idx], PALETTE order) |
| `src/evals/revisit/probe.py` + `evals/probe_env.py` + `evals/position_consistency/consistency.py` | FROZEN revisit/position-consistency spine (5503e75) |
| `src/evals/motion/motion.py` | Working motion-eval toolbox (curves + A/B helpers) |
| `src/evals/base.py` | Eval interface + REGISTRY (`import evals; evals.discover()`) |
| `src/wlog.py` | Lightweight W&B logger (no-op unless --wandb) |

### Datasets (repo root) & Checkpoints (`checkpoints/<env>/`)

Datasets at repo root: `bouncing.npy`, `occluded.npy` (+`_actions`/`_states`), `gridworld.npy`
(+`_actions`/`_states`/`_colors`). All gitignored.

**Checkpoints are organized by environment** (D-032 — a model trained on one env does NOT transfer
to another; the env MUST be explicit in the path). Layout `checkpoints/<env>/<role>.pt` (gitignored):
- `checkpoints/occluded/tokenizer.pt` — the frozen LPIPS temporal tokenizer (was `trained_autoencoder.pt`; EXP-006, occluded.npy)
- `checkpoints/occluded/dynamics_vanilla.pt` — retired H2/H3 vanilla baseline (was `my_dynamics.pt`; EXP-007)
- `checkpoints/bouncing/dynamics.pt` — unconditional bouncing dynamics (was `dynamics_bouncing.pt`; EXP-005)
- `checkpoints/bouncing/tokenizer.pt` — bouncing tokenizer (was `src/autoencoder_bouncing.pt`)
- `checkpoints/gridworld/` — GridWorld pipeline (being trained)
Authoritative experiment checkpoints stay under `experiments/EXP-NNN/` (frozen provenance, D-031).

## Config Dataclasses

All models use dataclass configs for reproducibility:

**AutoEncoderConfig** (B & C):
- `embedding_dim`: model width (256–512)
- `n_latents`: learned tokens per frame (4)
- `bottleneck_dim`: latent feature size (32–64)
- `patch_size`: input patch size (8–16)
- `mae_min/max_mask`: dropout range for MAE (C only)
- `max_temporal_length`: sequence length (16–32)
- `depth`, `n_heads`, `mlp_ratio`: transformer hyperparams

**DynamicsModelConfig** (D):
- Must match tokenizer: `bottleneck_dim`, `n_latents`
- `max_sampling_steps`: K_max ∈ {64, 128, 256} (must be power of 2)
- `inference_steps`: K at generation (typically 4)
- `context_signal`: τ_ctx = signal level of context frames during rollout (0.9; 1.0=clean, 0.0=pure noise)
- `n_actions`: 0 for unlabeled, >0 for action-conditioned

**ModelConfig** (models/lm.py):
- `vokab_size`: vocabulary size
- `embedding_dim`: 128
- `max_sequence_length`: sequence length
- `depth`, `n_heads`: transformer depth/width

## Common Workflow

### Training a Full Pipeline

1. **Generate data** (if needed):
   ```bash
   python src/datagen/generate_bouncing.py --n_episodes 5000 --out bouncing.npy
   ```

2. **Train tokenizer (C)** - produces z1 representations:
   ```bash
   python -u src/training/train_tokenizer.py --epochs 20 --batch-size 16
   ```

3. **Train dynamics model (D)** - predicts future z1:
   ```bash
   python -u src/training/train_dynamics.py --epochs 50 --batch-size 32
   ```

4. **Evaluate rollouts**:
   ```bash
   python -u src/training/train_dynamics.py --test-checkpoint
   ```

### Debugging Tips

- **Latent collapse** in C: Increase MAE dropout (`mae_max_mask` toward 0.9) or check `bottleneck_dim` is large enough
- **Poor dynamics**: Verify tokenizer is frozen and checkpoint is loaded; check clip length `chunk_len` matches model's `max_temporal_length`
- **Out of memory**: Reduce `batch_size` or `max_temporal_length`; use memory-mapped numpy (automatic in ChunkClipDataset)
- **W&B integration**: Set `--wandb-project` to override default; silent no-op if wandb not installed

## Notes on Design Choices

- **MAE dropout in C**: Prevents the decoder from ignoring patches and reconstructing the mean image; learned replacement token bridges masked positions
- **Shortcut forcing in D**: Allows efficient inference (4 steps instead of 128) while training with per-frame diffusion signal
- **RoPE on temporal axis**: Allows extrapolation beyond training sequence length
- **Frozen tokenizer in D**: Ensures latent space stability; dynamics model only learns transitions
- **Register tokens in D**: free scratch tokens (from recent ViT research). In the vanilla model they are unconstrained scratch; the FF7 line (D-014) repurposes them as the hidden-state *memory carrier* — temporal attention is position-wise, so each register slot is its own causal channel through time, and FF7's loss + register-carry inference make it relay occluded state past the window.

There are trained versions of Tokenizer which works well and had LPIPS loss during training (`checkpoints/occluded/tokenizer.pt`) and the trained vanilla dynamics model at `checkpoints/occluded/dynamics_vanilla.pt` (its earlier rollout "failure" was an inference bug — context noised at 90%; fixed via `context_signal=0.9`, see EXP-008/D-010). These are OCCLUDED-env models; GridWorld needs its own `checkpoints/gridworld/` pipeline.

Almost always run python commands with the "-u" flag so output doesn't get buffered.