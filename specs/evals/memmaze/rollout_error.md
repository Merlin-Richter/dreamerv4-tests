# rollout_error.py — quantitative Memory-Maze rollout-error: how fast decoded pixel error grows over a short autoregressive rollout, measured identically for every model.

The quantitative companion to `evals/memmaze/sheets.py` (which only illustrates). One reusable number
per rollout horizon: mean decoded pixel MSE against ground truth, computed the SAME way for vanilla,
memory-token, and archive dynamics models so their rollout quality is directly comparable on the same
held-out data. This is the short-horizon visual-error spine for the memmaze model comparison.

**Protocol** (per scored sample = one `(episode, start)` pair):
1. **Streamed prefill** — take `n_prefill` (default 128) sequential ground-truth frames + their true
   actions and commit them through the model's NORMAL sliding window. The dynamics window is only
   `max_temporal_length` (32) frames; `rollout_init` commits the first window in one forward then
   teacher-forces the rest one frame at a time, evicting as it slides. This is a 128-frame *streamed*
   prefill, **not** a 128-frame context window — as frames leave the window, memory / archive
   mechanisms are free to carry their information forward; a vanilla model simply forgets them.
2. **Scored rollout** — generate the next `n_gen` (default 32) frames autoregressively on the TRUE
   action sequence (`rollout_step(commit=True)`). GENERATED frames, not ground truth, become the
   visual history during scoring.
3. **Score** — decode every generated latent and compare to the real ground-truth frame with pixel
   MSE (frames in `[0,1]`, mean over `H·W·C`, vs RAW ground truth). Per horizon `1..n_gen`, average
   across samples.

**Reference baselines** (model-independent, computed once per eval, saved in the same JSON):
- `tokenizer_floor` — decode the TRUE latents for the scored frames (same trailing-window decode as
  the model prediction). The reconstruction ceiling: the smallest MSE reachable through this frozen
  tokenizer; every model curve sits above it.
- `copy_last` — hold the last prefill ground-truth frame constant for every horizon. The naive
  no-dynamics reference (error a static prediction accrues purely from the scene moving).

**Latents** are encoded in NON-OVERLAPPING tokenizer-window blocks, exactly as the training latent
cache does (`train_dynamics.py`): the causal encoder sees only its own window and latents are
~window-invariant, so this matches the training distribution and lets `n_prefill` exceed the
tokenizer's 64-frame one-shot encode limit. **Frame/action alignment** is the established one
(train_dynamics / sheets / generate): `action[t]` pairs with `frame[t]`, so the generated frame at
absolute position `p` is conditioned on `action[p]`.

**Comparability** is the point: `build_samples` derives the `(episode, start)` set as a pure function
of `(episode set, n_samples, need, seed)`, so any two models run with the same arguments score the
IDENTICAL samples. The saved JSON records enough provenance (checkpoint + tokenizer SHA-256, frames
file/shape, full protocol, the exact sample list, model config) for the companion plot to reject
series whose evaluation settings are not directly comparable.

## Interface
- `rollout_error(model, tokenizer, frames, actions, *, samples, n_prefill=128, n_gen=32, K=4,
  device="cpu", window=None, batch_size=16) -> dict` — the batched core. `samples` is a list of
  `(episode, start)`. Returns per-horizon numpy arrays: `mse` (model, NaN-aware mean), `mse_std`,
  `tokenizer_floor`, `copy_last`, `n_finite` (finite-sample count per horizon). `window` (total
  frames) forces a shorter sliding window than training (`max_ctx = window-1`); None = native.
  Everything is batched over `samples` (batch axis = episodes/starts) in chunks of `batch_size`.
- `build_samples(frames, episodes, n_samples, need, seed) -> list[(ep, start)]` — deterministic
  sample list; episodes cycled round-robin, starts drawn from a `seed`-seeded generator so `need =
  n_prefill + n_gen` frames fit. Episodes shorter than `need` are dropped.
- `val_episodes(n_episodes, val_fraction=0.05) -> np.ndarray` — the trainer's held-out ids (seed-0
  permutation), for parity with sheets; the `val12` file is already fully held out so the CLI
  defaults to `--val-fraction 0` (all episodes).
- Helpers: `_encode_windowed` (non-overlapping tokenizer-window encode), `_decode_tail` (trailing-
  window decode with a real latent prefix, matching the sheet), `_mse_per_horizon`. Reused from
  gridworld: `_load` (checkpoint loader), `_tokenizer_window`.

## Behavior
- Library `rollout_error` takes an already-loaded `model` + frozen `tokenizer` and returns arrays —
  pure, no file IO. The `__main__` CLI (`--checkpoint --tokenizer --frames --actions --out-dir --name
  --n-prefill --n-gen --n-samples --batch-size --K --window --seed --val-fraction --episodes`) is the
  convenience layer: it loads `{config, model_state_dict}` payloads like the other memmaze evals,
  builds the sample set, runs the eval, and writes `rollout_error_<name>.json` to `--out-dir` (default
  `outputs/rollout_error/`, gitignored). `--actions` defaults to `<frames stem>_actions.npy`; frames
  default to `data/memmaze9x9_val12.npy`.
- Decode: the generated block (and the floor's true block) is decoded by ONE tokenizer-decoder call
  on the trailing tokenizer window `[last (win−n_gen) context latents | block]`, sliced to the
  `n_gen` tail — the SAME decode the sheet uses, so sheet pixels and scored pixels agree. Requires
  `n_gen <= tokenizer window`. The context prefix is TRUE latents (frames that genuinely precede the
  rollout); the decoder is temporally causal so no future frame leaks into an earlier decode.
- Action-conditioning is load-bearing: if `model.n_actions > 0` and no actions file is found the CLI
  ERRORS (an unconditioned memmaze rollout is meaningless). Both memory and vanilla models take the
  same code path; `rollout_init` / `rollout_step` handle `n_memory>0` (written-memory relay) internally.
- Frames/decodes are BGR end-to-end with NO channel swap. MSE is reported in `[0,1]²` units
  (`metric: "pixel_mse_01"` in the JSON).

## Invariants
- The model never sees a ground-truth latent for any SCORED frame: `rollout_step` generates each from
  pure noise; only the true ACTION stream conditions generation. (True latents appear only as the
  frozen decoder's temporal prefix, and only for already-observed prefill frames.)
- The streamed prefill is a genuine 128-frame pass through the `max_temporal_length` window, never a
  128-frame context window — memory/archive models must relay pre-window state, not widen the window.
- Compared models must share protocol, tokenizer, frames, and sample set to be a fair comparison; the
  JSON carries the provenance for the plot to enforce this. A serial per-episode fallback is NOT
  acceptable — the prefill and rollout stay batched across samples.
- This is a SHORT-HORIZON pixel-error instrument: exact MSE can penalize a visually plausible rollout
  that diverges slightly from the recorded trajectory. Claims from it must keep that caveat, and it
  does not by itself establish world-model correctness or memory retention.
