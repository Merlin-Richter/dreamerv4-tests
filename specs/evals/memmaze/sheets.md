# sheets.py — qualitative Memory-Maze rollout sheets: render a model's action-conditioned rollout next to ground truth as one PNG.

The memmaze counterpart of `evals/gridworld/sheets.py`. GridWorld's *occlusion* sheet does not map here
(there is no controlled env or curtain action — occlusion in Memory Maze is natural: the agent looks
away and later looks back), so this module has ONE sheet kind:

- **rollout** — per sample block: TOP = ground-truth dataset frames, BOTTOM = model rollout (first
  `n_ctx` columns are context reconstructions, then free-run predictions **conditioned on the TRUE
  action sequence** from the dataset). Columns = timesteps, header label, thick yellow bar where
  context ends. With `n_gen` past the model's window it doubles as the qualitative long-horizon /
  consistency check (does the maze stay coherent after the window slides?).

**Long-context prefill:** before generating, the model processes `n_pre` (default 64) true context
frames via `rollout_init`'s long-context path (first window in one forward, the rest teacher-forced
sliding commits — a memory model absorbs the pre-window part through its relay). The sheet displays
only the LAST `n_ctx` of them (`"{n_pre} ctx ({n_ctx} shown)"` in the label): with a short context
the maze is mostly unobserved and GT-tracking is impossible by construction; long prefill is the
intended usage. `n_pre <= tokenizer window` (one-shot encode limit; 64 for the memmaze tokenizer)
and `n_ctx <= n_pre` (the shown context is a suffix).

Samples come from **held-out episodes** by default: the module reproduces `train_dynamics.py`'s
val split (`torch.randperm(n, generator=seed 0)`, first `round(n*val_fraction)` episodes) so sheets
are not rendered on trained-on episodes. This duplicates the trainer's split logic and MUST stay in
sync with it.

## Interface
- `val_episodes(n_episodes, val_fraction=0.05) -> np.ndarray` — the trainer's held-out episode ids
  (same permutation: seed 0, `n_val = min(max(1, round(n*val_fraction)), n-1)`).
- `rollout_sheet(model, tokenizer, frames, actions, *, episodes=None, n_samples=4, n_pre=64,
  n_ctx=8, n_gen=None, K=4, device="cpu", scale=2, seed=0, window=None) -> np.ndarray` — uint8 BGR
  sheet, one block per sampled episode. `frames` (N,T,H,W,3 uint8) and `actions` (N,T int) may be
  memmaps. `episodes` overrides episode selection (default: first `n_samples` of `val_episodes(N)`);
  the start offset within each episode is drawn uniformly per sample from a `seed`-seeded generator
  so `n_pre + n_gen` frames fit. `n_gen` defaults to `model.config.max_temporal_length - n_ctx` and
  MAY exceed the window (the carrying rollout slides). `window` (total frames) optionally forces a
  shorter sliding window than the model trained with (`max_ctx = window-1`), like gridworld sheets.
- Reused from `evals/gridworld/sheets.py` (the shared drawing/loading layer, one source of truth):
  `_sample_block`, `_assemble`, `save_sheet`, `_load`; `_tokenizer_window` from
  `evals/gridworld/recall.py`.

## Behavior
- Library function takes an already-loaded `model` + frozen `tokenizer` and returns an image array —
  pure, no file IO. The `__main__` CLI (`--checkpoint --tokenizer --frames --actions --out-dir
  --n-samples --n-ctx --n-gen --K --scale --seed --val-fraction --episodes --window`) is the
  convenience layer; it loads `{config, model_state_dict}` payloads exactly like the gridworld CLI
  and writes `sheet_memmaze_rollout.png` to `--out-dir` (default `outputs/sheets/`, gitignored).
  `--actions` defaults to `<frames stem>_actions.npy` next to the frames file; `--episodes` (ids)
  overrides the val-split selection; `--val-fraction 0` selects from ALL episodes.
- Rollout: encode the `n_pre` context frames with the tokenizer (one shot), then
  `model.generate(ctx, n_gen, K=K, action_idx=<true actions for pre+gen>, max_ctx=...)` — generate's
  `rollout_init` handles `n_pre >` the dynamics window (long-context prefill). Display decode is ONE
  tokenizer-decoder call on the trailing tokenizer window of the latent sequence, sliced to the
  displayed `n_ctx + n_gen` tail (safe: the window-invariance probe showed window-delta recon error
  ~60x below the recon error itself; requires `n_ctx + n_gen <=` tokenizer window).
- Action-conditioning is load-bearing: if `model.n_actions > 0` and no actions file is found the CLI
  ERRORS (an unconditioned memmaze rollout is meaningless); an unlabeled model (`n_actions == 0`)
  rolls free with a printed warning. The block label includes the action-id digit string for the
  rendered window with a `|` at the context boundary, so turns can be matched against the strip.
- Frames/decodes are BGR end-to-end with NO channel swap (tokenizer is BGR in/out); `cv2.imwrite`
  gets BGR. Default `scale=2` (memmaze blocks are wider than gridworld's: 32+ columns).

## Invariants
- Episode selection defaults to HELD-OUT episodes; the split reproduction must match
  `train_dynamics.py` (same seed-0 permutation and n_val formula) or the "held-out" claim is false.
- The model never sees ground-truth frames past the context — BOTTOM after the boundary is pure
  rollout (only actions come from the dataset).
- This is a QUALITATIVE instrument: it illustrates; memory/consistency CLAIMS are decided by the
  (future) memmaze recall/probe eval, not by these strips.
