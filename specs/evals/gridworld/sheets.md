# sheets.py — qualitative GridWorld rollout sheets: render a model's rollout next to ground truth as one PNG.

The visual companion to `recall.py`. `recall` gives the *number* (retention curves); `sheets` gives the
*picture* — stacked filmstrips you can eyeball to see whether the square is tracked in the clear and
whether it is remembered behind the curtain. Cheap (a handful of rollouts), cv2-only (no matplotlib, so it
runs in the cluster venv and locally on the 4070), BGR end-to-end (matches the env + tokenizer + `recall`).

Two sheet kinds, each a vertical stack of per-sample blocks; a block is two filmstrip rows (TOP / BOTTOM)
with columns = timesteps, a header label, and a thick yellow bar marking where context ends and the
rollout begins.

- **occlusion** — the memory picture, built from a controlled `GridWorldEnv` (not the dataset). The model
  sees `n_ctx` revealed frames, then rolls `n_occ` OCCLUDED steps (action=1) carrying its own cache/memory.
  TOP = the TRUE underlying square (rendered from env state — what is really behind the curtain). BOTTOM =
  the model's BELIEF: at each step a READ-ONLY reveal peek (action=0, `commit=False`) decoded to where the
  model thinks the square is. This is the same branching rollout `recall` scores, so the strip and the
  curve agree. A memory model should keep the belief on the true square past the latent window; a vanilla
  (`n_memory=0`) model's belief decays once the last revealed frame is evicted.
- **normal** — the in-the-clear sanity picture, from held-out dataset episodes that are fully revealed.
  TOP = ground-truth frames, BOTTOM = model rollout (first `n_ctx` cols are context reconstructions, then
  free-run predictions, curtain up throughout). Confirms motion tracking works when nothing is hidden.

## Interface
- `occlusion_sheet(model, tokenizer, *, seeds, n_ctx=4, n_occ=16, K=4, device="cpu", scale=4,
  window=None) -> np.ndarray` — controlled-env occlusion sheet (uint8 BGR image), one block per seed in
  `seeds`. `window` (total frames) optionally forces a shorter sliding window than the model trained with.
- `normal_sheet(model, tokenizer, frames, actions=None, *, n_samples=5, n_ctx=4, n_gen=None, K=4,
  device="cpu", scale=4, seed=0) -> np.ndarray` — free-run sheet over fully-revealed clips drawn from
  `frames` (N,T,H,W,3 uint8). With `actions` it picks windows whose curtain is up throughout; without, it
  takes the first window of random episodes. `n_gen` defaults to `model.config.max_temporal_length-n_ctx`.
- `save_sheet(path, sheet) -> None` — `cv2.imwrite` wrapper; warns and skips if `sheet` is empty.
- Helpers (internal): `_controlled_episode`, `_occlusion_belief`, `_free_rollout`, `_decode_seq`,
  `_row`, `_sample_block`.

## Behavior
- Library functions take an already-loaded `model` + frozen `tokenizer` (like `recall`) and return an image
  array — pure, no file IO, callable from a trainer/test/notebook. `save_sheet` and the `__main__` CLI
  (`--checkpoint --tokenizer --frames --out-dir --kind {occlusion,normal,both} --n-samples --n-ctx
  --n-occ --window --K`) are the local-run convenience layer; the CLI loads checkpoints the same way
  `interactive/play_dynamics.py` does (`{config, model_state_dict}` payloads). The CLI writes
  `sheet_occlusion.png` / `sheet_normal.png` to `--out-dir`, default `outputs/sheets/` (gitignored).
- Occlusion belief uses `model.rollout_init` then per occluded step a `rollout_step(commit=False)` reveal
  peek (the belief) followed by `rollout_step(commit=True)` occluded commit — identical alignment to
  `recall.roll_and_score`. Decode the peek with the tokenizer over a sliding window of its temporal length.
- Action ids are passed only when `model.n_actions>0` (reveal=0, occlude=1); for an unlabeled model the
  curtain has no meaning and the occlusion sheet is degenerate (rolls free) — the CLI warns.
- Decoded frames are treated as BGR with NO channel swap (the tokenizer is BGR in/out); env renders are
  BGR. The two rows are therefore directly comparable and `cv2.imwrite` gets BGR.

## Invariants
- Reveal peeks are READ-ONLY: a `commit=False` step must never advance/corrupt the carried rollout (the
  occluded rollout is the spine; peeks only branch off it) — same contract as `recall`.
- The env advances physics independently to supply the true square each step (the curtain never affects
  physics); the model never sees the occluded frames, only its own committed predictions.
- This is a QUALITATIVE instrument. Memory claims are decided by `recall`, not by these strips; sheets only
  illustrate what the curve already measured.
