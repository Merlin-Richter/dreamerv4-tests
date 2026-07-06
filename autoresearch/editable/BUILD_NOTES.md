# editable/train.py + editable/adapter.py — build notes (2026-07-06)

Seeded per the BUILD PLAN in `tasks/in-progress/autoresearch-harness.md`. The editable
package is now complete: `model.py` (vendored DynamicsModel), `rollout.py` (vendored
mem2mem loss), `train.py` (budgeted trainer), `adapter.py` (eval bridge).

## Design decisions

### train.py
- **Pins as module constants, not CLI flags**: `W_PIN=16`, `N_ACTIONS=5`, `N_LATENTS=4`,
  `BOTTLENECK_DIM=64` are hardcoded (loud header comment). The window pin is deliberately
  NOT exposed on the CLI — the loop agent would have to edit the constant, which the
  driver's window probe catches anyway.
- **Winner defaults** (GridWorld rollout-only mem2mem): `--mem2mem-frac 1.0`, bootstrap
  OFF (`--bootstrap` opt-in; off forces `n_d_unlocked=1` = d_min-only, exactly the
  `--no-bootstrap` semantics of `experiments/mem2mem/train_mem2mem.py`), `--ff9 0`
  (OFF; the fair no-FF9 ablation proved it unnecessary). The recipe's other knobs
  (`--ff9-norm-flow`, `--relay-grad-clip`, `--tbptt-frames`, curriculum flags) are kept
  for the loop to explore.
- **Cache location by hash only**: `--tokenizer` is hashed (sha256[:12]) to find
  `<data>/latents-<hash>.npy` (the `autoresearch.driver.latent_cache` naming convention);
  the tokenizer itself is never loaded — training never pays for it.
- **Random-offset clips**: `RandomClipDataset` draws a fresh uniform start offset per
  access (sound per the driver's window-invariance probe: train 9.4x/val 12.6x margins,
  cos 0.9975). One epoch = `T // clip_len` clips/episode, coverage-equivalent to fixed
  chunking. Offsets come from **torch's** RNG, not numpy — numpy's state forks identically
  into DataLoader workers. Val uses deterministic offset-0 chunking.
- **Budget**: clock starts at `main()` entry (setup counts — that is the realistic driver
  cost), checked after **every optimizer step**; on expiry: save, print
  `BUDGET_STOP step=N elapsed=S`, exit. Val is skipped on the expiring epoch. Secondary
  cap `--epochs` prints `EPOCHS_DONE ...`. Checkpoint is (re)saved at every epoch end and
  at exit.
- **LR schedule**: recipe's warmup(max(200, 5%)) -> flat -> cosine(80-100% -> 1e-6) laid
  out over `len(loader) * --epochs` steps, overridable via `--sched-steps`. **Caveat for
  calibration**: with a budget stop the schedule horizon must be sized so the budget lands
  near the cosine tail (else the run dies mid-flat or even mid-warmup). This is why
  `--sched-steps` exists — the driver can set it from a measured steps/sec.
- **`--n-memory 0` = the vanilla reference arm**: mem2mem asserts `n_memory>0`, so the
  trainer force-disables `mem2mem_frac` (with a printed note) and trains the pure windowed
  shortcut-forcing loss — the no-memory arm runs from the same file.
- **`--num-workers 0` default**: clips are ~32 KB mmap slices; Windows spawn overhead per
  epoch under a wall-clock budget outweighs any loading parallelism.
- **Checkpoint payload**: exactly `{"model_state_dict", "config": asdict(cfg)}`.
- No wlog/W&B: the driver owns logging (`runs/experiments.jsonl`); one print line/epoch.

### adapter.py
- `make_adapter(ckpt_path, tokenizer_path, device=None, K=None) -> factory`. Model +
  frozen tokenizer are loaded **once**; each `factory(env_or_none)` call returns a fresh
  `DynamicsAdapter` (per-episode rollout state) sharing the loaded modules. The env
  argument is ignored (candidates run `privileged=False` → it is `None` anyway).
- `begin`: frames uint8 RGB → float [0,1] → tokenizer encoder in **16-frame chunks**
  (the latent cache's encoding convention; full chunks batched along B — identical to
  sequential encodes since the encoder is windowed-causal; trailing partial chunk encoded
  as-is), then `model.rollout_init(context, actions, K)` — the long-context prefill
  (prefix 192 > W=16) teacher-forces the beyond-window frames through the sliding window
  with written-memory relay, which is exactly what it exists for. No API mismatches found;
  nothing to adapt.
- `step`: `rollout_step(state, action, commit=True)` (K shortcut steps from
  `config.inference_steps`=4 + the near-clean written-memory commit pass), decode the
  committed clean latent through the tokenizer decoder at T=1, return uint8 RGB via
  `round(frame*255)`.
- All inference `torch.no_grad` + eval-mode (dropout/MAE off) + bf16 autocast on cuda.
- Config reload drops `dtype` (torch.dtype doesn't survive JSON round-trips) and unknown
  keys (forward compat), mirroring `driver/latent_cache.load_tokenizer`.
- Imports from `autoresearch.frozen.tokenizer_model` only (the frozen layer is
  hash-checked; NOT from `src/`). `python autoresearch/editable/adapter.py --checkpoint …`
  runs a one-episode smoke.
- Both files import package-style (`autoresearch.editable.*`) and as plain scripts
  (fallback inserts the repo root on `sys.path`).

## Verification transcript (all on the 4070, `venv/Scripts/python.exe`, repo root)

a) **Standalone imports** — `import autoresearch.editable.train` +
   `import autoresearch.editable.adapter`: OK; script-mode (`runpy.run_path`) exercises
   the fallback import path: OK.

b) **Fake cache + budget training** — tiny datasets via
   `python -m autoresearch.frozen.datagen --out <scratch>/cf_tiny[_val] --n-episodes 8/4 --T 1024`
   (real sidecar shapes/dtypes verified: actions (N,1024) uint8, actions[:,0]==STAY==4),
   fake fp16 latents written as `latents-bd8f18857d71.npy` (hash of the real
   `checkpoints/colorfield/tokenizer.pt`), shape (N,1024,4,64). Then
   `train.py --budget-s 30 --epochs 200 --batch-size 8`:

   ```
   device=cuda params=7.75M W=16 (PINNED) n_actions=5 cache=latents-bd8f18857d71.npy
     train_eps=8 val_eps=4 clip_len=64 n_ctx choices=[4, 8, 16] mem2mem_frac=1.0
     bootstrap=False use_ff9=False ff9_k=0 n_memory=4 budget_s=30.0
   Epoch 1/200 | steps 16 | elapsed 11.7s | val(normal) 0.12505 | train mem2mem 0.19135 ...
   Epoch 3/200 | steps 48 | elapsed 27.9s | val(normal) 0.11018 | train mem2mem 0.18054 ...
   Epoch 4/200 | steps 54 | elapsed 30.0s | val(normal) nan | train mem2mem 0.17080 ...
   BUDGET_STOP step=54 elapsed=30.1
   ```
   Stopped mid-epoch on the per-step check; checkpoint reloads with keys
   `['config', 'model_state_dict']` and the pinned config
   (`max_temporal_length=16, n_actions=5, n_latents=4, bottleneck_dim=64, n_memory=4`).
   7.75M params at the default dims (embedding 256, depth 9, heads 16).

c) **Adapter end-to-end** — that checkpoint + the REAL tokenizer:
   `run_episode(make_adapter(...), EvalOutAndBack(20,30), map_seed=1, ep_seed=2,
   prefix_len=48, imag_len=64, privileged=False)` →
   `50 events, fidelity=0.031, 112 positions` in 5.7 s (score is garbage — the model was
   trained on random latents; the machinery is what was under test). Script-mode smoke
   (`python autoresearch/editable/adapter.py --checkpoint …`) also green.

d) **No forbidden imports** — grep over both files: no `src/`, no `experiments/` (and no
   `models/`/`training/`/`wlog` aliases). Adapter's only cross-package imports are the
   frozen layer (`tokenizer_model`, and `eval_comeback`/`eval_policies` inside the
   `__main__` smoke only).

Bonus: `--n-memory 0` vanilla arm smoke (10 s budget) — forces `mem2mem_frac=0.0`, trains
the normal windowed loss (val 0.016 on fake latents), BUDGET_STOP + reloadable checkpoint,
adapter episode OK.
