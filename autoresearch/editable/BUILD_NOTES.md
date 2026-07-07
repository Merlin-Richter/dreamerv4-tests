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

---

# SYM VARIANTS — train_sym.py + adapter_sym.py (2026-07-07)

SYMBOLIC-tier variants per `tasks/in-progress/colorfield-sym-frozen-layer.md` ("Model port").
`model.py` and `rollout.py` are UNCHANGED and shared with the pixel tier — the sym files only
replace the data path (no tokenizer anywhere). Built STRICTLY CPU-ONLY (an overnight run owned
the GPU); every command below ran with the GPU hidden and `device=cpu` verified.

## Design decisions

### The codec (adapter_sym.py owns it; train_sym imports it)
- **One-hot viewport rows ARE the latents**: `n_latents = 5` (one token per viewport ROW),
  `bottleneck_dim = 35` = 5 cells × 6 one-hot (=30) + 5 phase one-hot dims appended to EVERY
  row. Phase uses the ABSOLUTE episode tick (`t % 5`) — clips sliced at offset `s` encode
  `(s+i) % 5`, and the eval prefix starts at episode tick 0 so `arange(P)` is exact. The
  x-prediction target includes the phase block (trivially predictable, harmless — spec).
- `encode_latents(grids, ticks)` / `decode_latents(z)` (per-cell ARGMAX over the 6 classes,
  first 30 dims per row; phase dims ignored) live in **adapter_sym.py** — the eval bridge is
  the codec home (the sym analogue of the tokenizer living in adapter.py) — and
  **train_sym.py imports them**, so train-time and eval-time encodings can never drift. This
  is the one structural deviation from the pixel pair (train.py does not import adapter.py);
  taken deliberately, since encode/decode drift is the sym tier's analogue of a tokenizer
  mismatch. Codec dims are DERIVED from the frozen_sym env constants (`OUT_IDX+1`,
  `VIEW_CELLS`, `PHASE_PERIOD`); train_sym asserts its loud pin literals equal them.
- Codec is exact by construction: encode→decode round-trip verified on random grids
  (identity), phase block identical across rows, row sums = 6 (5 cells + 1 phase).

### train_sym.py (mirrors train.py; diffs only where the data path forces it)
- **Pins**: `W_PIN=16` (ticks — under phase-5 dilation only 3.2 effective moves), `N_ACTIONS=5`,
  `N_LATENTS=5`, `BOTTLENECK_DIM=35` as loud module constants; window pin NOT on the CLI.
- **No `--tokenizer`, no latent cache**: data dirs default `data/colorfield_sym[/_val]`;
  sidecars load via the frozen `ColorFieldSymDataset`. Latents are built **on the fly per
  clip**: per-tick centers are precomputed ONCE at load as a **vectorized path integral**
  (`cumsum` of action deltas; exactness cross-checked against the frozen
  `env.positions_from` on episode 0 — a per-episode Python-loop integration would have eaten
  ~10-30 s of the budget at N=5000). Dataset conventions are asserted at load
  (`actions[:,0]==STAY`, off-phase all-STAY, centers on-board). Per `__getitem__`: 64×
  frozen `env.render_grid` + one `encode_latents` — sub-ms numpy work; no cache needed.
- **`SymClipDataset`** keeps `RandomClipDataset`'s exact offset semantics (fresh uniform
  offset per access from **torch's** RNG; val = deterministic `j*clip_len` chunking; one
  epoch = `T // clip_len` clips/episode).
- Everything else is **verbatim train.py**: winner defaults (mem2mem_frac 1.0, bootstrap OFF
  = d_min-only, FF9 OFF), `--fixed-n-ctx`, `--sched-steps` + the short-budget warmup rule
  (10% capped at 200, floor 10), `--snapshot-at`, per-step budget check + `BUDGET_STOP`,
  `--n-memory 0` vanilla arm (forces mem2mem_frac=0), checkpoint payload
  `{"model_state_dict", "config": asdict(cfg)}`, LR schedule, val cap, num_workers 0.
- **GENERALITY RULE**: nothing beyond the pixel-tier recipe was added — no phase-indexed
  loss weighting, no sym-specific heuristics of any kind. The only env knowledge used is the
  frozen data interface itself (render + conventions), not a training heuristic.

### adapter_sym.py (mirrors adapter.py minus the tokenizer)
- `make_adapter(ckpt_path, device=None, K=None) -> factory` — **no tokenizer arg**. Model
  loaded once; each `factory(env_or_none)` returns a fresh `SymDynamicsAdapter` (env ignored;
  candidates run privileged=False → None). `load_dynamics` additionally asserts the
  checkpoint dims equal the sym codec (5, 35) — a pixel checkpoint fails loudly.
- `begin(prefix_grids (P,5,5) uint8, prefix_actions)`: one-hot encode grids + phases
  (`ticks = arange(P)`) → `(1,P,5,35)` → `rollout_init(context, acts, K)`; prefix 192 > W=16
  uses the long-context prefill (teacher-forced sliding window with written-memory relay).
- `step(action)`: `rollout_step(commit=True)` (K=`config.inference_steps`=4 shortcut steps +
  near-clean written-memory commit pass) → ARGMAX-decode to a (5,5) uint8 grid, ids 0..5.
- `torch.no_grad` + eval-mode throughout; bf16 autocast on cuda only (CPU stays fp32).
  `python autoresearch/editable/adapter_sym.py --checkpoint …` runs a one-episode smoke.

## Datasets (generated locally, CPU)

`venv/Scripts/python.exe -u -m autoresearch.frozen_sym.datagen --out data/colorfield_sym
--n-episodes 5000 --T 1024 --seed 0` and `--out data/colorfield_sym_val --n-episodes 250
--seed 777`. sha256 of the sidecars:

```
data/colorfield_sym/
 actions.npy    d8cb99f24671ddcc5925733c9d3be0947aa84125125b6d67144645bfad79ffb0
 ep_seeds.npy   5ac3f8fab189bd7371a089a5673e74afc07ea43afed2bdacae266713aac536af
 maps.npy       34e7ef16c7892aed54cfe6a57022bdd5d00d4101877d76be824a689fd26f16fd
 policy_ids.npy eb98cbb1743231225997155aa516c2aa9b9a590b895b00f795f10a5a652caef9
 starts.npy     476742daa3e2a758126f9359816ba12e870694d2d55f8a914e45850e2c506ef7
 meta.json      de1adf87c46042822227423213f225857a68e24136b2d9833f2459a9f5ced047
data/colorfield_sym_val/
 actions.npy    68c22b9616ba16cc451dc1e6308a777b28493163649a541d8f7117eff5110dff
 ep_seeds.npy   4cc40031491b3983b00a00b5abec3de661fdc717fbc12428535fefa423b5d93c
 maps.npy       e54337168d29b683378dc5d6b352d9319e64542b7403279a1436542bb4e58935
 policy_ids.npy a23392ebfe1fafd5f4c3ffea6f8d73ea421f25912262e119a474c314fcb62b30
 starts.npy     081d966505c28adf698c628daa4e3b88f77d634134de6236a427ec5990b0b31a
 meta.json      6b8fa1e48960381ee5405672f27a5708d4a1e4aa38d29e5381e6e7a5fc98c652
```

## Verification transcript (ALL CPU — `CUDA_VISIBLE_DEVICES=-1`, venv python, repo root)

**GPU-hidden caveat (this torch build, 2.12.0+cu126)**: `CUDA_VISIBLE_DEVICES=""` (empty)
leaves `torch.cuda.is_available()==True` with `device_count()==0` — the trainers' device
pick would choose "cuda" and crash. **`CUDA_VISIBLE_DEVICES=-1`** cleanly yields
`is_available()==False`; used for every run, and every run below printed `device=cpu` /
asserted `not torch.cuda.is_available()`. The GPU (owned by the overnight training run) was
never touched.

0) **Imports + codec round-trip** — package and script-mode imports OK; encode→decode
   identity on random (23,5,5) grids with ticks 37..59; phase block == `eye(5)[t%5]` on every
   row; row sums == 6.

1) **Clip pipeline == frozen render** — on a tiny scratch dataset (8 eps), deterministic
   `SymClipDataset` clips decode back EXACTLY to `ColorFieldSymDataset.episode()` grids at
   the same offsets (grids, actions, and phases checked at clip indices {0,1,15,16,40,last});
   `load_split` conventions + vectorized-positions cross-check pass.

2) **Budget training on the real dataset** (5000/250 eps) — memory arm, defaults except batch:
   ```
   train_sym.py --budget-s 60 --batch-size 8 --sched-steps 100 --snapshot-at 5
   device=cpu params=7.74M W=16 (PINNED) n_actions=5 codec=onehot(5x35) train_eps=5000
     val_eps=250 clip_len=64 n_ctx choices=[4, 8, 16] mem2mem_frac=1.0 bootstrap=False
     use_ff9=False ff9_k=0 n_memory=4 budget_s=60.0
   [snapshot] step 5 -> .../dynamics_sym_b8_step5.pt (elapsed 36s)
   Epoch 1/50 | steps 9 | elapsed 65.7s | ... train mem2mem 0.16648 (flow 0.1665 ...)
   BUDGET_STOP step=9 elapsed=65.7
   ```
   **CPU pace: ~0.15 steps/s at batch 8** (≈6.5-7 s/step steady-state; first step includes
   ~4 s setup), clip 64, mem2mem rollout-only. At the default **batch 64: ~0.009 steps/s**
   (one step = 110 s — a separate 60 s run BUDGET_STOPped cleanly at step 1). CPU smokes
   should run batch ≤ 8; real budget probes belong on the 4070 once it frees up.
   Checkpoint reloads with keys `['config','model_state_dict']` and the pinned config
   `(max_temporal_length=16, n_actions=5, n_latents=5, bottleneck_dim=35, n_memory=4)`;
   `--snapshot-at 5` side checkpoint written. 7.74M params at default dims.

3) **Adapter end-to-end** (privileged=False, smoke checkpoint):
   `python autoresearch/editable/adapter_sym.py --checkpoint .../dynamics_sym_b8.pt` →
   `episode OK: 65 events, fidelity=0.000, 17 imagination-born cells, band_err=2.12 cells,
   112 positions` in 6.7 s (prefix 48/imag 64). Full eval-sized episode (prefix 192 —
   exercises the long-context prefill past W=16 — imag 768): `860 events, ..., 960 positions`
   in **57 s CPU** (~74 ms/tick). Fidelity 0.0 is expected — the model has 9 optimizer steps;
   the machinery is what was under test.

4) **Vanilla arm** — `--n-memory 0 --budget-s 30 --batch-size 8 --sched-steps 50`: prints
   `n_memory=0 -> vanilla reference arm: forcing mem2mem_frac=0.0`, trains the normal
   windowed loss (14 steps / 30.7 s ≈ **0.46 steps/s** at batch 8 — no rollout
   serialization), `BUDGET_STOP step=14`, checkpoint reloads, adapter episode OK
   (65 events, 112 positions).

5) **frozen_sym gate suites re-run, all green**: `test_env`, `test_policies`,
   `test_datagen`, `test_eval` — `ALL PASS` each (run before and after the build; the sym
   variants touch nothing under `autoresearch/frozen_sym/`).

6) **No forbidden changes** — `git status`: only `autoresearch/editable/{train_sym,adapter_sym}.py`
   are new; `autoresearch/frozen/`, `frozen_sym/`, `driver/`, `src/` untouched. Imports:
   the sym files import only `editable.{model,rollout,adapter_sym}` + the frozen_sym layer
   (env/datagen read-only) — no `src/`, no tokenizer anywhere.

## SYM VARIANTS (2026-07-07 ~02:30, finished by orchestrator — build agent wedged on Monitor)
- train_sym.py / adapter_sym.py written by the delegated agent; verification completed by hand:
  * CPU guard lesson: MSYS drops empty-string env vars — CUDA_VISIBLE_DEVICES="" does NOT reach
    Windows python from Git-Bash (use -1). A "CPU" smoke consequently ran on the GPU and OOM-killed
    the overnight pixel calcurve run at step ~3750 (snapshots 250-2000 + final survived; partial
    curve was sufficient — appearance knee ~3k steps, memory absent at 3750).
  * Sym 20-min probe (1.31M, bs128, fixed n_ctx, seed 0): 237 steps @ 5.08s/step (same cost as
    pixel — same arch; the win is task difficulty), flow 0.171->0.052, clean BUDGET_STOP,
    checkpoint reloads. sheets_sym: first 0.44 / last 0.08 (seed5) — valid crisp frames, OUT band
    roughly tracked, partial scroll logic at 237 steps ~= pixel quality at ~2000-3750 steps.
    Speed headroom: 5s/step for 15 tokens @ 1.3M is dataloader/underutilization-bound — profile
    later (or let the loop find it; generality rule applies).
  * sheets_sym phase bug fixed: moves land at ticks t%5==0 -> per-tick pattern is 4xSTAY then
    move (not move-first).
- Sym datasets (5000+250, T=1024): sha256 first16 — train actions d8cb99f24671ddcc / starts
  476742daa3e2a758; val actions 68c22b9616ba16cc / starts 081d966505c28adf (maps/ep_seeds/
  policy_ids identical to pixel tier — same seeds, by construction).
- OVERNIGHT: detached symcurve run (6h, sched 4200, snapshots 250/500/1000/2000/4000) ->
  autoresearch/runs/symcurve/. Morning: sheets_sym + reduced frozen_sym eval per snapshot -> the
  sym steps-vs-quality curve -> budget decision.
