# ORIENT.md

Rewritten: 2026-06-26.

## IN FLIGHT (2026-07-03 — Memory Maze dynamics campaign, kicked off by Merlin)
Goal: train TWO dynamics arms on the frozen memmaze tokenizer's latents — **vanilla** (baseline) and
**mem2mem ROLLOUT-ONLY [LOCKED by Merlin]** (`--mem2mem-frac 1.0 --no-bootstrap` + FF9, the 411133
winner). Merlin DELEGATED spec edits + task management for this campaign (memory:
feedback-spec-edit-delegation). Done this session:
- **Latent disk cache** (task done): train_dynamics.py + spec — tokenizer encodes once per
  (frames, tokenizer) combo into fp16 `<frames>.latents-<sha12>.npy`; training never holds the
  tokenizer (VRAM + compute + 12x dataset shrink). `--encode-online` / `--build-latent-cache-only` /
  `--cache-batch`. mem2mem trainer ported. Window-invariance probe: GridWorld latent cos 0.9975,
  window-delta recon MSE 6x below recon error ⇒ arbitrary-offset slicing safe.
- **Model-dim CLI args** exposed in both trainers (+spec): `--embedding-dim/--depth/--n-heads/
  --n-registers` (+ `--context-length` in mem2mem trainer).
- **Prep DONE** (415098 @ 7d86b8d): actions (2900,1001) int64 **n_actions=6** + ALL labels extracted
  (agent_pos/dir, maze_layout, targets... — eval raw material, on /weka; small ones pulled local);
  latent cache `memmaze9x9.latents-fe2ff8440036.npy` (fp16, 3GB) built; **memmaze window-invariance:
  cos 0.9996, window-delta recon MSE 60x below recon error** (claim confirmed).
- **Calibration** (415100+415101): vanilla bs64 = 140 clips/s 42.6GB; mem2mem clip128 bs4 =
  8.7 clips/s 54GB (bs6+ OOM — the rollout holds all ~7 slides' graphs to one backward).
- **vanilla arm DONE (2026-07-04, task -> done)**: 415103 rc=0, 50ep 8h31m, train 0.00597 /
  **val 0.00431**, W&B wj0dcogd healthy (steady descent, clean grad-norm). Ckpt pulled +
  load-verified (`checkpoints/memmaze/dynamics_vanilla.pt`, locked config, 41.0M). EXPERIMENTS:
  `memmaze-dyn-vanilla`. (Recall scoring = separate follow-up once the memmaze eval exists.)
- **Rollout sheets DONE (2026-07-04, task `memmaze-rollout-sheets` -> done)**: NEW spec-backed
  `src/evals/memmaze/sheets.py` + `specs/evals/memmaze/sheets.md` (TOP=GT / BOTTOM=action-conditioned
  free-run, HELD-OUT val-split episodes, reuses gridworld drawing layer; smoke-tested local).
  Vanilla sheets (render job **415142** @ `306e147`) pulled to
  `experiments/memmaze-dynamics/sheets_vanilla/` + eyeballed: ctx recon crisp, rollout coherent +
  action-responsive, diverges from GT within a few steps (expected — no memory), past-window (64
  frames, 2 slides) STABLE, drifts to a washed-out generic-wall mode (no collapse). NOTES.md has the
  findings. Local iteration enabled: `data/memmaze9x9_val12{,_actions,_ids}.npy` (12 held-out eps).
- **Decisions made while Merlin AFK (all REVERSIBLE, cancel+resubmit possible):** 512/12/16, W=32,
  n_memory=8, 50ep, mem2mem lr 1e-4. New wrapper `scripts/clean_untracked.sh` (untracked remote
  artifacts blocking sync_code checkout).
- **STILL RUNNING: mem2mem rollout-only 415104** (clip128 bs4 **lr 1e-4** [agent call, flagged],
  n_memory 8, ff9 3, no-bootstrap, ETA ~31h from 2026-07-03 22:00, --hours 36, W&B t4ppeqzp) ->
  `checkpoints/memmaze/dynamics_mem2mem.pt`. @ SHA `1149bb4`, same 512/12/16 W=32 transformer.
- **STILL RUNNING: mem2mem NO-FF9 415143** (launched 2026-07-04 by Merlin's ask; memory on, ff9
  off — single-variable ablation vs 415104, `--no-ff9` + ckpt/W&B overrides) @ SHA `6858832`,
  --hours 36, W&B 5ez6niv5 -> `checkpoints/memmaze/dynamics_mem2mem_noff9.pt`. Startup verified
  (use_ff9=False, no ckpt clobber). Task `memmaze-dynamics-mem2mem-noff9`.
- **Long-context prefill DONE (2026-07-04, Merlin's ask; task `sheets-long-context-prefill` -> done,
  @ `4f3e9bf`)**: `rollout_init`/`generate` accept T_ctx > W (teacher-forced sliding commits,
  written-memory relay; spec'd + gate-tested). Memmaze sheets now PREFILL **n_pre=64** true frames
  and display only the last 8 ("64 ctx (8 shown)") — with 8-frame context the env was impossible by
  construction. pre64 vanilla sheets: job 415145 -> `sheets_vanilla/*_pre64.png` (early GT-tracking
  modestly better, still drifts — the setup where mem2mem must show its edge). make_sheets.sh
  renders pre64 for the arms landing next.
- **Playable memmaze-in-the-world-model DONE (2026-07-04, Merlin's ask; task
  `playable-memmaze-rollout` -> done)**: NEW spec-backed `src/interactive/play_memmaze.py` — the
  pygame twin of `external/memory-maze/gui/run_gui.py` (same keymap/pacing) but rendered by the
  dynamics carrying rollout on the local 4070 (~9 fps > 6 fps target). Reset = full context window
  (default W=32, up to 64) of real val12 frames committed via rollout_init + on-screen replay, then
  live `rollout_step` per held-key tick. Selftests green (dummy + windowed + n_ctx 64). pygame now
  in the repo venv. Use for eyeballing the mem2mem arms when they land.
- **Sparse-memory design DRAFTED (Merlin's ask): `tasks/drafts/sparse-memory-tokens.md`** — memory
  tokens only every Nth frame. Core: temporal attention is slot-wise, so presence-sparsity needs a
  broadcast-read (memory K/V as extra keys for every slot); writes stay complete iff N ≤ W−1 (every
  frame in-window for ≥1 write). Phased: Design B write-sparsity prototype on GridWorld via
  --model-module (no arch change) → Design A presence-sparsity in src/+spec (needs Merlin sign-off)
  → memmaze arm → eviction-exempt memory bank extension. No-FF9 arms gate the training recipe.
**NEXT when 415104/415143 land:** pull ckpts, W&B check (relay stability / flow+ff9 balance),
sheets via `make_sheets.sh ... dynamics_mem2mem[_noff9].pt runs/memmaze-sheets-...`, compare vs
`sheets_vanilla/`; then the memmaze recall/probe eval task (labels ready on /weka) — memory CLAIMS
wait for that eval (3-way: vanilla / mem2mem / no-ff9).

## NEW FINDING (2026-07-04 — vanilla in-window diagnosis, Merlin's question; task -> done)
Vanilla GridWorld dynamics can't predict square POSITION even in-window/fully-revealed (per
sheet_normal.png) because the diffusion-forcing objective barely ever demands prediction-from-
context: (GT flow target AND tau<=0.1) = 1.3% of frames = **0.4% of ramp-weighted loss**; the 25%
of frames at tau_idx=0 get the bootstrap SELF-distill target; at the other ~75% of tau the square
is readable from the frame's own noisy latent -> the model learns denoise+colors, never dynamics.
PROVEN by 2 local probes (`experiments/vanilla-inwindow-diagnosis/`): teacher-forced 1-step from
real revealed context = ~chance for vanilla at every ctx len, ~1.0 for ff9/mem2mem/no-ff9; and ff9
in a PLAIN forward (no carried memory) = 1.0 @ tau=0 => architecture innocent, objective guilty.
**CONFOUND FLAG for Merlin:** current "vanilla" is a weak baseline IN-window, so vanilla-vs-memory
recall gaps overstate the memory effect — incl. the memmaze vanilla arm 415103 (its early-drift
sheets partly this, not just "no memory"). Honest-baseline fix options (need spec edit, Merlin
decides): force (tau_idx=0, d=d_min, GT flow) mass into sample_tau_d, and/or drop the ramp on the
d_min flow term (deterministic envs), and/or a "vanilla-rollout" arm (mem2mem training minus
memory tokens) for a clean 2-factor design.

## DONE (2026-06-30 — Memory Maze 9x9 tokenizer) — FROZEN, ready for dynamics
Full run **412635 COMPLETED rc=0** (15ep bs6 LOCKED cfg, 10h16m, @ `be1258e`). **Final: val MSE 0.000074
(fg 0.00013 / bg 0.000033), latent_cos 0.235 (NO collapse), pred_std 0.16, 25/6887 skips — healthy.**
Checkpoint pulled + load-verified (config == LOCKED 512/12/16, n_latents=32 bottleneck_dim=16 L=64, 82.3M
params) -> `checkpoints/memmaze/tokenizer.pt`, **FROZEN**. Recon sheet pulled
(`experiments/memmaze-tokenizer/_recon_memmaze.png`, gitignored; 6 real-vs-recon strips, pixel MSE
0.00008): geometry/colors/objects (incl. the blue target sphere) faithful; only high-freq wall texture
slightly smoothed — acceptable at the 512-d/frame bottleneck. W&B `o9ldtn6t`. Task -> done; full details
in `experiments/memmaze-tokenizer/NOTES.md`.
**NEXT (needs Merlin's task):** Memory Maze dynamics + memory model on these latents (dims must match
n_latents=32 bottleneck_dim=16) + a Memory-Maze recall/probe eval built from the npz labels
(`agent_pos`, `maze_layout`, `target_*`). Note: val MSE was still descending at ep15 — more
data (train-part1..9) / epochs would sharpen further if ever needed (reversible).

## DONE (2026-06-29 — FAIR no-FF9 ablation) — FF9 is NOT necessary on GridWorld
Both arms completed clean (412506 no-norm, 412510 +relay-grad-clip 0.05; winner-config-minus-FF9, 50ep).
**Recall w8 max_k64 position_acc: Arm 1 (no norm) K4 0.989 / K2 0.999 / K1 0.999; Arm 2 (relay-clip) K4
0.985 / K2 0.996 / K1 1.000; winner WITH FF9 0.992; old confounded 411270 0.044; vanilla 0.042.** Clean
no-FF9 matches the FF9 winner and is flat to k=64 ⇒ the 50% full-noise rollout mode ALONE trains memory;
the 411270 "FF9 necessary → chance" was the CONFOUNDS (bootstrap+curriculum+instability+36ep), not a
missing FF9 (Merlin vindicated). Relay gradient verified to flow behind the window (probes), but EXPLODES
~3×/hop at init (88 @W=4) — the per-hop relay grad-norm (arm 2) tamed it (clip 0.133 epoch1 → 0.000 after)
yet was ~neutral on recall (stable d_min config rode out the transient via global clip); keep it OFF-by-
default for harder/longer-relay envs (Memory Maze). Residual FF9 edge only on long-horizon ball COLOUR
(~0.95 vs ~0.8 @k64). Visuals in `experiments/mem2mem-rollout-noff9-fair/` (compare_w8_k64_noff9.png +
occlusion sheets). Task → done. EXPERIMENTS: `mem2mem-rollout-noff9-fair`.

## (superseded) IN FLIGHT (2026-06-29 — FAIR no-FF9 ablation, TWO parallel ferranti jobs 412506 + 412510)
Re-testing the 411270 "FF9 is necessary" result, which Merlin flagged as conceptually off (the 50%
full-noise rollout mode should train memory even without FF9, *if* the relay gradient flows back behind
the window). **Investigation:** (1) the relay gradient IS healthy — `test_autograd.py` passes; a
training-scale probe (`probe_relay_grad.py`, real config, use_ff9=False) gives init-only-frame |grad|
**0.499 relay-on / 0.0 detached**. So no-FF9 collapse is NOT a severed-gradient bug; **411270 was
CONFOUNDED** (bootstrap+curriculum+instability+36ep). (2) BUT the relay gradient **EXPLODES backward at
init** (`probe_relay_decay.py` / `measure_clip_scale.py`: ~2–3×/hop; real-data deepest-hop |grad| 0.03
@W=16 but 26 @W=8, **88 @W=4**), self-correcting to ~1 once trained — a candidate reason no-FF9 fails
(early explosion + global grad-clip may stop the representation forming without FF9's dense scaffold).
**No dedicated relay normalizer exists** (carried memory = raw residual stream; only TBPTT(2N) + global
clip + pre-norm + training self-regularize). So running TWO arms in parallel:
- **Arm 1** (clean, winner-minus-FF9): `--no-bootstrap --no-ff9 --mem2mem-frac 1.0` 50ep → job **412506**
  @ SHA `8f54d09`, ckpt `dynamics_mem2mem_rollout_noff9_clean.pt`.
- **Arm 2** (+ NEW per-hop relay grad-norm): adds `--relay-grad-clip 0.05` (backward hook scales each
  carried memory's gradient down per-seq to ≤C; training-only, OFF=byte-identical, forward/inference
  unchanged) → job **412510** @ SHA `e266bea`, ckpt `dynamics_mem2mem_rollout_noff9_clip.pt`.
NEXT when they land: pull both + 4-way recall w8 max_k64 (K=4/2/1) vs winner (with FF9) + old 411270 +
baselines. Pre-registered: arm1 near-ceiling ⇒ FF9 not needed (Merlin vindicated); arm1 chance & arm2
near-ceiling ⇒ the relay explosion was the blocker, normalizer rescues it; both chance ⇒ FF9 is a needed
dense scaffold. Task `tasks/in-progress/fair-noff9-ablation.md`; NOTES `experiments/mem2mem-rollout-noff9-fair/`.

## What we're doing right now and why
Rebuild merged + spec→code sync done. Two campaigns just completed — both WINS:
1. **Retrain r2 (DONE):** dynamics with 5× data (5000 eps) + fixed LR schedule (warmup→flat→cosine
   80-100%). Fixed the r1 "position null" — FF9 position recall went from chance to near-perfect to
   k≈12 (then decays). val/loss 0.0058→0.0016. (NB: a PEP-604 `str|None` in the --model-module seam
   crashed on cluster py<3.10; fixed with `from __future__ import annotations`, 73b1c65.)
2. **mem→mem training (DONE, `experiments/mem2mem/`, `tasks/done/test-new-memory-training.md`):** new
   training signal teaching memory tokens to be built from prior memory tokens. Autograd check passes
   (relay grad 3.25e-3, 0.0 when detached). **Result: mem→mem holds position recall ~0.96 FLAT to k=20**
   where FF9 decays to 0.14 — long-horizon tail (k≥14) pos_acc vanilla 0.03 / FF9 0.20 / mem→mem 0.96.
   It carries hidden state indefinitely past the window — the core project goal demonstrated on GridWorld.

## Checkpoints (in `checkpoints/gridworld/`, all SHA-1688818-era, frozen tokenizer)
`tokenizer.pt`, `dynamics_vanilla.pt` (chance recall, no memory), `dynamics_ff9.pt` (recall to k≈12),
`dynamics_mem2mem.pt` (recall flat to k=20). Recall driver: `experiments/recall-ab/run.py` (3-way).

## Background: the spec→code sync (`f91e2a0`)
temporal cadence every-4th→every-3rd (`3×[spatial,temporal,spatial]`, depth 8→9), dynamics attention
gained the learnable per-head `logit_scale`, tokenizer decoder gained a sigmoid output bound, FF9
terminal frame gets a sampled τ, trainers moved to required `--frames/--tokenizer/--checkpoint`. **All
pre-sync checkpoints are architecture-incompatible — retrained.**

## DONE (2026-06-27→29, part 3 — FAIR bootstrap A/B) — bootstrap is FREE, not harmful
On `exp/mem2mem-rollout-only` @ SHA `851a7ab`. Merlin pushed back on the part-2 "bootstrap hurt" claim
(shortcut forcing logically shouldn't hurt). The 411221 negative was CONFOUNDED by 3 changes riding with
the bootstrap — (1) FF9 normalizer diluted by the small boot self-distillation term (mixed flow+boot mean
vs pure flow → down-weights memory ~2.4×; the WINNER was the non-standard pure-flow one), (2) new-half τ
resampled onto the d-snapped grid (~25% at τ=0; intrinsic to bootstrap), (3) 36 vs 50 ep. Ran a clean
2-arm factorial (both 50ep, FF9 normalized by pure d_min flow via `--ff9-norm-flow`; τ held identical via
curriculum in both): **Arm B (fair boot) 411502** vs **Arm A (control `--boot-loss-off`) 411503**, ferranti.
**RESULT (recall w8 max_k64, position_acc): Arm A (boot OFF) K4 0.998 / K2 1.000 / K1 1.000; Arm B (fair
boot) K4 0.968 / K2 0.980 / K1 0.999; winner (411133) 0.992; old unfair boot (411221) 0.472 — the collapse
is GONE.** Pre-registered verdict confirmed: B≈A≈winner (no halving — old negative was the confounds, esp.
the FF9-norm dilution: final FF9 0.013/0.0105 vs old boot's 0.054); B<A is small/consistent (~3pts, not
catastrophic); A≥winner so the τ-shift is benign. Few-step: Arm A is already perfect @K1/K2, so the
bootstrap's whole motivation is moot on GridWorld. **Decision (GridWorld only) — keep simple rollout-only +
FF9 + x-prediction; the bootstrap is safe but buys nothing HERE.** SCOPE: all arms keep full diffusion
forcing; the A/B toggles only the shortcut *bootstrap distillation* on coarse steps, NOT diffusion on/off.
No upside because GridWorld is DETERMINISTIC (delta next-state ⇒ single-step x-pred = the mean = correct).
On a STOCHASTIC env, one-step x-pred mode-collapses to the conditional mean ⇒ multi-step diffusion is
required and the bootstrap keeps few-step sampling on-trajectory. Do NOT read this as "drop shortcut
forcing"; validating it needs a stochastic env (`tasks/drafts/harder-grid-env.md`). Ckpts pulled
(`dynamics_mem2mem_rollout_boot_fair.pt`, `_bootctrl.pt`); `experiments/mem2mem-rollout-boot-fair/` (NOTES +
`compare_w8_k64_4way.png`). Task → done. galvani socket still DOWN; ferranti UP (check via WSL).

## Just finished (latest session, 2026-06-27, part 2 — bootstrap + FF9 ablations)
On branch `exp/mem2mem-rollout-only`. Added shortcut **bootstrap distillation** to the rollout new-half
loss (audited correct, EXPERIMENTS `V-newhalf-loss`) + a finest-first **step-size curriculum**, and ran
two ferranti jobs (~3h each, rollout-only):
- **mem2mem-rollout-boot (411221, flow+FF9+bootstrap+curriculum, 36ep):** NEGATIVE. Recall HALVED vs the
  no-boot winner (position_acc @K=4 0.47 vs 0.99; @K=1 0.50 vs **0.999**). Final FF9 loss 0.054 vs 0.010
  (5× worse memory sufficiency) — the bootstrap lowered flow, and the flow/ff9 normalization then
  down-weighted FF9, undertraining memory. The plain no-boot model already does K=1 near-perfectly, so the
  bootstrap was unmotivated here. **Keep the simple rollout-only + FF9 (no bootstrap).**
- **mem2mem-rollout-noff9 (411270, --no-ff9, 36ep):** FF9 NECESSARY. Without it recall = chance at every k;
  training also unstable. The rollout flow loss alone (even the 50% full-noise mode) does not carry position.
New code is behind flags (`bootstrap=`/`--no-bootstrap`, `n_d_unlocked`/`--no-curriculum`,
`use_ff9=`/`--no-ff9`); defaults ON but the winner config is `--no-bootstrap` (or just the original code).
New tool `scripts/pull_file.sh` used to pull both checkpoints.

## Just finished (latest session, 2026-06-27, part 1)
- **mem2mem rollout-only experiment DONE — WIN** (`experiments/mem2mem-rollout-only/`, task done). Trained
  with `--mem2mem-frac 1.0` (the mem→mem sliding rollout ALONE, no normal-window batches; log confirms
  `train normal: 0.00000`). ferranti job 411133 (2h51m, 50ep), ckpt `dynamics_mem2mem_rollout.pt`.
  Recall @ window=8 max_k=64 position_acc (mean / tail k≥14 / k=64): rollout-only **0.992 / 0.988 / 1.000**
  — FLAT to k=64, ≥ the 50/50 model (0.988/0.984/0.984), ≫ FF9 (0.375/0.111) and vanilla (0.040/0.033).
  **The normal shortcut-forcing loss is not needed for the retention win — the mem→mem rollout is sufficient.**
  Minor cost: in-window val(normal) 0.005 vs 0.0027. Work on branch `exp/mem2mem-rollout-only` (NOT merged
  to master). EXPERIMENTS: `mem2mem-rollout-only`.
- **New cluster wrapper `scripts/pull_file.sh`** — rsync ONE arbitrary repo-relative file (e.g. a checkpoint
  in `checkpoints/<env>/`) back from the cluster; fills the gap where `pull_results` only syncs `runs/<run>/`
  (previously needed a cp-into-runs hack). Tested (happy path + BAD_REF/BAD_CONFIG error paths). README updated.

## Just finished (prior session, 2026-06-26)
- **Working tree committed as a verified rollback point** (`786e6ce`): the in-flight `--window`/`max_ctx`
  eval tooling (dynamics `rollout_init`/`generate` knob, recall `window=` + CLI + meta block, NEW
  `plot_recall.py` + `sheets.py` with specs), CLAUDE.md docs, and done→archive task moves. Verified
  specs↔src consistent for every changed/new file; all 4 gate suites green. Closed task
  `check-git-changes-in-specs-and-src` (→done). Minor non-blocking drift noted: `sheets.py` CLI has extra
  `--actions`/`--occ-seed0` not in its spec.
- **Cluster: BOTH sockets DOWN** (ferranti + galvani, confirmed via WSL). Needs Merlin `open_master.sh`.
  Backlog task `retrain-mem2mem-with-only-rollout-training` (rollout-only vs 50/50; test window=8 max_k=64)
  is BLOCKED on this.

## Earlier this session
- **Cache-equivalence verification** (task → `tasks/done/verify-cached-vs-uncached-rollout-identical.md`).
  New gate test `src/tests/test_dynamics_cache.py` (green) + `experiments/verify-cache-equiv/`. Result,
  independently re-verified (critical-claim-verifier, fp64): the carrying KV-cached rollout `==` an
  uncached current-window recompute **bit-exact within the window**, but **diverges materially (O(1), not
  fp) once the sliding window evicts** — because ≥2 stacked temporal layers freeze each committed frame's
  deep-temporal K/V at its commit-time receptive field. Clean dichotomy: 1 temporal layer → exact;
  ≥2 → diverges. So the cache is a correct optimization *within* a window but **not** past it.
  EXPERIMENTS.md: `V-cache-equiv`. HOWTO updated (`rope_kv_cache_caveat.md`).

## Open decisions flagged for Merlin (do not silently resolve)
1. ~~Cache train/inference semantics gap~~ — **RESOLVED 2026-06-26**: the post-eviction divergence IS
   the intended information-preservation mechanism (Merlin). Recall correctly measures the carried path.
2. **Recall `k`↔tick alignment** (documented atop `src/evals/gridworld/recall.py`) — still needs sign-off
   before recall numbers are trusted. **This now gates the recall A/B.**

## Models ready (retrain DONE, `tasks/done/retrain-models.md`)
Trained on ferranti @ SHA `0a0e070`, pulled to `checkpoints/gridworld/`: `tokenizer.pt` (fg-weight 2.0,
val fg_mse 1.7e-5, no collapse), `dynamics_vanilla.pt` (n_memory=0), `dynamics_ff9.pt` (n_memory=4,
ff9_k=3). Both dynamics action-conditioned (n_actions=2). EXPERIMENTS: `R-gridworld-retrain`.

## Recall A/B history (`experiments/recall-ab/`)
r1 (under-trained): position null. r2 (5× data + LR fix): FF9 position recall near-perfect to k≈12 then
decays. 3-way (+mem→mem): mem→mem flat ~0.96 to k=20. See `NOTES.md` + `results_*.json`. EXPERIMENTS:
`recall-AB`, `R-gridworld-retrain2`, `mem2mem-train`.

## NEXT (for Merlin to steer)
1. **Recall `k`↔tick alignment** sign-off (decision #2) — still needed before *absolute* k numbers are
   trusted (the model-vs-baseline *comparisons* above are convention-robust).
2. **Graduate mem→mem?** It's the campaign's headline win. If keeping, fold it into `src/`+spec (it lives
   in `experiments/mem2mem/` now). Possible ablations: mem2mem-only vs 50/50; n_ctx schedule; longer
   max_k to find where it breaks; segmented-backward TBPTT (footprint); a harder env than GridWorld.
3. Cluster: ferranti UP, idle (all jobs COMPLETED). **galvani socket DOWN** (needs `open_master.sh
   --cluster galvani` from Merlin).

## Background — the memory research
Frontier question: do per-timestep **memory tokens** let a Dreamer-4 world model retain hidden/off-screen
state past the short latent window? The clean code keeps the two ideas worth keeping: the **FF9
sufficiency loss** (memory must reconstruct future frames from memory alone) and the **carrying KV-cached
inference** (read old memory tokens' cached K/V, write new ones each step). The recall eval is the
result-defining spine — changing it silently redefines results.

## Parked
- Discrete/VQ memory idea (only if the carrying relay still drifts on GridWorld after retraining).
