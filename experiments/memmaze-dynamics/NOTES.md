# Memory Maze 9x9 dynamics — experiment notes

Campaign kickoff 2026-07-03 (Merlin). Two arms on the frozen memmaze tokenizer's latents
(`checkpoints/memmaze/tokenizer.pt`, 412635, n_latents=32 bottleneck_dim=16):
1. **vanilla** Dreamer-4 dynamics (n_memory=0) — baseline (task `memmaze-dynamics-vanilla`).
2. **mem2mem ROLLOUT-ONLY** [LOCKED by Merlin]: `--mem2mem-frac 1.0 --no-bootstrap` + FF9
   (the GridWorld 411133 winner config) (task `memmaze-dynamics-mem2mem`).
Trained via the NEW latent disk cache (tokenizer never on the GPU during training; see
`tasks/done/latent-cache-for-dynamics-training.md`).

## Window-invariance probe (GridWorld reference numbers, local 4070, fp32)
`probe_window_invariance.py --frames data/gridworld.npy --offset 8`: latent cos-sim mean **0.9975**
(min 0.977), rel-L2 5.8%; decoded recon window-delta MSE **1.33e-6** vs recon-vs-GT ~8.2e-6 (6x below
the recon error). ⇒ arbitrary-offset slicing of cached latents is safe (Merlin's claim confirmed on
GridWorld; memmaze probe in the prep job). JSON: `probe_window_invariance_gridworld.json`.

## Prep results (job 415098 @ 7d86b8d, rc=0, ~8 min)
- **npz keys:** action(1001,6 one-hot), agent_dir(1001,2), agent_pos(1001,2), image,
  maze_layout(1001,9,9), reset, reward, target_color(1001,3), target_pos, target_vec,
  targets_pos(1001,3,2), targets_vec — ALL extracted to `data/memmaze9x9_<key>.npy` on /weka.
  **actions: (2900,1001) int64, n_actions=6** (`memmaze9x9_actions.npy`, auto-detected by trainers;
  pulled local along with agent_pos/agent_dir). maze_layout & targets stay on /weka for the eval task.
- **Latent cache built:** `data/memmaze9x9.latents-fe2ff8440036.npy` (2900,1001,32,16) fp16, 452s
  (6.4 eps/s, cache-batch 16).
- **Memmaze window-invariance probe (offset 32 vs 0, 8 eps):** latent cos **0.999648** (min 0.9978),
  rel-L2 2.5%; recon MSE window-delta **1.19e-6** vs recon-vs-GT 7.3e-5 (**60x below the recon
  error** — even stronger than GridWorld). Arbitrary-offset slicing confirmed safe on memmaze.
  JSON: `probe_window_invariance_memmaze9x9.json` (pulled local).

## Calibration (bs/throughput, H100, synthetic latents)
- Round 1 (job 415100 @ 37330e6): **vanilla** 41.0M @ 512/12/16 W=32: bs64 **0.455 s/step,
  140.6 clips/s, 42.6GB** (bs128 OOM) -> use bs64. **mem2mem OOM @ bs8/clip128**: the rollout keeps
  every slide's graph until one backward (~7 slides + FF9 forwards ≈ bs×2×clip activation footprint).
- Round 2 (job 415101 @ 965268b): mem2mem clip128: bs4 = 0.462 s/step, **8.7 clips/s, 54.0GB** (bs6
  OOM); clip96: bs6 14.8 clips/s 51.8GB; clip64: bs8 33.4 clips/s 32.1GB. **Chosen: clip128 bs4**
  (keeps the GridWorld winner's 4x-window relay ratio; ~37 min/epoch).

## Run config (50 epochs both arms = GridWorld parity; window 32; 512/12/16 = 41.0M params)
- vanilla: bs64, lr 3e-4 (default), ~8.5h -> --hours 12.
- mem2mem rollout-only [structure LOCKED]: clip128 bs4, **lr 1e-4** (agent judgment: bs 16x smaller
  than GridWorld's 64 -> ~sqrt-scaled from 3e-4; conservative vs the init relay explosion at 512-d;
  n_ctx sampled from {4,8,16,32}), n_memory 8, ff9 3, --no-bootstrap, ~31h -> --hours 36.
- Config choices 512/12/16 + W32 + n_memory 8 were proposed to Merlin (AFK at decision time) —
  REVERSIBLE; he can cancel + override.

## Vanilla arm LANDED (2026-07-04)
- Job **415103 COMPLETED rc=0**: 50 epochs in 8h31m (613 s/ep), final **train 0.00597 / val 0.00431**
  (diffusion loss only — vanilla). W&B `wj0dcogd`: steady val descent (still improving slowly at ep50),
  grad-norm decayed cleanly, no spikes/instability. Checkpoint pulled + load-verified locally
  (`checkpoints/memmaze/dynamics_vanilla.pt`, config == locked 512/12/16 W32, n_actions=6, n_memory=0,
  41.0M params).
- **Qualitative rollout sheets** (task `memmaze-rollout-sheets`, DONE): NEW spec-backed
  `src/evals/memmaze/sheets.py` (+ `specs/evals/memmaze/sheets.md`) — TOP=GT / BOTTOM=action-conditioned
  free-run on HELD-OUT episodes (reproduces the trainer's seed-0 val split), reuses the gridworld
  drawing layer. Render job (`make_sheets.sh`): ferranti **415142** @ `306e147` (rc=0) — in-window
  (8ctx|24gen) + past-window (8ctx|56gen) sheets + a 12-episode val frames/actions slice.
- **Sheet findings (vanilla, eyeballed 2026-07-04)** — `sheets_vanilla/_sheet_rollout_{in,past}_window.png`
  (gitignored, `_` prefix):
  - Context reconstructions crisp; rollouts locally coherent (walls/floor/horizon geometry, textures,
    objects) and visibly action-responsive (turns match the action digits in the labels).
  - Rollout diverges from GT within a few steps (wrong wall colors/layout, objects dropped) — EXPECTED
    for a no-memory model in a partially-observed maze; beyond the 8-frame context the true maze is
    unknowable. This is the baseline picture the memory arm must beat.
  - Past-window (56 gen, window slides twice): STABLE — no black-collapse/explosion; quality softens
    deep into the rollout, drifting toward a washed-out pale-green "generic wall" mode (mild
    mode-averaging) while staying scene-like and action-responsive.
- **Local iteration enabled:** `data/memmaze9x9_val12{,_actions,_ids}.npy` (12 held-out episodes,
  148MB, episodes [1544,1459,121,2876,1623,2639,2075,1855,729,875,451,2272]) — the sheets CLI verified
  locally on the 4070 against this slice (`--frames data/memmaze9x9_val12.npy --episodes 0 1 ...`;
  NOTE: ids in the slice are POSITIONS 0..11, not original episode ids).

## Third arm: mem2mem no-FF9 (launched 2026-07-04, Merlin)
Memory on, FF9 off — the memmaze counterpart of GridWorld `mem2mem-rollout-noff9-fair` (there the
clean no-FF9 arm MATCHED the FF9 winner). Single-variable ablation vs 415104: identical
`train_mem2mem.sh 50 4 --lr 1e-4` plus `--no-ff9`, ckpt/W&B overridden ->
`checkpoints/memmaze/dynamics_mem2mem_noff9.pt`. ferranti **job 415143** @ `6858832`, --hours 36,
W&B transformer-mem2mem/`5ez6niv5`. Startup verified clean: cache HIT, use_ff9=False,
mem2mem_frac=1.0, bootstrap=False, clip128 bs4, n_actions=6, 41.04M, checkpoint override took (no
clobber of the running 415104 ckpt). Watch: relay stability without the FF9 scaffold. Also gates the
sparse-memory design (`tasks/drafts/sparse-memory-tokens.md`).

## Provenance
- Prep: ferranti **415098** @ `7d86b8d`. Calibration: **415100** @ `37330e6`, **415101** @ `965268b`.
- Training jobs (submitted 2026-07-03 22:01, both @ SHA `1149bb4`):
  - vanilla: ferranti **job 415103** (`train_vanilla.sh 50 64`, --hours 12) ->
    `checkpoints/memmaze/dynamics_vanilla.pt`, W&B transformer-D-dynamics/memmaze-dyn-vanilla.
  - mem2mem rollout-only: ferranti **job 415104** (`train_mem2mem.sh 50 4 --lr 1e-4`, --hours 36) ->
    `checkpoints/memmaze/dynamics_mem2mem.pt`, W&B transformer-mem2mem/memmaze-dyn-mem2mem.
- NEW wrapper `scripts/clean_untracked.sh` (git-clean specific untracked remote paths) — needed
  because the prep job's probe JSON blocked sync_code checkout after being pulled + committed.
