# Train mem2mem-rollout memory dynamics on Memory Maze latents

Requested by Merlin 2026-07-03. Runs in parallel with `memmaze-dynamics-vanilla`.

## Goal
Port the GridWorld headline win to the real 3D memory env: memory-token dynamics trained with the
mem->mem sliding-rollout signal (`experiments/mem2mem/train_mem2mem.py` + `rollout.py`), on the cached
memmaze latents. **[LOCKED by Merlin 2026-07-03]: the structure to train is mem2mem ROLLOUT-ONLY** —
the 411133 winner config: `--mem2mem-frac 1.0 --no-bootstrap` + FF9 (no normal-window batches).
relay-grad-clip OFF by default (watch for the init relay explosion on this longer-relay env; the
`--relay-grad-clip` flag exists if training is unstable — ORIENT 2026-06-29).

## Prereqs
- `latent-cache-for-dynamics-training` (mem2mem trainer consumes the cache).
- `memmaze-dynamics-prep` (actions npy + latent cache on cluster).

## Open config decisions (ask Merlin with throughput data before submitting)
- Same model-size question as vanilla (keep the two arms' transformer config IDENTICAL for a fair
  comparison; only memory/FF9/training-signal differ).
- mem2mem specifics: frac LOCKED at 1.0 (rollout-only, see Goal); still open: `--clip-len` (rollout
  length; GridWorld used 64 = 4x window), n_ctx choices, `--ff9 K --n-memory M`.
- Compute budget: rollout training is ~sequential over the clip -> slower per sample than vanilla.

## Done means
Trained checkpoint `checkpoints/memmaze/dynamics_mem2mem.pt` pulled + verified, W&B healthy (watch
relay stability), qualitative rollout sheet vs vanilla, provenance in
`experiments/memmaze-dynamics/NOTES.md` + EXPERIMENTS.md line. Memory CLAIMS wait for the memmaze
recall/probe eval (follow-up task) — sheets illustrate, never decide.

Sheets tooling READY (2026-07-04, task `memmaze-rollout-sheets` done): when 415104 lands, submit
`bash experiments/memmaze-dynamics/make_sheets.sh checkpoints/memmaze/dynamics_mem2mem.pt
runs/memmaze-sheets-mem2mem` (~1 min job), pull, compare against
`experiments/memmaze-dynamics/sheets_vanilla/`. NB: mem2mem ckpts save the same
{config, model_state_dict} payload, so the CLI loads them unchanged; local renders work on the 4070
via `data/memmaze9x9_val12.npy` (`--episodes 0..11` = positions in the slice).

## Provenance
- ferranti job 415104 @ SHA 1149bb4 (train_mem2mem.sh 50 4 --lr 1e-4, --hours 36), submitted 2026-07-03. clip128 bs4 lr1e-4 W32 512/12/16 n_memory8 ff9_3 no-bootstrap (41.0M). ~31h ETA. lr 1e-4 = agent judgment (bs 16x smaller than GridWorld) — flagged to Merlin.

## Status 2026-07-05 — landed INCOMPLETE (TIMEOUT ep28/50), interim ckpt pulled + verified
- sacct: **TIMEOUT at 1-12:00** (36h walltime) as the nightlog predicted; last completed epoch **28/50**.
- Log healthy through ep28: val(normal) 0.00848 -> ~0.0039-0.0041 plateau (below vanilla 415103's
  0.00431 — note mem2mem never trains the normal loss); flow 0.0423 -> 0.0068 still descending;
  ff9 0.163 -> 0.079; no instability, relay-grad-clip never needed. W&B t4ppeqzp.
- **Ckpt pulled** (ep28 interim, remote mtime 09:35) -> `checkpoints/memmaze/dynamics_mem2mem.pt`,
  strict-load OK vs its saved config (41.0M, n_memory=8 ff9_3 W32 512/12/16 n_actions=6, 0 non-finite
  tensors). `play_memmaze.py --selftest 12` PASSES on it (median 150 ms/step ~6.7 fps > 6 target;
  slower than vanilla's ~9 fps — 8 extra memory tokens/frame).
- OPEN (Merlin, from the nightlog decision packet): resume 415104 for the last 22 ep vs accept ep28
  vs faster recipe. flow/ff9 still descending at kill => more epochs would help, but ep28 is usable.
- Remaining for done: W&B pass, sheets (`make_sheets.sh ... dynamics_mem2mem.pt`, + pre64) vs
  `sheets_vanilla/`, EXPERIMENTS line + NOTES provenance — after Merlin's resume/accept call.
