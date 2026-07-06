# Train mem2mem-rollout memory dynamics on Memory Maze WITHOUT FF9 (memory on, ff9 off)

Requested by Merlin 2026-07-04, right after the vanilla arm landed. Third arm of the memmaze
dynamics campaign: identical to `memmaze-dynamics-mem2mem` (415104) minus the FF9 sufficiency loss —
the memmaze counterpart of the GridWorld `mem2mem-rollout-noff9-fair` result (there: clean no-FF9
MATCHED the FF9 winner; the 50% full-noise rollout mode alone trained memory).

## Config (single-variable ablation vs 415104)
`train_mem2mem.sh 50 4 --lr 1e-4 --no-ff9` + checkpoint/W&B-name overrides:
clip128 bs4 lr1e-4 W=32 512/12/16, n_memory 8, `--mem2mem-frac 1.0 --no-bootstrap`, seed 0.
`--no-ff9` keeps n_memory=8 (config ff9_k stays 3 for ckpt-config parity; the loss term is skipped).
relay-grad-clip OFF (matches 415104); the flag exists if the relay explodes on this longer-relay env.

## Question
Does the GridWorld finding (FF9 unnecessary — rollout noise-mode alone trains memory) transfer to
the real 3D env? Also gates the sparse-memory-tokens design (`tasks/drafts/sparse-memory-tokens.md`):
per-frame FF9 injection is awkward with every-Nth-frame memory, so a working no-FF9 recipe simplifies
that path considerably.

## Done means
Checkpoint `checkpoints/memmaze/dynamics_mem2mem_noff9.pt` pulled + verified, W&B healthy (watch
relay stability without the FF9 scaffold — GridWorld showed init relay explosion ~3x/hop), rollout
sheets via `make_sheets.sh`, provenance in `experiments/memmaze-dynamics/NOTES.md` + EXPERIMENTS
line. Memory CLAIMS wait for the memmaze recall/probe eval — 3-way (vanilla / mem2mem / no-ff9).

## Provenance
- ferranti **job 415143** @ SHA `6858832` (`train_mem2mem.sh 50 4 --lr 1e-4 --no-ff9` + ckpt/W&B
  overrides, --hours 36), submitted 2026-07-04 08:57. W&B transformer-mem2mem/`5ez6niv5`.
  Startup log verified: cache HIT, use_ff9=False, mem2mem_frac=1.0, bootstrap=False, clip128 bs4,
  n_actions=6, 41.04M params, checkpoint = dynamics_mem2mem_noff9.pt (override took).

## Status 2026-07-05 ~11:30 — STILL RUNNING (ep41/50); INTERIM ckpt pulled for Merlin's playtest
- Job healthy at ep41/50 (26h30m elapsed; ~39 min/ep, will finish within walltime, ETA ~17:00).
  Log: ff9 0.0000 every epoch (ablation real), flow 0.0067->0.0066, val(normal) ~0.0049-0.0052,
  LR entering cosine decay (9.76e-05), no relay instability without the FF9 scaffold so far.
- **INTERIM ep41 ckpt pulled** (remote mtime 10:51) -> `checkpoints/memmaze/dynamics_mem2mem_noff9.pt`,
  strict-load OK (41.0M, 0 non-finite; config ff9_k=3 retained as documented — loss was off).
  `play_memmaze.py --selftest 12` PASSES (median 114 ms/step ~9.0 fps).
- **NB: local ckpt is a SNAPSHOT at ep41** — re-pull after the job completes (~ep50 + LR-decay tail)
  before any results/sheets/eval; do NOT let ep41-interim numbers stand in for the final model.
- Note for log readers: `d_unlocked 1/8` is pinned by design under `--no-bootstrap`
  (train_mem2mem.py:195; coarse d has no loss target without the boot term) — not a stalled curriculum.

## Status 2026-07-06 — JOB COMPLETED rc=0; FINAL ckpt pulled + verified
- sacct: COMPLETED 0:0, elapsed 1-07:34:12 (within the 36h walltime). Full 50/50 epochs.
- Final log: val(normal) 0.00508 (best 0.00479 @ep48), flow 0.0063, ff9 0.0000 every epoch
  (ablation real), LR decay tail completed to 1e-06, `train normal: 0.00000` throughout,
  NO relay instability without the FF9 scaffold (GridWorld's init relay explosion did not bite).
- **FINAL ckpt re-pulled** (remote mtime Jul 5 16:32 = job end; replaces the ep41 interim) ->
  `checkpoints/memmaze/dynamics_mem2mem_noff9.pt`. Strict-load OK (41.04M, 0 non-finite,
  512/12/16 n_memory=8 n_actions=6; config ff9_k=3 retained as documented — loss was off).
  `play_memmaze.py --selftest 12` PASSES (median 81 ms/step ~12 fps).
- REMAINING for done: sheets via make_sheets.sh (+ pre64), W&B pass (5ez6niv5), NOTES/EXPERIMENTS
  line; memory CLAIMS wait on the memmaze recall/probe eval (4-way).
