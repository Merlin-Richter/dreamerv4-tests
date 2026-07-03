# Memory Maze dynamics prep: actions + labels extraction, latent cache build, invariance probe

Requested by Merlin 2026-07-03 (prereq for the two memmaze dynamics training tasks).

## Steps (one ferranti job)
1. **Extract actions + eval labels from the raw npz** (`data/memmaze9x9_raw/` on /weka; the tokenizer
   converter kept only `image`). First print the key list of one npz, then extract per-key npys aligned
   with `data/memmaze9x9.npy` episode order (sorted rglob, same as convert_memmaze.py):
   - `memmaze9x9_actions.npy` (N, T) int64 (argmax if stored one-hot) — for action conditioning.
   - Label arrays for the future recall/probe eval: `agent_pos`, `agent_dir?`, `targets_pos`,
     `maze_layout`, `target_*` — whatever exists, each as its own npy/npz.
2. **Build the latent cache** for (tokenizer.pt @ 412635, memmaze9x9.npy) via
   `train_dynamics.py --build-latent-cache-only` (~10-20 min encode, fp16, ~3GB).
3. **Window-invariance probe on memmaze**: encode a few episodes at window offsets 0 vs 32, report
   latent cos-sim/MSE on overlapping frames + decoded-recon delta. Record numbers in NOTES.
4. Pull actions + labels + (optionally) the 3GB latent cache to local for 4070 iteration.

## Done means
Actions/labels npys exist on cluster (+ pulled), latent cache exists on /weka, probe numbers recorded
in `experiments/memmaze-dynamics/NOTES.md`, provenance (job id + SHA) recorded.

## Provenance
- ferranti job 415098 @ SHA 7d86b8d (experiments/memmaze-dynamics/prep.sh, --hours 2 --cpus 8), submitted 2026-07-03. RISK flagged in prep.sh: fails fast if data/memmaze9x9_raw was cleaned (re-download via memmaze-tokenizer/cluster_prep.sh then re-run).

## RESULT (2026-07-03)
DONE — job 415098 @ 7d86b8d rc=0 (~8 min). All 12 npz keys extracted to /weka (actions (2900,1001) int64 n_actions=6; agent_pos/dir/maze_layout/targets for the future eval); latent cache built (2900,1001,32,16) fp16 in 452s; memmaze window-invariance probe: cos 0.9996, window-delta recon MSE 60x below recon error (claim confirmed). actions/agent_pos/agent_dir/probe-json pulled local.
