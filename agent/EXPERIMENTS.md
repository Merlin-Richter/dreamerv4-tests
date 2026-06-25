# EXPERIMENTS — short index

> One line per experiment: what it tested → result. Artifacts live in `experiments/EXP-NNN/`.
> Pre-rebuild experiments (EXP-001..033) are archived in `agent/archive/EXPERIMENTS-full.md`.

| id | what it tested | result |
|----|----------------|--------|
| V-rebuild-dyn | spec-rebuild dynamics_model.py + gridworld/recall.py vs specs (CLAIM 1+2) | FAITHFUL. P1: n_memory=0 forward bit-identical to src_old (maxdiff 0.0, labeled+unlabeled). P2: rollout_step(commit=False) leaves cache+next_pos unmutated (5 branches); carried state intact. P3: oracle readout exact at every k (0/640 fails). FF9 detached scaler balances ff9->diffusion magnitude, grad flows to memory_tokens. Artifacts: experiments/verify-rebuild-dynamics-recall/ |
