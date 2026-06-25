# EXP-030/031/032 eval runbook (mechanical — for processing results when checkpoints land)

Cluster jobs (ferranti, branch feat/ff9-rollout-training): 409752 (EXP-030 h24), 409754 (EXP-031
h44 deep), 409753 (EXP-032 vanilla w32). All run.sh stage the checkpoint into their run dir for pull.

## When a job finishes (check: `wsl.exe -e bash -lc "cd <repo> && bash scripts/job_status.sh --cluster ferranti"`)

### EXP-030 (run dir gw-ff9roll-m24-s0, ckpt dynamics_ff9roll_m24.pt)
```
# pull checkpoint
wsl.exe -e bash -lc "cd <repo> && bash scripts/pull_results.sh gridworld-ff9roll-m24-s0 --what all --cluster ferranti"
# eval under BOTH relay (its trained inference) and windowed (EXP-028 comparable), <=k32 to overlay baselines
./venv/Scripts/python.exe -u experiments/EXP-030/recall_relay.py --dynamics experiments/gridworld-ff9roll-m24-s0/dynamics_ff9roll_m24.pt --tag ff9roll_m24_relay    --inference relay    --n-per-k 64 --max-k 32
./venv/Scripts/python.exe -u experiments/EXP-030/recall_relay.py --dynamics experiments/gridworld-ff9roll-m24-s0/dynamics_ff9roll_m24.pt --tag ff9roll_m24_windowed --inference windowed --n-per-k 64 --max-k 32
```

### EXP-031 (run dir gw-ff9roll-d44-s0, ckpt dynamics_ff9roll_d44.pt) — eval to k44 (its train depth)
```
wsl.exe -e bash -lc "cd <repo> && bash scripts/pull_results.sh gridworld-ff9roll-d44-s0 --what all --cluster ferranti"
./venv/Scripts/python.exe -u experiments/EXP-030/recall_relay.py --dynamics experiments/gridworld-ff9roll-d44-s0/dynamics_ff9roll_d44.pt --tag ff9roll_d44_relay --inference relay --n-per-k 64
```

### EXP-032 (run dir gw-vanilla-w32-s0, ckpt dynamics_vanilla_w32.pt) — vanilla, windowed only (no memory)
```
wsl.exe -e bash -lc "cd <repo> && bash scripts/pull_results.sh gridworld-vanilla-w32-s0 --what all --cluster ferranti"
./venv/Scripts/python.exe -u experiments/EXP-030/recall_relay.py --dynamics experiments/gridworld-vanilla-w32-s0/dynamics_vanilla_w32.pt --tag vanilla_w32 --inference windowed --n-per-k 64 --max-k 32
```

## Baselines already available (no re-run)
- experiments/EXP-028/recall_env_vanilla.json (vanilla window-16, windowed) — the EXP-027 floor.
- experiments/EXP-028/recall_env_ff9.json (FF9 sufficiency-only, windowed) — the EXP-028 memory result.
- experiments/EXP-030/recall_env_ff9_norollout_relay.json (FF9 sufficiency-only under RELAY inference)
  — the untrained-B2-relay "before" picture (this session). Isolates training vs inference.

## Compare plot (the headline view)
```
./venv/Scripts/python.exe -u experiments/EXP-030/plot_rollout_compare.py \
  --series "vanilla w16:../EXP-028/recall_env_vanilla.json:tab:red" \
  --series "FF9 (no rollout):../EXP-028/recall_env_ff9.json:tab:green" \
  --series "FF9 relay (untrained B2):recall_env_ff9_norollout_relay.json:tab:olive" \
  --series "FF9+rollout h24 relay:recall_env_ff9roll_m24_relay.json:tab:blue" \
  --series "FF9+rollout h44 relay:recall_env_ff9roll_d44_relay.json:tab:purple" \
  --out compare_rollout.png
```

## The questions each comparison answers
- **FF9+rollout relay vs FF9 (no rollout) windowed** (EXP-028): does rollout-training extend recall past
  the EXP-028 decay (~k28) / the 16-window cliff? The headline.
- **FF9+rollout relay vs FF9-norollout relay**: isolates the TRAINING contribution (same inference) —
  does training the relay fix the untrained-B2 drift (V-T013-eval)?
- **h44 vs h24**: does deeper training reach further (P1's horizon==train-depth on the real task)?
- **vs vanilla w32**: does the relay beat brute-force context?
- **in-window (k<=8) regression check**: rollout models must NOT drop below vanilla/FF9 in-window
  (tripwire: the rollout branch fighting ops 1+2).

## Reconciliation: append to experiments/EXP-030/NOTES.md (Expected/Observed/Surprise/Hypothesis/Tripwires/Next),
update EXPERIMENTS rows (status->done + headline), then the consolidated MORNING BRIEF in ESCALATIONS.md.

> SUPERSEDED NOTE (2026-06-25): the "relay" inference framing here is deprecated — there is only the normal sliding-window inference. Corrected verdict: rollout-training does NOT beat FF9-no-rollout under it. See ESC-022.
