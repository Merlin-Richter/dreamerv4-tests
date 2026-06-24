# ORIENT.md

Rewritten: 2026-06-23 (tokenizer FROZEN; Merlin redirected to nailing down the recall eval).

## What we're doing right now and why
Building the **GridWorld discrete-memory pipeline on the cluster**. Cluster interface (T-003) built +
validated; env at 6×6 anti-overfit geometry (D-038); recall eval core built (D-040). **Tokenizer now
FROZEN** (EXP-025 → checkpoints/gridworld/tokenizer.pt, D-044). Merlin's steer: "lets get the eval
down" → focus is the GridWorld recall eval: freeze its design + wire the model adapter so it can score
a dynamics rollout, the gate to H2/H3 memory work.

## Tokenizer FROZEN (ESC-017 resolved)
- EXP-025 (job 408760, W&B 70k76148 @ f1e3d6c) → checkpoints/gridworld/tokenizer.pt (best=ep29, D-044).
- Caveat logged: fg_frac~0.32 (mask catches the curtain) → ball judged by recon visual, not fg_mse alone.
- Tripwire (D-044): if the tokenizer-roundtrip recall CEILING (next exp) can't read out position/colour
  at reveal frames, the latent is the bottleneck — revisit tokenizer before any dynamics run.

## HOW TO OPERATE THE CLUSTER (read before touching it)
- **All `scripts/` verbs run in WSL**, NOT Git Bash (shared ssh-socket namespace — D-036). Invoke:
  `wsl.exe -e bash -lc "cd /mnt/c/Users/richt/OneDrive/Desktop/Code/transformer && bash scripts/<verb> ..."`.
- Two clusters, no default: `--cluster ferranti` (H100, in use) / `galvani` (A100, unconfigured).
- Master socket is opened by Merlin in WSL; `ERROR: AUTH_DEAD` → ask him to re-run `open_master.sh`.
- `submit_job` default `--cpus 8` (needed — else SLURM gives cpu=2 and starves the GPU). bf16+TF32 on.
- Run-tuning recipe (batch/epochs/venv/W&B) in `HOWTO/cluster.md` "Run tuning notes". `scripts/cluster.env`
  was reconstructed this session (D-041 turn) after I deleted it; verified via cluster_health — if a
  cluster field misbehaves, suspect a reconstructed value.

## IN FLIGHT: EXP-028 FF9 v2 memory method on GridWorld — ferranti job 409625 (D-047)
Vanilla baseline ACCEPTED (ESC-019). FF9 v2 (full-state memory token, n_memory=4, ff9_k=3) training,
budget-matched to EXP-027. Expected: colour retained PAST the 16-window (beats vanilla cliff); position
likely still cliffs (dynamic state → op-3). When done: build FF9-AWARE recall adapter (generate dispatches
to generate_full_state_memory — current dynamics_rollout_frames uses generate_cached/vanilla path) and
eval vs vanilla. TWO open eval-method items from Merlin: (1) convert the recall eval to ENV-DIRECT
generation (headline.png used the val SET; memory: evals-use-env-directly) — re-run vanilla under it for a
matched A/B; (2) sheets already env-direct. window-8 demo is in-distribution (causal masking), not OOD.

## (resolved) EXP-027 vanilla baseline DONE (ESC-019)
The GridWorld no-memory FLOOR is established. Recall (150 held-out val eps, job 409559): position_acc
model 0.573 in-window (vs copy-last 0.118) → 0.015 past window (k≥16); matched-horizon control flat
~0.70 at all k (→ the cliff is MEMORY LOSS, not weak dynamics); even static colour →chance past window.
Both D-046 tripwires clear; rollout protocol audited (V-EXP027). View: experiments/EXP-027/headline.png.
The clean discrete bench is now LIVE end-to-end: frozen tokenizer + frozen eval + audited rollout +
vanilla floor. NEXT (after Merlin): the MEMORY method on GridWorld (FF7/FF9/op-3 line on the clean
bench) — bring a method proposal + decision for sign-off before training. Housekeeping: archive
dynamics_vanilla.pt under experiments/EXP-027/ + fix run.sh to save checkpoints into the run dir.

## EXP-027 vanilla baseline TRAINED (job 409479 COMPLETED) — recall eval done (D-046)
Training clean: val diffusion 0.0146→0.00139 monotone over 80 ep, no explosion (grad-clip 1.0 works).
(409473 crashed ep3 on the persistent_workers stale-offset bug, same as the tokenizer; fixed 03a2f71.)
Checkpoint at CLUSTER checkpoints/gridworld/dynamics_vanilla.pt — NOT in the run dir so pull_results
can't reach it (run.sh saved to the wrong place; the tokenizer correctly saved into runs/). Plan: stage
it into the run dir inside the eval job, then pull. RECALL EVAL BUILT: dynamics_rollout_frames (adapter.py, faithful per-event open-loop rollout +
matched-horizon control) + experiments/EXP-027/{eval.py,eval_run.sh,recall_design.md}. Plumbing
validated locally on the smoke ckpt (oracle self-test=1.0). Protocol AUDITED by critical-claim-verifier → SUPPORTED (a/b/c, no bug; V-EXP027). **Eval IN FLIGHT:
ferranti job 409559** (@ f3ea659, gridworld-vanilla-s0-eval; stages dynamics_vanilla.pt back). When
done: pull results.json + headline.png + checkpoint, reconcile vs D-046 tripwires, present recall
curves (present-then-stop). Smoke ckpt at C:/Users/richt/AppData/Local/Temp/gw_dyn_smoke.pt. Self-provisioning run.sh: locates frozen tokenizer
(runs/gridworld-tok-v3/tokenizer.pt, fallback checkpoints/gridworld/; fail-fast if absent), regens
seed-42 gridworld data if absent, trains vanilla (bs64 lr3e-4 80ep seed0, grad-clip 1.0, n_actions=2)
→ checkpoints/gridworld/dynamics_vanilla.pt. WATCH: (1) tokenizer found on node? (2) no grad explosion
(clip should hold); (3) val diffusion decreasing. RISK if tokenizer absent on cluster → escalate
(can't push artifacts via wrappers; fallback = retrain tokenizer, ~1.4h, deterministic).
When done: pull dynamics_vanilla.pt, then wire dynamics-rollout frame source + run recall on HELD-OUT
(val) episodes (NOT eps 0-499 which overlap train). present-then-stop.

## NOW: training the vanilla GridWorld dynamics baseline (ESC-018 resolved; D-046)
- Eval CORE FROZEN (D-045, ESC-016 resolved): per-k judging + off-grid k {3,6,12,16} for W&B.
- EXP-026: tokenizer-roundtrip recall == oracle == 1.0 at every k → frozen latent NOT the bottleneck
  (D-044 tripwire cleared). View: experiments/EXP-026/headline.png.
- **Metric semantics (Merlin's correction):** BOTH colour and position are memory tests — colour =
  static retention (failable; copy-last passes it trivially but a model can hallucinate), position =
  memory + reasoning (retain + simulate the bounce under occlusion). Report both; position is the
  harder headline. (memory: project_gridworld_metric_semantics)

## NEXT ACTIONS (in order)
1. **Vanilla GridWorld dynamics on the cluster (IN PROGRESS, D-046).** Frozen tokenizer + frozen eval.
   Grad-clip fix at train_dynamics.py:466–468 first; RE-PROFILE batch (latent-space compute ≠ bs64).
2. **Wire the dynamics-rollout frame source** into adapter.py (rollout → decode → score) + the
   matched-horizon open-rollout control → real recall curves vs k vs the EXP-026 oracle/copy-last ref.
3. Decide where the during-training W&B recall eval hooks in (flatten_for_wandb ready; off-grid k).

## EVAL (D-040) — built + validated, NOT frozen
Headline = graded `position_score` (exact=1.0, adjacent=0.25, →0 by Chebyshev d=3) + `position_acc`
(exact, chance 1/36); ball + bg 4-way color via most-different-cell detection; per-k counts + SE.
Self-validates (oracle 1.0, random≈analytic chance 0.086, copy-last decays). **KEY FINDING:** the 6×6
bounce has period 2·(6−1)=**10**, so copy-last (no-memory) spikes to **1.0 at k≡9 (mod 10)**. ⇒ judge
memory by beating copy-last *per k*; for the periodic W&B eval pick occlusion lengths OFF that grid
(e.g. k∈{3,6,12,16}). A single averaged scalar would be inflated by the periodic spikes.

## Open escalations / worries
- **ESC-016 Q1 OPEN:** GridWorld eval design sign-off + freeze + periodicity handling + where to put
  the periodic-W&B eval. (Q2 compute-tier = cluster, answered.) Merlin gave the refined spec (D-040);
  awaiting his "freeze it / here's where to use it."
- **Tripwire (D-038):** if periodic/ballistic extrapolation ≈ oracle on position even off-period, the
  6×6 env is too easy → add cells/state.
- Dynamics batch-size re-profile (above).

## Parked (pre-pivot; resume only if Merlin redirects)
- C1/motion (EXP-021 ckpt ~ep10), exposure-bias/open-loop compounding. ESC-014 op-3 relay open.
- Occluded-line H3 (FF7 color SUPPORTED; FF9 v2 static-color SUPPORTED; position open). checkpoints/occluded/.
