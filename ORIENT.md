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

## NEXT ACTIONS (in order) — focus: NAIL DOWN THE EVAL
0. **Tokenizer-roundtrip recall CEILING (cheap, do first).** Feed encode→decode of TRUE frames as a
   frame source into the existing recall scorer → upper bound on what ANY GridWorld dynamics model can
   recall through this frozen latent. Directly exercises the new tokenizer; de-risks the pipeline
   (D-044 tripwire). No model needed; just add a tokenizer-roundtrip frame source to recall.py.
1. **Eval-design FREEZE (Merlin's call — ESC-016 Q1).** Core BUILT (D-040): graded position_score +
   exact acc + ball/bg colour + per-k SE + reflection split, self-validated (oracle 1.0, copy-last
   decays, random≈chance). Open for him: (a) bless + FREEZE the design (§8 freeze before method exps);
   (b) periodicity handling (period-10 → copy-last spikes to 1.0 at k≡9 mod10 → judge per-k; periodic
   W&B eval uses off-grid k {3,6,12,16}); (c) where the during-training W&B eval lives (flatten_for_wandb
   exists, hook into train_dynamics TBD).
2. **Wire the model adapter** — add a dynamics-rollout frame source (rollout → decode → score) + the
   matched-horizon open-rollout control (docstring says it lives in the adapter). Then recall curves vs k
   vs copy-last/oracle. Needs a trained dynamics model.
3. **Vanilla GridWorld dynamics on the cluster** (record a decision first; grad-clip fix at
   train_dynamics.py:466–468 first; RE-PROFILE batch — latent-space compute ≠ tokenizer bs64).

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
