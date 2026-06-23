# ORIENT.md

Rewritten: 2026-06-23 (EXP-025 GridWorld tokenizer v3 IN FLIGHT on ferranti, job 408737).

## What we're doing right now and why
Building the **GridWorld discrete-memory pipeline on the cluster**. The cluster interface (T-003) is
built + validated; env reworked to a 6×6 anti-overfit geometry (D-038); recall eval built (D-040).
Getting a USABLE frozen tokenizer (one whose latents encode the moving square, not background-only)
is the gate to the GridWorld dynamics / H2-H3 memory work. EXP-024 failed; EXP-025 is the fixed retry.

## AWAITING MERLIN — EXP-025 tokenizer WORKS (ESC-017, present-then-stop)
- **EXP-025 (job 408760, W&B 70k76148) SUCCEEDED.** D-043 stability fix worked: val/mse monotone
  0.0056→**4.7e-6** (NO explosion; 130× below EXP-024's pre-explosion peak), latent_cos 0.08–0.11 (no
  collapse), grad_norm bounded (spike-guard skipped ~6 steps), and the **recon strips visibly show the
  colored square at the right cell + colour ≠ bg** (experiments/gridworld-tok-v3/recon.png). Confirms the
  corrected diagnosis: EXP-024 failed from a LOSS EXPLOSION, not sparse-target collapse.
- Checkpoints staged: experiments/gridworld-tok-v3/{tokenizer.pt(best=ep29), tokenizer_last.pt}. NOT yet
  copied to checkpoints/gridworld/ — awaiting Merlin's freeze blessing (ESC-017 Q2).
- Caveat logged: fg_frac~0.32 (mask catches the curtain) → judge ball by recon visual, not fg_mse alone.

## HOW TO OPERATE THE CLUSTER (read before touching it)
- **All `scripts/` verbs run in WSL**, NOT Git Bash (shared ssh-socket namespace — D-036). Invoke:
  `wsl.exe -e bash -lc "cd /mnt/c/Users/richt/OneDrive/Desktop/Code/transformer && bash scripts/<verb> ..."`.
- Two clusters, no default: `--cluster ferranti` (H100, in use) / `galvani` (A100, unconfigured).
- Master socket is opened by Merlin in WSL; `ERROR: AUTH_DEAD` → ask him to re-run `open_master.sh`.
- `submit_job` default `--cpus 8` (needed — else SLURM gives cpu=2 and starves the GPU). bf16+TF32 on.
- Run-tuning recipe (batch/epochs/venv/W&B) in `HOWTO/cluster.md` "Run tuning notes". `scripts/cluster.env`
  was reconstructed this session (D-041 turn) after I deleted it; verified via cluster_health — if a
  cluster field misbehaves, suspect a reconstructed value.

## NEXT ACTIONS (in order)
1. **When 408737 finishes:** confirm COMPLETED → `pull_results --cluster ferranti gridworld-tok-v3`
   (tokenizer.pt + recon.png) + fetch W&B curve (grad_norm bounded? no explosion? val/fg_mse dropped?
   val/mse < ep9's 6e-4?). **VERIFY the ball is visibly reconstructed (right cell + a colour ≠ bg) — do
   NOT trust low MSE.** If good → report to Merlin (he asked to be told when it works) + freeze to
   checkpoints/gridworld/tokenizer.pt. If it exploded again or ball still dropped → see D-043 tripwires.
2. **Then: vanilla GridWorld dynamics on the cluster** (record a decision first). Uses the frozen
   tokenizer. `train_dynamics.py` is perf-fixed (D-041) BUT **re-profile its batch size** — it trains
   in latent space (frozen-tokenizer encode per step), a different compute profile than the LPIPS
   tokenizer's bs64. Watch util on the first run.
3. **Then: wire the eval model-adapter.** `src/evals/gridworld/recall.py` is frame-source based
   (oracle/copy-last built); add a dynamics-rollout frame source → run recall curves (graded position
   + ball/bg color) vs occlusion k → compare to copy-last/oracle. Then decide the periodic-W&B-during-
   training eval with Merlin.

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
