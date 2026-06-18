# BOARD.md — task board

Updated: 2026-06-18 (cleanup pass: completed/historical entries moved to `BOARD-archive.md`).

> Live task state only: in-progress / next / awaiting-Merlin / parked / blocked.
> Completed, superseded, and dropped work lives in `BOARD-archive.md` (grep there for history).

## In progress
- **GridWorld tokenizer on ferranti — RUNNING (job 405629, EXP-024).** 30ep bs64 LPIPS(vgg), ~95% util
  (D-041 perf fix), ETA ~15:45 → `runs/gridworld-tok-v2/{tokenizer.pt,recon.png}`. Provenance:
  feat/motion-prediction @ d5cef58. NEXT on completion: `pull_results --what checkpoints` → review recon
  → present-then-stop (this is the frozen GridWorld tokenizer; check recon quality + no latent collapse).

## Next (after the tokenizer lands)
- **Vanilla GridWorld dynamics on the cluster** (record a decision first). Frozen tokenizer; train_dynamics
  perf-fixed (D-041) but RE-PROFILE batch (latent-space compute profile ≠ tokenizer bs64).
- **Wire the eval model-adapter** (`src/evals/gridworld/recall.py` is frame-source based; add a
  dynamics-rollout source) → recall curves (graded position + ball/bg colour) vs k → vs copy-last/oracle.
  Then periodic-W&B eval with Merlin.

## Awaiting Merlin
- **ESC-016 Q1 — GridWorld eval sign-off / freeze.** Refined eval BUILT (D-040): graded position_score
  (exact=1/adj=0.25/→0@d3) + exact acc (chance 1/36) + ball/bg 4-way colour + per-k counts/SE; validated
  (oracle 1.0, random≈chance, copy-last decays). **Periodicity finding:** copy-last spikes to 1.0 at
  k≡9 (mod 10) (6×6 bounce period 10) → judge per-k; periodic W&B eval should use off-grid k {3,6,12,16}.
  Pending: freeze + where to wire the during-training eval. (Q2 compute=cluster, answered.)

## Parked (pre-GridWorld-pivot; resume only if Merlin redirects)
> Detail in `BOARD-archive.md` + ORIENT.md "Parked". Occluded-env models under `checkpoints/occluded/`.
- **Occluded-line H3 — DYNAMIC-state relay (op-3).** FF7 colour SUPPORTED (EXP-010); FF9 v2 static-colour
  SUPPORTED beyond window (EXP-017); dynamic POSITION still unsolved. Next method = sequential stop-grad
  register-relay (IDEAS.md). **Blocked on ESC-014** (relay gradient design; V-T014 refuted pure detach →
  cheap tbptt-k sweep recommended before the Mode B build).
- **C1 / motion (exposure-bias / open-loop compounding).** EXP-021 ckpt (~ep10) trained but un-evaluated;
  EXP-018/020/022 established compounding diagnosis + DAgger C1 loss + inference-trust (context_signal) lever.
- **Position-memory consistency metric** — BUILT (`src/probe/position_consistency.py`, EXP-013) but of
  uncertain strength; Merlin chose NOT to freeze it (ESC-009). Revisit only if a position method needs a yardstick.

## Blocked
*(none)*
