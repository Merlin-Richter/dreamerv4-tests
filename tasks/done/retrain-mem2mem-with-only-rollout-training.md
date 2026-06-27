# Retrain the mem2mem model with only the rollout training instead of 50/50 split

We want to verify rather just the rollout training can also yield the same impressive results that the 50/50 split achived.
So run new experiment on the same code, but train only with the rollout trainig. Then do the rollout test with window=8 max_k = 64 and check rather it still has new perfect retention at high k.
## RESULT (2026-06-27) — DONE. YES: rollout-only matches (slightly beats) the 50/50 split.
Branch `exp/mem2mem-rollout-only` @ 2951ee6, ferranti job 411133 (2h51m, 50ep, `--mem2mem-frac 1.0`,
ckpt `dynamics_mem2mem_rollout.pt`). Log confirms `train normal: 0.00000` (genuinely rollout-only).
Recall @ window=8, max_k=64 — position_acc (mean / tail k>=14 / k=64):
  vanilla 0.040/0.033/0.000 · FF9 0.375/0.111/0.125 · mem2mem 50/50 0.988/0.984/0.984 ·
  rollout-only 0.992/0.988/1.000  → FLAT to k=64, perfect retention preserved.
The mem->mem sliding rollout alone is sufficient; the normal shortcut-forcing loss isn't needed for the
retention win. Minor cost: in-window val(normal) 0.005 vs 0.0027. Artifacts: experiments/mem2mem-rollout-only/
(NOTES.md, compare_w8_k64_4way.png). Also shipped scripts/pull_file.sh (pull one file off cluster).
