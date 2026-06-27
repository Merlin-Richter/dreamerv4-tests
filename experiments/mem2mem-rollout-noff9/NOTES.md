# mem2mem rollout-only, NO FF9 — is the explicit sufficiency loss even needed?

Parallel ablation against `experiments/mem2mem-rollout-boot/` (job 411221). Drops the FF9 sufficiency
term entirely: memory is trained ONLY by the rollout flow loss.

## Hypothesis
Memory-carrying pressure in the rollout already comes from TWO places: (a) the explicit FF9 term, and
(b) the 50% **full-noise** mode, where every latent in the window is pure noise so the new-half flow loss
can only be satisfied by reconstructing the scene from the carried memory. This run removes (a) and keeps
(b): if the noise-mode flow loss alone trains memory to carry hidden state, FF9 is redundant for this task.
Verified offline: with `use_ff9=False`, force_mode="noise", the relay gradient still reaches an evicted
frame's memory construction (|grad|≈9e-4, exactly 0 when the relay is detached) — so memory IS still trained.

## Setup (provenance)
- Branch `exp/mem2mem-rollout-only`, SHA: see EXPERIMENTS / job record. Cluster ferranti (H100), parallel
  with job 411221.
- IDENTICAL to 411221 except `--no-ff9` (rollout-only, bootstrap ON, curriculum ON). Config still carries
  ff9_k=3 / n_memory=4 (model architecture & checkpoint shape unchanged) — only the loss term is dropped.
- Command:
  ```
  python -u experiments/mem2mem/train_mem2mem.py \
    --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
    --checkpoint checkpoints/gridworld/dynamics_mem2mem_rollout_noff9.pt \
    --epochs 36 --batch-size 64 --clip-len 64 --ff9 3 --n-memory 4 --mem2mem-frac 1.0 --no-ff9 \
    --wandb --wandb-project dreamerv4-gridworld --wandb-name gw-dyn-mem2mem-rollout-noff9
  ```
- Dropping FF9 removes one forward/slide, so this run is FASTER than 411221 (< ~3h).

## Eval plan
Recall @ window=8, max_k=64 (+ few-step K=2/1), 4-way vs: rollout-boot (411221, flow+ff9),
rollout-only no-boot (`dynamics_mem2mem_rollout.pt`), and the baselines. Question: does retention survive
without FF9?

## Result (2026-06-27) — FF9 is NECESSARY: without it, retention = chance.
Job 411270 COMPLETED (36ep). `ff9 0.0000` confirmed throughout (FF9 cleanly dropped); rollout-only.
Training was UNSTABLE: val(normal) climbed 0.006→0.022 right when the curriculum fully unlocked (ep11,
full bootstrap, no FF9), plateaued ~0.015–0.02, then the LR cosine decay pulled it back to 0.009 by ep36.
Recall @ window=8 max_k=64 K=4: position_acc **~0.03–0.08 at EVERY k incl. k=2** = chance (1/36≈0.028),
i.e. the model behaves like vanilla — memory carries nothing. (Oracle self-test 1.0, instrument fine.)

Conclusion: the rollout flow loss alone — even with the 50% full-noise mode that *should* force memory
use, and even though the relay gradient verifiably flows — does NOT train the memory tokens to carry
hidden position. The explicit FF9 sufficiency term is doing the load-bearing work. Caveat: the run was
also unstable, so "FF9 needed" is partly entangled with "no-FF9 needs different HPs"; but chance-level
retention even at small k is strong evidence FF9 is required on this task. ckpt
`dynamics_mem2mem_rollout_noff9.pt`; curve recall_dynamics_mem2mem_rollout_noff9.json.

## Status
- [2026-06-27] DONE. branch SHA `0d1cdca`, ferranti job 411270 (36ep), --no-ff9. Result above.
