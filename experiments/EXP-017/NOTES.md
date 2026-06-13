# EXP-017 — FF9 v2 memory-token baseline (overnight)

Decision: D-024 | launched per Merlin's overnight-run request 2026-06-14 (late-night).
Provenance: master @ d1a38d1; tokenizer trained_autoencoder.pt (frozen); data occluded.npy +
occluded_actions.npy; 4070 (venv). Config: `config.yaml`. Launch: `run.sh`.

## Purpose
The architectural baseline for the memory-token line: distinct MEMORY tokens (registers → pure scratch)
trained with the FF9 v2 memory-only-sufficiency loss (ops 1&2: write-mem←latents, read-mem→latents),
at the EXACT EXP-010(FF7)/EXP-012(vanilla) budget. NOT expected to beat FF7 on beyond-window depth
(V-T013: within-window sufficiency only; the cross-window relay = op-3 = T-014 is built on top of THIS
next). This run gives us the trained memory-token model + a clean FF9-v2 reference.

## Why this and not the relay tonight
Merlin asked for a relevant overnight run. The full detached relay (Mode B, op-3) is exactly where the
verifier (V-T014) flagged bug-prone surface (cache-under-grad graph blowup, carry-norm, anchored per-step
loss). Rushing it at midnight risked a silently-broken run wasting the night. FF9 v2 is built + smoke-green
and is the needed foundation, so it's the high-value low-risk overnight choice. Relay built carefully next.

## Status / what's deferred
- TRAIN ONLY. Checkpoint `ff9v2_s0.pt` saved every epoch (resumable / usable at any stopping point).
- **Eval DEFERRED to tomorrow** — needs `generate_full_state_memory` (the memory-carry inference path,
  analog of generate_memory). Without it, `generate()` would NOT carry memory across rollout steps →
  a probe tonight would mis-measure the model as ~vanilla. So no probe tonight.
- W&B skipped (unattended robustness); per-epoch train/val loss + the diffusion/ff9 split are in
  `train.log`.

## TODO tomorrow (on completion)
1. Build `generate_full_state_memory` + dispatch (`use_full_state_memory`) — memory-carry rollout
   (reuse memory_rollout_init/step machinery on the memory slot) + smokes.
2. Memory-sufficiency probe (PRIMARY readout: L(memory) ≪ L(no-memory) within window).
3. Frozen-probe color at n_occ {12,16,24,32,48} vs vanilla_s0 (EXP-012) + ff7_k3 (EXP-010); expect ≈FF7.
4. Reconcile (Expected/Observed/Surprise/tripwires per D-024) → present-then-stop.
5. ESC-014 (relay gradient design: tbptt-k sweep / dynamic-state probe / train-to-depth) still OPEN —
   Merlin's call before the op-3/Mode-B build.

## Reconciliation
Expected (D-024): clean training; memory-sufficiency within window; no base-dynamics regression;
frozen-probe color ≈ FF7. Observed: <fill tomorrow from train.log + eval>.
