# Long-context prefill for rollout sheets (model processes 64 ctx frames, sheet shows few)

Requested by Merlin 2026-07-04: sheets should give the model a LONG context (64 frames) already
processed through the sliding window before generation starts — without putting all 64 on the sheet —
because with 8 frames of context the memmaze env is impossible by construction (maze mostly
unobserved) and long prefill is the intended usage (memory absorbs pre-window context).

## Design
- `DynamicsModel.rollout_init` learns long-context prefill: `T_ctx > max_temporal_length` → first
  window committed in one forward (existing path, byte-identical for T_ctx <= W), remaining TRUE
  frames teacher-forced one committed step at a time (`_commit_context_frame`: the same near-clean
  commit pass as `rollout_step(commit=True)` — written-memory relay + eviction — with the provided
  latent instead of a generated one). `generate()` gets it for free. Spec updated; gate test added.
- `evals/memmaze/sheets.py` gains `n_pre` (default 64 = 2x the dynamics W=32; 64 also = tokenizer
  window, the one-shot encode limit): model prefills `n_pre` frames, sheet displays only the last
  `n_ctx` context columns + the rollout. Display decode = one-shot tokenizer window on the tail
  (window-invariance probe: safe). Labels say "{n_pre} ctx ({n_ctx} shown)".
- NB for vanilla (no memory) the >W prefill is information-equivalent to last-W context; for the
  mem2mem arms it is exactly the memory-relevant setup (older half only reachable via memory).

## Done means
Gate tests green (incl. new long-prefill test), spec<->src in sync (dynamics + memmaze sheets),
local smoke on val12 real data, vanilla sheets re-rendered with n_pre=64 (cluster), pulled +
eyeballed, NOTES/ORIENT updated. make_sheets.sh passes --n-pre 64 for future arms.
