# Community Dreamer 4 mem2mem

This experiment extends the accepted `nicklashansen/dreamer4` Memory Maze baseline with the smallest
optional memory-token feature and a rollout-only trainer. It is deliberately outside spec-backed
`src/`. The pinned upstream commit remains `b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6`; setup first
applies the accepted baseline integration patch and then `upstream-memory.patch`.

The implementation is locked to:

- 8 optional memory tokens; `n_memory=0` has no new parameter or state key;
- W=32, stride=16, L=128;
- the first W-frame window is grounded and only constructs initial written memory;
- six later windows score absolute frames 32..127 once each;
- 50/50 per-sequence latent-present and memory-load-bearing modes;
- the community finest-step flow plus self-bootstrap objective (`k_max=8`, bootstrap step 5,000,
  self fraction 0.25), with no FF9 and no archive objective;
- a dedicated near-clean commit/write pass, never a denoising intermediate;
- a real 64-frame TBPTT boundary after four slides. The first segment is backwarded and released,
  boundary memory is detached, then the last two slides are backwarded. Both use 1/6 loss scaling and
  one optimizer step is taken per complete long-clip batch;
- exact online `(episode,start,W)` tokenizer windows. Whole-128-frame encoding is not used.

Local gates against a patched checkout:

```powershell
venv/Scripts/python.exe -u experiments/dreamer4-community-mem2mem/validate_model.py `
  --dreamer4 C:/path/to/patched-dreamer4 `
  --vanilla-checkpoint C:/path/to/approved/dynamics-final.pt
venv/Scripts/python.exe -u experiments/dreamer4-community-mem2mem/validate_resume.py `
  --dreamer4 C:/path/to/patched-dreamer4
```

`validate_data.py` is the server-side identity gate because the full converted train/eval trees live
on ferranti. It verifies the approved tokenizer hash, conversion manifests, RGB/action convention,
content-disjoint episodes, WMDataset-to-frame action alignment, and exact repeated W-window encoding.

The trainer's active clock covers only dynamics rollout forward/backward/optimizer work. Data loading,
the frozen tokenizer's seven exact window encodes, checkpoint I/O, and diagnostics are outside that
clock. Checkpoints preserve optimizer/scaler state, all RNG state, active seconds, and the next
counter-keyed batch step, so resume neither repeats nor skips an effective optimizer step.

H100 calibration job 429432 froze production at batch size 24, four loader workers, and 128 MiB shard
cache per worker. It achieved 98.75% mean active-interval utilization (p05 97%), used at most 54,843 MiB
of 81,559 MiB HBM, and passed an exact process restart at 180.337252 active seconds before stopping at
360.055772 seconds. The exact frozen values and measured wall-time projection are in
`production-config.json`. Production writes resumable hourly state to `memory-latest.pt`; only a
checkpoint whose audited active clock reaches 172,800 seconds is copied to `memory-final.pt`.
