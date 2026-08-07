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
- an offline FP32 cache of every exact `(episode,window_start,W=32)` community-tokenizer result.
  The cache preserves the vanilla encoder's independent-window semantics bit-for-bit while ensuring
  that tokenizer work is performed once, before dynamics training. Whole-episode or whole-128-frame
  encoding is not substituted.

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
content-disjoint episodes, WMDataset-to-frame action alignment, and the reference online encoder.
`build_latent_cache_h100.sh` then creates the 2,810,100 exact W-window entries with shape
`(2810100,32,8,64)` in FP32 (171.515 GiB) and hashes the completed array. The validator checks the
manifest identity, proves sampled cached rows are bit-identical to online encoding (`max_abs=0`), and
verifies all seven window lookups used by a 128-frame training clip. Production additionally re-hashes
the complete array before starting its training clock.

The production clock now matches the accepted vanilla baseline: cumulative wall time from entry into
the training loop through data loading, cached-latent reads, dynamics forward/backward/optimizer,
logging, and periodic checkpoints. Setup, cache construction/validation, and the final checkpoint write
remain outside the 48-hour clock, as in vanilla. Checkpoints preserve optimizer/scaler state, all RNG
state, cumulative training-loop seconds, the exact cache-manifest hash, and the next counter-keyed batch
step, so resume neither repeats nor skips an optimizer step.

The prior online-encoding calibration/job are retained as superseded provenance only. Job 429438 was
cancelled after 4:03:56 because its narrow dynamics-only timer would have granted the memory arm roughly
twice the physical H100 time of vanilla. After the exact cache is built, a short cached-loader H100
calibration must re-confirm batch size, resume, utilization, and HBM. Production then starts fresh from
seed 0 in one 54-hour allocation and stops after exactly 172,800 cumulative training-loop seconds.
