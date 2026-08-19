# Community Dreamer 4: d_min-only arm (does the bootstrap fix transfer?)

## Goal

Test whether the objective fix that transformed OUR memmaze vanilla transfers to an **independent
implementation**. Run the community Dreamer 4 dynamics with `d_idx` pinned to `d_min` in training,
against the already-completed community vanilla baseline.

Merlin approved the approach 2026-08-11 ("Ok, then use theirs") and set the budget at 12 hours on a
rented 5090. **Superseded 2026-08-19: Merlin moved the arm to ferranti (H100) and set 24 active
hours.** Hours on ferranti cost fairshare rather than money, and 24 h halves the training-budget
gap to the 48 h control instead of quartering it.

## Why this and not a tokenizer swap

Merlin asked whether to give the community trainer OUR latent cache + tokenizer "for fairness".
Answer: no, and the reasoning is on the record because it will come up again.

- This is a **within-implementation A/B** (their vanilla vs their vanilla + d_min pin). Both arms
  share the same frozen tokenizer, so the tokenizer floor **cancels** — it cannot bias the result.
- The tokenizer cap only affects *cross-implementation absolute* comparisons, and that is already
  solved by the **floor-free `self_mse`** metric (pixel MSE vs `decode(z_true)`, floor exactly 0),
  already implemented for both sides in `experiments/memory-inert-probe/latent_error_{ours,community}.py`.
- Swapping the tokenizer would invalidate the **completed 48 h vanilla baseline** (ferranti 423141,
  sha `7b077938…`), which was trained with their tokenizer — turning a 1-run experiment into 2 runs
  and discarding a finished one.
- Four concrete integration hazards if done anyway: token factorization (theirs 16x32 packed 2:1 to
  8x64, ours 32x16), noise-schedule calibration against a different latent scale, encoder window
  semantics (theirs 8 frames, our cache built at 64), and data conversion (`train-part0-v2` vs our
  `train-part0`, incl. channel order + action alignment).
- The existing in-progress task `memmaze-community-dreamer4-mem2mem.md` already forbids exactly this
  ("No frame resizing, color conversion, altered action shift, alternate tokenizer, or
  in-repository latent cache is allowed") for the same comparability reason.

## The arm

**The change is a single CLI flag; upstream needs no patch.** In upstream `dynamics_pretrain_loss`
(`train_dynamics.py`) each batch already splits into `B_emp = B - B_self` rows pinned to
`step_idx = emax` (= d_min, pure x-prediction flow loss) and `B_self = round(self_fraction * B)`
rows that sample coarser `d` and carry the self-bootstrap term. Setting **`--self_fraction 0.0`**
gives `B_self = 0`, so `do_boot` is never true and `loss` reduces to `loss_emp` exactly. No new
parameters; the checkpoint still loads `strict=True` in the stock `Dynamics` class.

Everything else identical to the locked baseline: `train-part0-v2` data, approved community
tokenizer (`347052fa…`), seq len 32, `d_model=512`, depth 8, 4 heads, `time_every=1`, packing 2,
4 registers, 1 isolated agent token, `k_max=8`, action conditioning on, LR 1e-4, wd 0.01, clip 1.0,
seed 0, batch size 128. Do NOT warm-start from the vanilla checkpoint.

One consequence is not a free variable: with `self_fraction=0` the whole batch of 128 carries the
flow loss, versus 96 for the control. That is inherent to dropping the bootstrap — the compute
stops being spent on it — not a separate knob, and it is exactly what a compute-matched
comparison is supposed to capture.

## The sampling schedule is not a free choice: score at K=1, K=8 and K=4

Upstream keys the shortcut step size through `nn.Embedding(log2(k_max)+1, d_model)` (`model.py`
`step_embed`). A d_min-only model only ever trains row `emax=3` (K=8). Rows 0..2 stay at their
random initialization forever, and **row 2 is K=4 — the default eval schedule** (`--eval_d 0.25`,
and the hardcoded value in the historic `evaluate_dynamics.py`). Scoring this arm at K=4 does not
merely take it out of distribution; it reads a parameter that never received a gradient.

`experiments/dreamer4-community-baseline/gate_dmin_only.py` proves this against the pinned
upstream sources, and passed locally on the 4070 on 2026-08-19:

- `self_fraction=0` → `bootstrap_mse` 0.0, `loss_self` 0.0, `loss == loss_emp` bit-exact;
- d_min-only `step_embed` row gradients: rows 0, 1, 2 exactly `0.000e+00`, row 3 `5.40e-04`;
- control (`self_fraction=0.25`) rows 0, 1, 2: `7.97e-07`, `2.35e-06`, `5.73e-07` — trained.

The gate must perturb `flow_x_head` before probing: upstream zero-inits it, so at initialization
every upstream gradient is exactly zero and a naive probe reads a false zero everywhere. This is
the same zero-init the baseline runbook already flags for why early predictions are 0.

The dead row is only the *second-order* problem, and reasoning from it alone would pick the wrong K.
The dominant one is **re-noising**: with x-prediction the first sampling step already emits ẑ₁, and
every further step re-noises the model's OWN prediction to an intermediate tau — an input never
produced during training. The ColorField testbed measured a d_min-only model at **K=1 0.978 /
K=2 0.680 / K=4 0.615 / K=8 0.532**, and at K=1 the `d_idx=0` row is *also* untrained, so the
input distribution clearly dominates the dead row.

Hence score at three schedules, not two:

- **K=1** (`--schedule shortcut --eval-d 1.0`) — the sampler degenerates to `z = x1_hat`: one
  forward, no re-noising. This is how a d_min-only x-prediction arm should be read. Verified
  supported by their scheduler (`_is_pow2(1)` holds) and by their sampler math (`tau=0`, `dt=1`,
  `denom=1` → `z = x1_hat` exactly). In-distribution for the control too, whose self rows sample
  `step_idx` uniformly over {0,1,2}.
- **K=8** (`--schedule finest`) — the step size the arm actually trained (d_min = 1/k_max), and
  in-distribution for the control. Fair, but pays the re-noising cost.
- **K=4** (`--schedule shortcut --eval-d 0.25`) — the historic protocol, kept for continuity, and
  the worst case for the arm: re-noised *and* reading the dead `step_embed` row 2.

If the arm wins at K=1 and loses at K=4, it has not "broken shortcut sampling" — it declined to
learn it, which is exactly what dropping the bootstrap term buys and costs. That is a far more
precise claim than "bootstrap breaks it", and it is the one the 3x3 matrix can support.

## Pre-registered prediction (this is what makes it a mechanism test, not a repeat)

Their bootstrap fraction is **25%**; ours was **83%** (`K_min=4` => d_min is 1 of 6 d-levels).
If the bootstrap term is the mechanism, d_min-only should help them **measurably but by clearly
less than our 15x** at h1. If it helps by roughly the same factor, we have misattributed the cause
and the writeup in `agent/EXPERIMENTS.md` (`vanilla-dmin-only`) needs revisiting.

Their headline h1 is 0.0061 against a tokenizer floor of 5.85e-4, i.e. ~10x headroom — a **positive
result is clearly readable**; only a null would be confounded by the ceiling. Note that up front.

## Budget

**24 active H100 hours** on ferranti (Merlin, 2026-08-19), against the control's 48. The arm is
therefore deliberately under-trained by 2x, a confound that points *against* d_min: a win is still
meaningful, a null is not conclusive. State this explicitly in the writeup.

Two mitigations are built into the protocol rather than left as caveats:

1. `phase5_evaluate_dmin.sh` also scores the **control's own periodic checkpoint nearest the arm's
   final step**, giving a step-matched read-off alongside the compute-matched one. Compute-matched
   and step-matched disagreeing is itself a finding.
2. The compute saving is small here and must not be oversold. Our own arm saved a lot because our
   bootstrap share was 83%; theirs is 25%, and the two bootstrap sub-forwards run on only 32 of
   128 rows with detached targets, so cost per step goes roughly 3.5 → 3.0 units. Expect ~15-20%
   more steps/s, not 3x. Record the achieved steps/s rather than assuming it.

## Scoring

Report through `phase5_evaluate_dmin.sh`: 3 checkpoints (arm final, control final, control
step-matched) x 3 schedules (K=1, K=8 finest, K=4), one instrument, one seed, 64 held-out
sequences. The historic run used 4 sequences, too few to separate arms; the driver also reruns
that exact 4-sequence K=4 configuration as a drift check against the recorded mse 0.007988 /
PSNR 20.98 dB.

Do NOT let a training-loss improvement stand as the result. The d_min-only loss is not comparable
across arms in the first place — the control's average is padded by a trivially satisfiable
self-distillation term — and 415103 had val 0.0043 with h1 0.0226. The rollout instrument decides.

## Status

- [2026-08-11] Written to backlog on Merlin's instruction; NOT started, prep-only.
- [2026-08-19] **Moved to in-progress; retargeted to ferranti at 24 h on Merlin's instruction.**
  Branch `codex/memmaze-community-d4-dmin`, cut from the immutable accepted-baseline ledger commit
  `1ae9cdc`. Added `gate_dmin_only.py` (passed locally; numbers above), `phase5_dynamics_dmin.sh`
  (the 24 h arm — `--self_fraction 0.0` and `--eval_schedule finest` are its only differences from
  `phase3_dynamics.sh`; it refuses to run with a nonzero self_fraction and fails the run if any
  logged `boot_mse` is nonzero), and `phase5_evaluate_dmin.sh` (the 3x2 scoring matrix plus
  arm-identity assertions and a control-hash immutability check). `evaluate_dynamics.py` gained
  `--schedule`/`--eval-d`/`--sheet-sequences`, with defaults that reproduce the historic protocol
  exactly. The vast rental plan in the original draft is dropped, not deferred.
  NEXT: sync to ferranti and submit the 24 h arm.
- [2026-08-19] **Submitted.** Implementation commit `5f06a89` pushed to
  `origin/codex/memmaze-community-d4-dmin`; ferranti synced to that exact SHA. Ferranti job
  **438958** (`memmaze-d4-dynamics-dmin-24h`, 1x H100, 16 CPUs, 30 h allocation for a 24 h active
  budget) submitted 18:25 Europe/Berlin, PENDING behind a 186-job queue. Artifacts land under
  `runs/memmaze-d4-dynamics-dmin-24h/`. Active training seconds: **0**.
  Do NOT re-sync ferranti to a newer commit while 438958 is queued or running — the remote
  checkout is shared, and moving it would swap the code under the job.
  NEXT: confirm the run starts, that `gate_dmin_only.py` passes on the cluster, that `boot_mse`
  logs as exactly 0.000000, and that H100 utilization holds >=90%; then record achieved steps/s
  against the control's 1.7255 before believing any speedup claim.
- [2026-08-19 22:07] **RUNNING and healthy at ~11% of budget.** Job 438958 started 19:24:29 CEST on
  `mlcbm004`, node-local staging 26 s, first GPU work 19:25:45. At step 17,300 / 2.69 h:
  - **Arm identity holds on the cluster:** `gate_dmin_only.py` PASSED all 13 checks on the H100 with
    values matching the 4070 run (row-3 grad 5.4022e-04 vs 5.4023e-04 local; rows 0,1,2 exactly 0).
    Every logged line shows `B_self=0` and `boot_mse=0.000000`. Conversion + split-disjointness
    validation passed.
  - **Health:** 971 telemetry samples, **98.76% mean GPU utilization** (control 98.69%), 36,499 MiB
    HBM (control 36,550), 608 W (control 613). Only 0.82% of samples below 90%; longest sub-20%
    interval 10 s. Passes the >=90% gate.
  - **Action conditioning is live:** matched-noise `action_shuffle` ratio 3.7-4.4 (untrained = 1.0).
  - **Throughput: 1.775-1.796 steps/s vs the control's 1.7255 — only ~3% faster, NOT the 15-20%
    predicted above.** The prediction was wrong and the writeup must use the measured number. Cause:
    the estimate counted only the transformer, but every step also pays the frozen tokenizer encode
    of 128x32 frames, which is identical in both arms and dilutes a saving that applies to just two
    sub-forwards on 32 of 128 rows. Projection: ~153,000 steps in 24 h vs the control's 298,164 in
    48 h, i.e. ~51% of its steps — so the step-matched control checkpoint will be near `step_0155000`.
  - **The loss trap reproduces exactly as pre-registered.** At matched steps the arm's *total* loss is
    consistently WORSE (step 17,200: 0.007594 vs 0.006212) while its **flow_mse is equal or better**
    (0.022931 vs 0.025492; ratio across steps 1k-17.2k = 0.90-1.02). The control's average is padded
    down by its near-free bootstrap term (boot_mse ~0.005 against flow_mse ~0.025). Do not compare
    the headline losses; `flow_mse` is the apples-to-apples diagnostic, since both arms compute it on
    `step_idx=emax` rows over the same tau grid.
  - Far too early to read: 17.3k of ~153k steps, and the control's FINAL flow_mse is 0.014994 against
    the arm's current 0.023. Nothing here is a result; the rollout instrument decides.
  NEXT: let it run to the 24 h stop, then sync ferranti to the eval commit and submit
  `phase5_evaluate_dmin.sh`. Confirm there that the control's periodic `step_*.pt` checkpoints still
  exist — the step-matched read-off degrades to a warning if they were cleaned.
