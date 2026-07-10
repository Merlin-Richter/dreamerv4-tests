# autoresearch — ColorField-sym

Improve a small world model's
**memory** on the ColorField-sym environment (5×5 one-hot viewport scrolling over a 15×15
iid-color board; info seen long ago must be repainted when revisited). Training uses a
mem→mem rollout objective; scoring uses a frozen "comeback" eval that rewards correctly
repainting cells by how long ago they were seen.

## Setup

To set up a new experiment run (if truely new; first check rather you can continue an existing loop), work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jul9`). The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files**: read these for full context:
   - `autoresearch/program.md` — this file.
   - `autoresearch/editable/train_sym.py` — the trainer you modify. Hyperparameters are
     its argparse DEFAULTS — change them in the file, not via CLI.
   - `autoresearch/editable/model.py` — the dynamics transformer (editable).
   - `autoresearch/editable/rollout.py` — the mem→mem training objective (editable).
   - `autoresearch/editable/adapter_sym.py` — eval bridge + one-hot codec (editable, but
     see the warning below).
   - `autoresearch/frozen_sym/env.py` + `eval_comeback.py` — environment + scorer
     (READ-ONLY; read them to understand what is scored; editing will result in fail automatically).
4. **Verify the compute box is reachable** (a rented Vast.ai box, RTX 5090 — no queue):
   `wsl.exe -e bash -lc "cd /mnt/c/Users/richt/OneDrive/Desktop/Code/transformer && bash scripts/vast_status.sh"`
   If it prints `ERROR: AUTH_DEAD`, SELF-HEAL it — vast has no 2FA, you are allowed and
   expected to reopen the socket yourself:
   `wsl.exe -e bash -lc "cd /mnt/c/Users/richt/OneDrive/Desktop/Code/transformer && bash scripts/open_master.sh --cluster vast"`
   Only if reopening fails repeatedly is the box itself down/stopped — THEN stop and tell
   the human (restarting the instance needs the Vast console, which only they have).
5. **Initialize the ledger**: create/read `autoresearch/results.tsv` with just the header row
   (see Logging results). Leave it untracked by git — do NOT commit it.
6. **Confirm and go.** Your very first run establishes the BASELINE: run the experiment
   with the code exactly as-is. Your second run repeats the baseline unchanged — the
   score difference between the two IS your run-to-run noise estimate; treat score deltas
   smaller than that as ties.

## Experimentation

Each experiment trains on the vast box's RTX 5090 for a **fixed budget of 600 seconds**
(wall clock, enforced remotely by the trainer's per-step BUDGET_STOP — you never manage
time), then runs a fixed probe + eval suite. You launch it as ONE command (from the repo
root, after committing your change):

```
wsl.exe -e bash -lc "cd /mnt/c/Users/richt/OneDrive/Desktop/Code/transformer && bash autoresearch/loop/run_experiment.sh --cluster vast" > run.log 2>&1
```

Redirect everything as shown — do NOT tee or stream the output into your context.
The command syncs your pushed commit, launches the box job, waits, and appends the job's
log (which ends in the summary + score lines) to stdout. A full cycle is ~14–16 min
(no queue: ~1 min sync/launch + 1 min pace probe + 10 min train + ~3 min probes/eval).

**If your shell kills long commands** (symptom: the run command dies with rc=124 mid-wait
— e.g. a 10-minute tool cap): the REMOTE job is already running detached and is
unaffected; only the local waiter died. Do NOT relaunch. Re-attach instead: poll
`wsl.exe -e bash -lc "cd ... && bash scripts/vast_status.sh"` about once a minute until
your run (`loop-<sha>`) shows DONE, then fetch its log with
`wsl.exe -e bash -lc "cd ... && bash scripts/vast_status.sh loop-<sha> --tail 500" >> run.log`
and continue as normal. Give the initial command your shell's maximum timeout so the
launch phase (~2 min) always completes inside it.

**What you CAN do:**
- Modify `autoresearch/editable/{train_sym,model,rollout}.py` — everything is fair game:
  the objective, the architecture, hyperparameters (as file defaults), the memory
  mechanism, the LR schedule, the data sampling. Out-there ideas are explicitly welcome:
  memory tokens retained past window eviction, learned compression of evicted frames,
  sparse attention into the deep past, new state modalities, different backprop through
  the relay — anything that fits the two resource prices below. The only thing is that it has to stay a transformer.
- Modify `adapter_sym.py` ONLY if your model change requires a new inference path. The
  codec (`encode_latents`/`decode_latents`) and its dims must stay consistent with
  training — train_sym imports them from there, so drift breaks you loudly.

**The two resource prices (instead of architectural bans):**
1. **Training compute** is priced by the fixed 600s budget — a slower design trains fewer
   steps. Nothing else limits training.
2. **Carried rollout state is capped in BYTES**: everything your model carries across
   inference steps (KV cache + any persistent per-episode tensors) must fit
   `state_budget: 518400` bytes (1.5× the seed's dense-W=16 cache), measured by the job's
   state probe (`state_check: FAIL` ⇒ the run is void; so is per-step state GROWTH —
   unbounded accumulation is a buffer, not memory). This is why a giant dense context is
   a non-starter, but ANY clever bounded-state mechanism is legal — including earning
   headroom via fp16 K/V or grouped-query attention (`gqa_groups` exists in the config).

**What you CANNOT do:**
- Modify `autoresearch/frozen_sym/` (env, datagen, policies, scorer), `autoresearch/loop/`,
  `autoresearch/driver/`, `scripts/`, or the datasets. These are integrity-checked;
  a tampered run scores 0.
- Hide carried state from the probe (stashing tensors in globals/closures). The probe
  sweeps the adapter object graph; kept diffs are human-reviewed and buffer-shaped gains
  are visually obvious in `real_bins` — it will be caught, and the run voided.
- Install packages or add dependencies.
- Touch the box except through `autoresearch/loop/run_experiment.sh`, plus exactly two
  sanctioned extras: the read-only `scripts/vast_status.sh` (health check / re-attach) and
  `scripts/open_master.sh --cluster vast` (AUTH_DEAD self-heal). Never `vast_run.sh` or
  `vast_cancel.sh` directly; never cancel jobs — they self-terminate within the hour.

**The goal: the highest `score`** (0..1; v2.2-sym, continuous — no hard gates):

    score = fid · (0.2·ent + 0.8·composite)

- `fid` = (move-fidelity + hold-fidelity)/2 — exact shift/hold correctness of your
  imagined rollout (the seed's weak spot: move ≪ hold). Improving it pays score directly.
- `ent` — degenerate-collapse guard (≈1 for any honest model; 0 for a color-collapsed
  world, so collapse can't farm the fidelity credit).
- `composite` — the memory term (chance-corrected comeback accuracy, consistency can only
  amplify real retention).

A perfectly coherent scroller with ZERO retention scores exactly **0.2** — that's the
"trivial subproblem solved" waypoint; **all headroom above 0.2 is memory-only**. Ties
broken by simpler code. Secondary signals to guide you — `inwindow_shift`/`inwindow_past`
(teacher-forced probes), `flow_final`, `real_bins` (the per-age curve: memory-shaped gains
lift FAR bins, not just near ones). These inform your next idea but only `score` decides
keep/discard.

**VRAM** is a soft constraint (RTX 5090, 32 GB — `peak_vram_mb` in the summary; the seed
peaks ~13.5 GB). Some increase is fine for real gains; do not blow it up — an OOM crash
burns a full cycle.

**Simplicity criterion**: all else equal, simpler is better. A tiny gain that adds ugly
complexity is not worth it; equal results from less code is a keep. Weigh complexity cost
against improvement magnitude.

## Output format

The job log (end of your run.log) contains grep-able `key: value` lines:

```
state_bytes:      345600
state_growth_bps: 0.0
state_budget:     518400
state_check:      PASS
inwindow_shift:   0.7080  (n=1849)
inwindow_past:    0.2136  (n=103)
score:            0.103836
fid:              0.5192  (move 0.3101 n=1680 / hold 0.7282 n=7536)
ent:              1.000  (kl 0.0018 n=3126)
composite:        0.000000
real_cc:          0.000000
consistency_cc:   0.003438
flags:            []
real_bins:        {'[1,17)': '0.000', '[17,33)': '0.000', ...}
eval_seconds:     95.7
---
steps:            5381
train_seconds:    600.1
sec_per_step:     0.112
flow_final:       0.0085
peak_vram_mb:     13560
gpu_util_pct:     92
window_frames:    16
```
(these are the REAL v2.2 seed-baseline numbers — your first run should land very close;
the measured duplicate-baseline pair was 0.103836 / 0.104758, i.e. noise ≈ 0.001)

Extract the essentials with:

```
grep -E "^(score|fid|inwindow_shift|inwindow_past|state_check|steps|peak_vram_mb):" run.log
```

If the grep is empty, the run crashed — `tail -n 60 run.log` shows the failure (Python
traceback, sha-gate failure, or a cluster error line like `ERROR: AUTH_DEAD`).

## Logging results

When an experiment finishes, append it to `autoresearch/results.tsv` (tab-separated).
Header + 7 columns:

```
commit	score	inwindow_shift	steps	vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. score (e.g. 0.135200) — 0.000000 for crashes
3. inwindow_shift (e.g. 0.6720) — 0.0 for crashes
4. steps completed in the budget
5. peak VRAM in GB (peak_vram_mb / 1024, .1f)
6. status: `keep`, `discard`, or `crash`
7. short description of what the experiment tried

## The experiment loop

The run lives on your `autoresearch/<tag>` branch.

LOOP FOREVER:

1. Look at the git state: current branch + commit.
2. Modify the editable files with ONE experimental idea.
3. `git commit -am "<idea>"` **and `git push -u origin HEAD`** — push from YOUR shell
   (Windows Git Bash has the credentials; the WSL runner deliberately does not push and
   will fail with `ERROR: BAD_REF` if you forgot). The runner refuses a dirty tree.
4. Run the experiment command above (`> run.log 2>&1`).
5. Grep the results out of `run.log`.
6. Empty grep = crash. Read the tail, judge: dumb bug → fix (amend or new commit), re-run;
   fundamentally broken idea → log `crash`, revert, move on. `ERROR: AUTH_DEAD` → SELF-HEAL
   (`open_master.sh --cluster vast`, see Setup step 4) and retry; escalate to the human only
   if reopening itself fails repeatedly (box down/stopped).
7. Append the row to `autoresearch/results.tsv` (never commit this file).
8. If `score` improved beyond the noise band (from your duplicate-baseline runs):
   ADVANCE — keep the commit as the new base.
9. Otherwise: `git reset --hard HEAD~1` back to the previous best.

You are a completely autonomous researcher. Works → keep; doesn't → discard; advance the
branch and iterate. Rewind past your best very, very sparingly (if ever).

**Where to hunt (context from the manual probes that seeded this loop):** the current
recipe's teacher-forced shift-copy acc is ~0.67 (floor for a healthy dynamics model ≈ 1.0)
— fixing in-window prediction fully is likely prerequisite to any real memory gain, and
under v2.2 it pays score DIRECTLY (fid climbs toward the 0.2 floor; the seed starts ~0.13). Levers
already suspected but untested: `tau0_anchor` → 1.0; the ramp weight `w(τ)` giving anchored
τ=0 frames only `ramp_min` weight; the 50/50 clean/noise mode ratio; FF9 (`--ff9 k` — OFF
in the seed, but it was load-bearing in one GridWorld regime); memory token count; LR.
History that transfers from GridWorld: bootstrap distillation bought nothing at K=1-4;
relay grad explodes at init (~3×/hop) — `relay_grad_clip` exists if training destabilizes.

**Timeout**: there is no queue on vast, so a healthy cycle is ~14–16 min. If a run exceeds
~25 minutes wall, check once with
`wsl.exe -e bash -lc "... bash scripts/vast_status.sh"` — RUNNING with a progressing log
line means wait; a vanished run or a stalled log means treat as crash.

**NEVER STOP**: once the loop has begun, do NOT pause to ask the human whether to
continue. No "should I keep going?". The human might be asleep and expects you to work
indefinitely until manually stopped. If you run out of ideas, think harder — re-read the
frozen scorer for what actually moves the metric, re-read your results.tsv for near-misses
to combine, try more radical architecture changes. The loop runs until interrupted.
