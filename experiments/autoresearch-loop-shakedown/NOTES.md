# autoresearch-loop-shakedown — first live end-to-end loop iteration (vast 5090)

2026-07-10. Merlin's ask: "do one iteration of auto research with a real attempt and run,
just to check that it would work and train etc."

## Provenance
- Branch `autoresearch/jul10-shakedown`; runner+gitignore fixes @ `7e41889`, printer fix @
  `778fd63`, records @ `a956478`/`1bb8e58`. Backend: **vast** (RTX 5090, no scheduler),
  ferranti+galvani sockets were DOWN.
- Iter 1: run `loop-7e41889` (box checkout 7e41889) — rc=1.
- Iter 2: run `loop-a956478` — rc=0 clean. Training code byte-identical between the two SHAs
  (only the eval_reduced printer differs). Checkpoints stay on the box (loop design); metrics
  here + `autoresearch/results.tsv` (ledger, untracked by design; snapshot in this dir).

## What the shakedown proved
Full cycle green on the third backend: sync → vast_run (detached) → data sha-gate (procedural
datagen byte-identical on the 5090 box) → 60s pace probe (543–544 steps/min) → sched-sized 600s
budgeted train (5378–5380 steps, 0.112 s/step, util 92%, peak 13.2 GB) → state probe PASS
(345600 B carried state, zero growth) → in-window probe → reduced frozen eval (96 s) →
grep-able summary block → ledger row.

## Numbers (iter1 / iter2 — identical config, wall-clock jitter only = run-to-run noise)
- steps: 5380 / 5378 · flow_final: 0.0085 / 0.0085
- inwindow_shift: 0.7058 / 0.7020 · inwindow_past: 0.2039 / 0.2330
- fidelity: 0.6536 / 0.6622 — **FAIL both** (gate) → **score_gated 0.000000 both**
- composite_raw (ungated): 0.0092 / 0.0046 · real_cc: 0.0131 / 0.0066
- real_bins ≈ 0 at every age (chance-corrected): the seed carries ~nothing.

**Calibration finding:** the seed baseline GATES TO ZERO at 600s on this backend (fidelity
0.65–0.66 < threshold). Sub-gate metrics are tight across the pair (±0.01); the ungated
composite varies ~2× at near-zero magnitude — at this budget the only trustworthy improvement
signal is clearing the fidelity gate, not composite deltas near 0.

## Defects found + fixed (the point of a shakedown)
1. `eval_reduced.py` real_bins printer assumed dict; frozen scorer returns a list of bin dicts
   → rc=1 at the LAST print of iter1. Fixed (778fd63), verified against iter1's pulled eval JSON.
2. `results.tsv` + `run.log` are untracked-by-design but the runner refuses any dirty tree
   (`git status --porcelain` includes `??`) → iteration ≥2 would always DIRTY_TREE. Gitignored.
3. Agent-shell Bash timeout (120s default / 600s max) kills a held `run_experiment.sh`
   (~15-min cycle) → rc=124; the remote job survives (setsid+nohup) — re-attach by polling
   `vast_status.sh`. Proposal for Merlin: split runner into fast `launch` / idempotent `collect`.
4. Operator lessons (self-inflicted, recorded): piping a wrapper into `tail` masks its exit
   code (`sync|tail && vast_run` launched despite BAD_REF); never hand-type a full SHA.
   The launch lock survived both abnormal exits — no orphan.

## Open for Merlin (before a real loop night)
program.md still targets ferranti in 3 places (his file); ~~0-scoring baseline acceptable as
calibration?~~ RESOLVED by v2.2 below; launch/collect runner split; vast box stop/start (billing).

## v2.2 SCORING + RE-BASELINE (same day, Merlin's call: "hard gates are a bad idea")
Gates replaced by the continuous score = fid·(0.2·ent + 0.8·composite) — full design in the
colorfield-sym task addendum; implemented @ `9e3425f`, 9/9 tests green (v2.1 fences kept
verbatim on composite), independently verified by critical-claim-verifier (5/5 CONFIRMED, incl.
bit-identity of v2.1 component math vs HEAD; probes in verify_v22/).
- Iter 3 (`loop-9e3425f`, v2.2 baseline, rc=0): **score 0.103836** = fid 0.5192·0.2 exactly
  (move 0.3101 / hold 0.7282 — the split exposes how weak the actual scrolling is; the old
  pooled 0.66 was 4/5 trivial holds), ent 1.0 (kl 0.0018), composite 0.000000 (all bins 0).
  5381 steps/600s, state PASS. eval_reduced_iter3_9e3425f_v22.json.
- Iter 4 (`loop-9e3425f-b`, duplicate baseline, rc=0): **score 0.104758** (fid 0.5046 =
  move 0.2982/hold 0.7110, ent 1.0, composite 0.0095 — near-zero bins fluctuate ~0.01 but
  the headline barely moves). **Pair Δ = 0.0009 ⇒ the v2.2 run-to-run noise band ≈ 0.001**;
  a keep needs > best + ~0.002-0.003. Sub-metrics noise: fid ±0.015, composite ±0.01 (at
  near-zero magnitude). eval_reduced_iter4_9e3425fb_v22.json.
- Loop-viability read: baseline 0.104; fidelity work pays up to 0.200; everything above is
  memory-only. AUTH_DEAD self-heal exercised live mid-run (vast socket died; reopened by the
  agent per the no-2FA contract; detached job unaffected — third abnormal-exit survival).
