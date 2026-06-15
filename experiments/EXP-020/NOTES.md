# EXP-020 — C1 multistep treatment (PARTIAL) vs vanilla control (EXP-019)

Status: **PRELIMINARY readout on a PARTIAL checkpoint.** EXP-020 C1 training was cancelled by
Merlin at epoch ~17/40 (~42% budget); checkpoint `c1_h4_s0.pt` is the last periodic save (epoch 17,
val 0.0036). Control `EXP-019/vanilla_ctrl_s0.pt` is the FULL 40-ep run (val 0.0082). So this is a
budget-MISMATCHED, directional A/B — not the clean same-budget comparison D-027 designed.

Provenance: code @ a07fdee; tokenizer trained_autoencoder.pt; eval `experiments/EXP-020/ab_eval.py`
(reuses EXP-018 probe_multistep + frozen probe env/detector 5503e75). 48 probe episodes (seed 20000+),
curtain-UP (k=0, no occlusion), horizon 24. Output `ab.json` + `ab_eval_run.log`.
NB: ab_eval.py had a str/int key bug in cross_chance_h + the print loop (curves are int-keyed); fixed
(pure formatting fix; the model eval itself was correct on the first run).

## Reconciliation (§5)

Expected (D-027): C1's open-loop pos_err drops BELOW the budget-matched vanilla control, esp. mid-horizon
h4–h12, WITHOUT clean val/diffusion regressing past ~0.003 and WITHOUT teacher-forced (per-step map)
regressing. Collapse monitor: predicted displacement must track GT (~3.2px), not 0 (copy-last degenerate).

Observed (48 ep, H=24):
```
            h1    h2    h4    h8    h12   h16   h24   crossChance  predDisp(gt)
 control OL  23.1  21.5  22.9  20.5  19.7  20.1  20.1     h=1       7.10 (3.18)
 control TF  23.6  22.0  22.4  19.0  17.5  16.8  19.5
 c1      OL   2.6   3.6   4.6   8.6  14.5  17.0  18.8     h=20      4.60 (3.18)
 c1      TF   2.6   2.1   2.2   2.3   2.6   2.2   2.0
```
- C1 teacher-forced per-step map is FLAT + accurate (2.6→2.0px h1→h24), consistent with EXP-018's
  ff7/ff9 good-per-step regime. Control TF is at CHANCE (~18-23px) even at h1.
- C1 open-loop resists compounding far longer: crosses chance (18px) at h≈20 vs control at h=1.
  Mid-horizon (the D-027 target band) C1 4.6@h4 / 8.6@h8 / 14.5@h12 vs control ~20 throughout.
- Collapse monitor PASSES: C1 predDisp 4.60 (gt 3.18) — coherent motion, not the ≈0 copy-last
  degenerate V-T017-C1 C-B warned about. Control predDisp 7.10 = erratic over-movement (incoherent).

Surprise: HIGH — but mixed, see caveats.
- Favorable: C1 wins on BOTH per-step accuracy AND open-loop compounding, DESPITE 42% of control's budget.
- Anomaly: the control's TEACHER-FORCED 1-step is at chance (23.6px), ~5x WORSE than EXP-012's
  budget-matched vanilla_s0 (4.66px). The 250-ep subset appears to cripple vanilla motion learning.
  This is a REAL model property, not an eval bug (C1 through the identical eval path gives sane numbers,
  so the path works). It inflates the apparent C1 gap: part of the win is "C1 learns motion AT ALL on a
  tiny subset" not purely "C1 fixes compounding on top of an already-good per-step map" (the intended test).

Tripwires checked (D-027):
- (1) "open-loop unchanged vs control" → NOT triggered (C1 hugely better OL).
- (2) "val/diffusion regresses past ~0.003" → partial ckpt val 0.0036 (epoch 17, ramp not finished;
  combined loss, not clean-diffusion-only). Not a clean read; revisit at full budget.
- (3) "per-j multistep loss flattens to context-independent floor (prior-emission)" → collapse monitor
  predDisp 4.6 (not ≈0) argues AGAINST prior-emission; per-j logging during the full run to confirm.

Hypothesis impact (H-motion): C1 SUPPORTED directionally — DAgger multistep loss reduces autoregressive
compounding and yields a strong per-step map even under-trained. NOT a strong claim yet (budget mismatch
+ anomalously weak control). Need: full 40-ep C1 + understand control weakness.

Next: ESCALATE to Merlin (present-then-stop). Recommend (a) finish full 40-ep EXP-020 for clean
same-budget delta, (b) sanity-check the control's chance-level TF (subset-size effect? EXP-019 training
underperformance?) — quantify against EXP-012 vanilla_s0 (4.66px) on this probe before trusting magnitude.

## Control sanity-check (D-028 #2) — the weak control is REAL, not an eval bug
Ran ab_eval with --control=EXP-012/vanilla_s0.pt (FULL occluded data, 100ep) vs
--c1=EXP-019/vanilla_ctrl_s0.pt (250-ep subset, the A/B control). 24 ep, H12. (Labels are just
the two slots.) sanity_control.json / sanity_control.log:
```
            h1    h2    h4    h8    h12   crossChance  predDisp(gt)
 vanilla_s0  TF  5.2   4.5   4.1   4.9   3.9        (full data)
 EXP019ctrl  TF 21.1  16.8  19.5  14.1  20.5   h=1  (250-ep subset)
```
=> vanilla_s0 (full data) has the expected flat ~4.5px per-step map (harness validated AGAIN).
The EXP-019 250-ep control is at CHANCE teacher-forced => the 250-episode subset cripples vanilla
motion learning. So the EXP-020 A/B on the 250-ep subset is confounded: it conflates "C1 learns a
per-step map AT ALL on tiny data" with "C1 fixes open-loop COMPOUNDING on a good per-step map"
(the intended EXP-018 test). Clean compounding test needs a regime where the vanilla control HAS a
good TF map. Cheapest such control already exists: vanilla_s0 (full data, ~4.5px TF). => candidate
next decision after the full 250-ep run: a larger-data C1 run compared against vanilla_s0.
