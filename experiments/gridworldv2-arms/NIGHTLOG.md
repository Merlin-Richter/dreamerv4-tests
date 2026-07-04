# Night log — 2026-07-05, autonomous research window (Merlin asleep, full authority granted)

Standing orders: own decisions, own experiments, subagents freely; track everything here + task
files. All conclusions provisional until written up with numbers.

## Agenda (set 00:20)

1. **Anchor**: when B/C (dense mem2mem on v2) land -> full 4-way eval, 256-rollout re-run of key
   comparisons (64 rollouts = +-0.06 noise). Decides whether D's ~0.5 plateau is the env ceiling
   or sparse's price.
2. **D in-window dip** (k=4-6: 0.56-0.75 vs A's 0.95, yet 0.94 at k=10 w8 pure-memory):
   delegated to an independent subagent with checkpoints + env; hypotheses to discriminate:
   (i) memory-vs-latents arbitration miscalibration (mixed-visibility windows undertrained),
   (ii) systematically bad first in-rollout write (pos 8), (iii) eval-driver or sparse-integration
   bug (action off-by-one, mask/phase at eval positions), (iv) noise (64 rollouts).
3. **Covert-channel probe** (mine): A's residual 0.2-0.3 at w16 high-k suggests vanilla relays
   belief through its OWN committed occluded latents (nothing forces generated "gray" latents to
   be information-free). Probe: sanitize committed occluded latents (replace with encoded TRUE
   curtain frame) -> residual should drop to ~copy_last if covert. Affects interpretation of ALL
   arms.
4. **Capacity axis** (Merlin's "sparser but bigger"): launch D2 = sparse n=8, n_memory=16 now
   (jobs are ~35 min; result by morning regardless of anchor outcome).
5. **Memmaze night landings**: 415205 (~02:40), 415104 (~05:00) -> pull, verify, sheets, W&B.
6. Stretch (only if 1-5 resolve cleanly): n=16 variant (needs W=32 model) or a mixed-visibility
   training mode for D, depending on what the dip diagnosis says.

## Decisions made

- D2 (n_memory=16) launched before the anchor lands: the axis is wanted by Merlin regardless,
  and jobs are cheap. Risk: if D has a systemic bug, D2 wastes one job — acceptable.

## Findings (running)

- 00:05 A+D early eval (64 rollouts): A = honest baseline on v2 (w8 0.97 in-window, eviction
  collapse; w16 residual 0.2-0.3 at high k = covert-channel suspect). D = ~0.42-0.59 FLAT to
  k=64 at both windows (memory carries across ~8 write relays, no decay) but in-window dip
  k4-6. Committed 437c820.

- 00:50 **Covert-channel probe: NEGATIVE (good).** A's high-k residual SURVIVES sanitization
  (teacher-forced true-gray commits): w16 0.14-0.30 sanitized vs 0.14-0.34 standard. Not a covert
  latent channel. D also unchanged under sanitization -> its carrier is the memory tokens, as
  designed. (probe_covert_channel.py, results_covert_channel.json)
- 00:55 **The residual is explained — and it changes the eval's meaning.** Exact 36-state Bayes
  filter over ONLY the in-window visible actions (uniform init; clamping concentrates the
  posterior): **B1(w16) rises to 0.45-0.61 at k>=12; B1(w8) 0.22-0.42.** (bayes_baselines.py)
  Consequences:
  (1) GridWorldV2 has a HIGH no-memory floor — the action stream alone is informative. Memory
      claims must beat B1(w), not chance/copy_last. This is a v2-specific eval insight; the
      compare plot must carry B1 as a baseline curve.
  (2) A (vanilla) sits BELOW B1 (0.14-0.30 vs 0.45-0.61 at w16): it does not even fully exploit
      visible actions. Honest but suboptimal floor model.
  (3) D's w16 plateau (~0.42-0.61) is ~AT B1(w16) — at the native window, long-horizon D is NOT
      distinguishable from action-posterior inference. At w8 D (0.42-0.58) exceeds B1(w8)
      (0.22-0.42) by ~0.1-0.25 -> the memory relay does carry something beyond actions.
  (4) Sharpest structure: D w8 k=10 (age 7, still reading write@8 which SAW the revealed context)
      = 0.94-0.98 — far above B1 — then drops to ~0.45 at k=12, exactly when the belief first
      comes from a RELAYED write (write@16 = f(write@8, occluded window)). One write->write relay
      under occlusion loses ~half the accuracy; further relays lose ~nothing (flat to k=64).
      The weak link is the belief-UPDATE-from-memory+actions operation, not storage or reach.
  NEXT: B/C dense anchor decides whether the first-relay cliff is sparse-specific or v2-intrinsic;
  then 256-rollout re-runs of the key numbers.

- 01:05 Sharpened dip timeline (desk analysis, for the dip agent to confirm): committed positions
  are 4+k, so write@8 enters the cache exactly at k=4 — the dip's onset. Within the write@8
  generation (k=4..11, w8) accuracy RISES with age: 0.70@k4 -> 0.75@k6 -> 0.84@k8 -> 0.94@k10
  (inverted staleness!), then the k=12 cliff = first branch reading the RELAYED write@16.
  So the anomaly is really two phenomena: (a) fresh-write interference (a just-written set makes
  things WORSE than an old good one — possibly the write itself is fine but reading it at small
  RoPE distance is undertrained or the branch over-weights it), and (b) the one-relay accuracy
  halving. Watchers: B/C (dense anchor), D2 415232, dip agent, memmaze 415205/415104.

- 01:40 **Dip ROOT CAUSE found (dip-investigation agent, causal spoof probes, 256 rollouts):**
  write-aligned-only training windows (all s%8==0) teach the model to DELETE the write-phase
  frame's action when integrating over a continuous cache — belief follows the del-o(write)
  trajectory at 0.81-0.88 concordance; the write CONTENT is correct (contains its own action);
  the wrong belief rides the NON-memory channels. Full report:
  experiments/gridworldv2-arms/dip-investigation/REPORT.md. Three structural footnotes:
  (a) w8 eval violates D's W>=2n relay invariant -> w8 k>=12 numbers are a relay-BROKEN regime;
      my earlier "D>B1 at w8" claim is RETRACTED (that was the register side-channel);
  (b) REGISTERS are an unrestricted memory side-channel in the sparse design (D scores 0.762 at
      w8 k=11 with zero write keys in cache!) — design question for Merlin: mask registers'
      temporal channel too, or accept and attribute honestly;
  (c) sparse trains d_min-only but eval K=4 conditions on untrained d_idx=2 (v1 precedent says
      benign; untested here).
  This also REVISES my "one-relay halving" reading: part of the k=12 cliff is one-deleted-action-
  per-generation error accumulation (deterministic artifact), not pure relay loss.
- 01:45 FIX applied (pre-registered prediction: k4-6 >= ~0.93, del-share -> ~0): window-phase
  randomization (r per call, writes stay absolute; orphan-diagonal fallback added to the mask for
  write-less window prefixes). Smoke 6/6 + all-r sweep green. Retrains launched @ f38aaea:
  **415239 sparse-n8-fix (m4), 415240 sparse-m16-fix (m16)**. Old-D2 (415232, bugged recipe)
  kept as a bug-impact comparison at m16.

- 01:25 **DENSE ANCHOR (B/C landed): both ~1.00 FLAT to k=64 at w16 AND w8.** The dense
  per-frame relay solves v2 perfectly — belief-updates-with-actions under occlusion are lossless
  across 8 slides. Consequences:
  (1) The compounding premise is now contradicted on BOTH envs (v1 411133, v2 here) — sparse
      memory cannot be motivated as a compounding fix; its case must rest on the long-reach bank
      + cache economics + (Merlin's) redundancy argument.
  (2) Env ceiling = 1.0 confirmed -> D's ~0.5 plateau is sparse's price (as trained), not env
      difficulty. Fixed retrains (415239/415240) will show how much was the phase artifact.
  (3) FF9 unnecessary replicated on v2 (B == C) — closes the scrutiny gap about 415143 not
      gating the new env.
  (4) Also noteworthy: dense passed the w8 eval too (its relay needs only 1-frame reach, no
      invariant violation there).

- 01:50 D2 (m16, BUGGED recipe) evaled: w16 dip milder (0.83-0.84 at k4-8 vs m4's 0.56-0.67),
  plateau ~0.53-0.61 (vs 0.42-0.59). Capacity helps at the margins; the phase artifact dominates.
  Honest capacity comparison = Dfix (m4) vs D2fix (m16) when 415239/415240 land (~02:15).
