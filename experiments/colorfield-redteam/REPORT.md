# Red-team of the ColorField comeback eval (composite_gated) — verdict report

(Independent adversarial red-team, background agent, 2026-07-06. Runnable exploits + numbers in
this directory: exploits.py, harness.py, run_*.py, artifacts/full_analysis.txt.)

## VERDICT: EXPLOITABLE (moderate-to-high severity) — at the ORIGINAL scoring semantics

Claim tested: "composite_gated is high iff the model performs genuine long-range retention."

Headline numbers (original semantics: additive 0.7·real + 0.3·consistency, equal-weight bins,
no chance correction; prefix 192 / imag 768 / min_events 30, all 6 bins qualified):

| Adapter | composite_gated | real | consist | actually knows |
|---|---|---|---|---|
| oracle | 1.000 | 1.00 | 1.00 | ground truth |
| GeoOutWorld(W=inf) | 1.000 | 1.00 | 1.00 | full prefix memory (framework sanity) |
| GeoOutWorld(W=128) | 0.766 | 0.666 | 1.00 | 128-frame buffer |
| **GeoOutWorld(W=64)** | **0.623** | 0.462 | 1.00 | **64-frame buffer; CHANCE past 64** |
| GeoOutWorld(W=32) | 0.561 | 0.372 | 1.00 | 32-frame buffer |
| HonestShortMem(W=64) | 0.496 | 0.451 | 0.60 | honest 64-frame |
| perfect_imaginary | 0.428 | 0.183 | 1.00 | ZERO content retention |
| **HonestShortMem(W=16)** | **0.251** | 0.210 | 0.35 | honest 16-frame |

Core defects:
- **S2 (the big one)**: equal-weight mean over qualified age bins with a 0.2 chance floor —
  "win near bins, chance far bins" caps the penalty for total long-range failure. A bounded
  64-frame buffer (the DEFAULT transformer context) clears 0.6 while at chance in the entire
  regime the eval exists to measure.
- **S1**: free consistency (0.3 additive) — zero-retention liar (0.43) OUTSCORES honest-16
  (0.25): metric non-monotone in memory capability.
- **S3**: OUT tiles at weight 0.1 still lift real by ~+0.06–0.08 via geometry-only knowledge.
- **S4**: adapter_factory receives the real env (honor system) — env.map ⇒ instant 1.0.
  Must be sandboxed at the harness level.
- **S5**: registration is exactly recoverable from prefix frame 0's OUT bands + action integral
  (verified 40/40) — exploits need no env peek.
- Gates HELD: fidelity blocks confinement/OUT-sea steering; entropy blocks collapse;
  bin-starvation and OUT-flooding attacks FAILED.

Ranked fixes proposed: (1) age-weight or top-bin scoring; (2) per-bin chance correction;
(3) hard long-range gate; (4) neutralize free consistency (drop or gate/multiply);
(5) in-map-only accuracy; (6) sandbox env access.

Methodology self-correction (logged): first short-memory adapter silently had full memory
(refreshed last-seen at re-entry; the tracker reads at max visibility) — caught via paint-vs-read
dump; all reported numbers use the corrected freeze-belief-once-per-visit version. Lesson: the
eval records at max visibility, so a faithful bounded-memory model must decide remember/forget
once per visit.

## Definitive exchange-rate curve (original semantics, frozen eval config, 6/6 bins qualified)

GeoOutWorld (persistent aligned world, correct-OUT geometry, genuine memory window W):

| W | 0 | 16 | 32 | 48 | 64 | 96 | 128 | 256 | inf |
|---|---|---|---|---|---|---|---|---|---|
| composite_gated | 0.473 | 0.496 | 0.561 | 0.584 | **0.623** | 0.700 | 0.766 | 0.882 | 1.000 |

Composite crosses 0.6 at W ≈ 56–64. HonestShortMem: W=16 → 0.251, W=64 → 0.496, W=128 → 0.668.
Zero-retention floor 0.43–0.47 > honest-16 (0.25): non-monotone. The ">0.6 = long-range memory"
reading is false under the original semantics.

## Resolution (applied after this report — see eval_comeback.py + tests)

Fix bundle implemented by the orchestrator (Merlin sign-off pending before freeze):
per-bin CHANCE CORRECTION (in-map c=1/5, OUT c=1/6, clamped at 0) + MULTIPLICATIVE consistency
(composite = real_cc × (0.7 + 0.3·consistency_cc) — consistency can only amplify real retention,
never substitute for it) + privileged/sandboxed adapter factories (real models get None, not env)
+ a bounded-window monotonicity regression test in the frozen gate tests.
