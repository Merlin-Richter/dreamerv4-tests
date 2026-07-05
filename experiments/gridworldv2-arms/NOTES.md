# GridWorldV2 arms campaign — canonical record (night of 2026-07-04→05, autonomous window)

Ordered by Merlin; executed autonomously. Chronology + ops detail: `NIGHTLOG.md`. Headline figure:
`compare_w16_r256.png`. Raw curves: `recallv2_*.json`, `results_r256_w16.json`,
`results_bayes_baselines.json`, `results_covert_channel.json`; dip root-cause:
`dip-investigation/REPORT.md`.

## Provenance (all ferranti, frozen v1 tokenizer, 50ep, seed 0; data = gridworldv2 5000 eps @ 415221)

| arm | job | SHA | ckpt (checkpoints/gridworldv2/) |
|---|---|---|---|
| A vanilla-τ0 | 415222 | b3db367 | dynamics_vanilla_tau0.pt |
| B dense m2m + FF9 | 415223 | b3db367 | dynamics_m2m_ff9.pt |
| C dense m2m no-FF9 | 415224 | b3db367 | dynamics_m2m_noff9.pt |
| D sparse n=8 m=4 (write-aligned BUG) | 415226 | a85eed7 | dynamics_sparse_n8.pt |
| D2 sparse n=8 m=16 (BUG) | 415232 | a85eed7 | dynamics_sparse_n8_m16.pt |
| Dfix sparse n=8 m=4 (phase-fix) | 415239 | f38aaea | dynamics_sparse_n8_fix.pt |
| D2fix sparse n=8 m=16 (phase-fix) | 415240 | f38aaea | dynamics_sparse_n8_m16_fix.pt |

## Results (position_acc, w16 = the invariant-valid window, 256 rollouts)

| arm | in-gap k4–10 | plateau k12–64 |
|---|---|---|
| C dense no-FF9 (B identical) | 1.00 | **1.00 flat** |
| D2fix sparse m16 | 0.82–0.91 | **0.69–0.73** |
| Dfix sparse m4 | 0.75–0.83 | 0.52–0.63 |
| D sparse m4 bugged | 0.59–0.73 | ~0.50 |
| A vanilla-τ0 | 0.77–0.94 (in-window) | 0.17–0.26 |
| **B1 exact-Bayes floor** (in-window actions only) | 0.28–0.42 | **0.45–0.61** |

## Conclusions

1. **The dense per-frame relay is lossless on v2** (1.00 to k=64, both windows, FF9 or not):
   belief-updating with the action stream under occlusion, across 8 slides. Combined with v1
   (411133), the "per-frame rewriting compounds errors" premise is contradicted on every env we
   can measure. Sparse memory's case must rest on the long-reach bank + cache economics +
   redundancy (Merlin's argument), NOT on error compounding.
2. **Sparse write-slots WORK but pay ~0.3 vs dense at this scale** (n=8, 50ep): best arm
   (m16 + phase-fix) holds ~0.70 flat to k=64, clearly above the B1 no-memory floor. Whatever
   survives the early relays is carried ~losslessly for 7+ write generations — storage and reach
   are not the bottleneck; the write-update operation is.
3. **Both of Merlin's design axes confirmed**: window-phase-robust training (+~0.1; the
   write-aligned-only bug taught action-deletion at write phases — root-caused causally by the
   dip agent, `dip-investigation/REPORT.md`) and bigger memory sets (+~0.12; m16 also eliminates
   the fresh-write interference dip that m4 retains ~0.78).
4. **GridWorldV2 eval insight**: with visible actions, the no-memory floor is HIGH (B1 to 0.61
   at w16) — wall-clamping makes the action stream informative from a uniform prior. All memory
   claims on v2 must clear B1(w), not chance/copy_last. (A sits BELOW B1 past eviction — vanilla
   doesn't fully exploit visible actions. The bugged sparse arm's plateau ≈ B1 exactly — its
   memory added nothing at w16.)
5. **Design findings for the v3 spec discussion** (from the dip investigation, verified):
   (a) the w8 eval violates sparse's W ≥ 2n relay invariant — w8 sparse numbers at k≥12 are a
   relay-broken regime; (b) **registers are an unrestricted temporal side-channel** that can
   smuggle dense memory past the sparse-write restriction (0.762 at w8 k=11 with ZERO write keys
   in cache) — either mask registers' temporal channel in a follow-up arm or attribute claims
   accordingly; (c) sparse trains d_min-only but is evaluated at K=4 conditioning (v1 precedent
   benign; untested here).
6. Negative/retracted along the way: A's high-k residual is NOT a covert latent channel
   (survives sanitized true-gray commits — it's legitimate-but-partial action inference); my
   early "D beats B1 at w8" claim retracted (register side-channel in the relay-broken regime).

## Open follow-ups (Merlin's call)

- Close the remaining 0.3 gap: more epochs / n_memory scaling / mixed-visibility training modes /
  a within-gap FF9-style anchor at write slots; or accept the price and test the actual prize —
  the LONG-REACH bank (memory attending past the window over sparse writes), where dense cannot
  compete at all.
- Register-masked sparse arm for clean attribution.
- v2 sheets tooling (qualitative filmstrips) if desired; recallv2 spec sign-off (DRAFT).
