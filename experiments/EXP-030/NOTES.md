# EXP-030 (+031/032) — FF9 rollout-training NOTES

## Purpose
Test whether training the memory->memory relay (op-3) on the gradient path (FF9 rollout-training,
D-048) makes the FF9 memory tokens carry DYNAMIC hidden state (square position) through occlusion past
where the untrained relay / the no-memory baselines fail. Gated by EXP-029 P1 (recall horizon ~ train
depth for dynamic state) -> train the relay to the eval depth.

## Inference semantics (READ THIS before interpreting the A/B)
Three inferences, all through the FROZEN recall core (D-045), env-direct episodes (N=64/k, n_ctx=8):
- **windowed** (EXP-028 corrected): plain sliding-window `generate_cached(plain=True)`. Memory tokens
  carried WITHIN the 16-frame window via temporal attention; latents are the main carrier. The model
  has the full latent window. Vanilla uses this (no memory).
- **relay** (D-048, the inference rollout-training trains): `generate_updating_memory`. A 2-frame
  `[noise-source | new]` window, the persistent memory token RE-WRITTEN and carried each step (op-3/B2).
  Source = pure noise (A1) => **memory is the ONLY carrier**; there is NO latent window. This is the
  honest "is memory a sufficient state" probe (the imagination north-star), but it is HARDER than
  windowed because it discards the latent history. NB: under relay, the curtain-UP "control" is NOT a
  free-run-in-the-clear (source is still noise) -> it is memory-only with a curtain-up action, so its
  position is also memory-limited, not a dead-reckoning ceiling. Don't read the relay control as the
  EXP-027 in-the-clear control.
- **snapshot** (B1, EXP-017/028 ref): frozen memory snapshot, static state only.

The clean attribution of "does rollout-training help" = **FF9+rollout vs FF9-no-rollout under the SAME
relay inference** (isolates training). The relay-vs-windowed gap is the inference difference.

## Baseline established this session: FF9 (no rollout) under RELAY (untrained B2 relay)
`recall_env_ff9_norollout_relay.json` (FF9 v2 checkpoint from EXP-028, gridworld-ff9-s0/dynamics_ff9.pt).
The untrained updating relay **collapses within ~2-3 hops**:
- position_acc (chance 0.028): k1 0.67 -> k2 0.42 -> k3 0.17 -> k4 0.08 -> ~chance (<=0.06) for k>=5.
- color_acc (chance 0.25): k1 0.98 -> k4 0.61 -> k8 0.28 (~chance) -> stays ~0.25-0.34.
=> The "before" picture: re-writing the memory each step WITHOUT training the relay corrupts it almost
immediately (matches V-T013-eval: B2 drifts). This is the bar FF9+rollout must clear under relay.
For windowed-inference baselines reuse experiments/EXP-028/recall_env_{vanilla,ff9}.json (vanilla cliffs
at the 16-window; FF9 decays to chance ~k28).

## Runs (pending; eval via EVAL_RUNBOOK.md when checkpoints land)
- EXP-030 (job 409752): FF9+rollout h24, window16, clip28, tbptt12, tail, +ff9 3, warmup20.
- EXP-031 (job 409754): FF9+rollout h44 deep, clip48, tbptt16.
- EXP-032 (job 409753): vanilla window-32 control.

## Reconciliation (EXP-030 h24; N=64/k, env-direct, frozen scorer)
Expected (D-048): FF9+rollout under relay holds position past k=4 (clears the untrained-relay
collapse) and past the 16-window; in-window (k<=8) NOT regressed.

Observed (position_acc, chance 0.028):
- FF9+rollout RELAY: k1-3 ~0.98-1.0, k4 0.83, k5 0.73, k6 0.52, k8 0.20, k12 0.14, k16 0.06 -> ~chance.
- FF9-norollout RELAY (untrained B2): k3 0.17, k5 0.06, k6 0.02 -> chance by k4.
  => ROLLOUT-TRAINING WORKS as designed: ~3-4x longer useful dynamic-position memory (k~3 -> k~8),
     same inference. The credit-assignment fix is real and large. In-window (k<=4) near-PERFECT (1.0),
     better than EXP-027 vanilla in-window (0.57) -> NO regression (D-048 tripwire clear).
- BUT position decays to chance by k~12-16, SHORT of the h=24 training depth -> the continuous M=4
  memory DRIFTS on dynamic state, exactly P1's (EXP-029) prediction for a continuous dynamic relay.

Observed (color_acc, STATIC, chance 0.25) — the clean WIN:
- FF9+rollout RELAY: 0.86/0.86/0.84/0.88/0.78 at k16/20/24/28/32 — FLAT, high, BEYOND the window.
- FF9-windowed (EXP-028): 0.67/0.47/0.38/0.39/0.34 — decays past window.
- FF9-norollout RELAY: ~0.28-0.31 — ~chance.
  => The trained PERSISTENT relay carries STATIC hidden state flat beyond the window (the DreamerV4
     h-state goal) where the latent-window inference decays and the untrained relay collapses.

Cross-inference context (NOT a fair same-inference A/B; different carriers):
- FF9-norollout WINDOWED holds position BEST to ~k16 (k12 0.81, k16 0.44, chance ~k28) because the
  16-frame LATENT sequence carries it (open-loop), not a bounded memory. The relay (2-frame, pure
  memory) is inherently weaker on position. Neither solves beyond-window (k>16) DYNAMIC position.
- **TRADE-OFF (D-048 tripwire, partial): rollout-training HURT the windowed path.** FF9+rollout under
  WINDOWED: k8 0.88, k12 0.33, k16 0.05 — vs FF9-norollout windowed k8 1.00, k12 0.81, k16 0.44. So
  the rollout objective shifted capacity toward the persistent relay at the COST of the windowed
  latent dead-reckoning (the best dynamic-position path). The in-window k<=8 hit is mild (1.00->0.88);
  the real regression is k12-16. => rollout-training does NOT advance BEST-achievable dynamic position
  (the no-rollout windowed model is still best); it improves the RELAY path (and static color) while
  degrading the windowed path. The two carriers compete for capacity. Warmup 20ep / M=4 may be too
  tight (EXP-033 M=16 tests capacity).
- vanilla w32 (windowed): peaks ~0.58, decays to chance by ~k16 -> growing the window to 32 did NOT
  beat blind dead-reckoning (the limit is dead-reckoning ~14-16 steps, not window size). The memory
  method does not need to beat brute-force context here because brute-force context also fails.

Surprise: mild. The relay-training win on dynamic position is real but smaller-horizon than the h24
training (drift caps it ~k8-12); the STATIC-color beyond-window win is clean and strong. Consistent
with the whole project (static retained, dynamic hard) + P1 (continuous dynamic relay drifts).

Hypothesis impact (H3): SUPPORTED for STATIC hidden state via a trained bounded memory (carried flat
beyond window). For DYNAMIC position: rollout-training is a real, large improvement over the untrained
relay but does NOT achieve beyond-window retention -> the binding constraint is REPRESENTATION
STABILITY (continuous memory drift), not credit. Next lever = DISCRETE / quantized memory (VQ: a
finite-state memory can't drift continuously) or more memory capacity, NOT more credit. (See
EXP-029-design/orchestrator_analysis.md idea 2.)

Tripwires (D-048): in-window k<=8 NOT regressed (helped) ✓; training stable (val 0.081, rc=0) ✓;
P1's "no free extrapolation for dynamic state" CONFIRMED on the real task (position decays well before
h24). Not a HALT — favorable-leaning with a clear, honest limit + a principled next step.

Next: (1) EXP-031 deep h44 relay (running) — does deeper training push the position knee right, or is
it drift-capped regardless (P1 says capped)? (2) EXP-030 WINDOWED eval (does rollout help/hurt the
windowed path vs EXP-028?). (3) Morning decision for Merlin: pursue DISCRETE memory (VQ) for dynamic
position, OR consolidate the static-memory relay win + the negative dynamic result. Present-then-stop
at the consolidated brief (ESC-022).

## EXP-031 (deep h44) result — DEPTH IS A LEVER
Relay position_acc h44 vs h24: k8 0.45 vs 0.20, k12 0.33 vs 0.14, k16 0.20 vs 0.06, k32 0.13 vs 0.02;
to k44: ~0.05-0.13. Graded position k16 0.25 vs 0.11. Color FLAT 0.83-0.91 to k44.
- Deeper rollout training (h44) clearly pushes the dynamic-position knee right (h44 > h24 throughout
  k8-32) -> GridWorld's DISCRETE state extrapolates better than P1's continuous probe; depth helps.
- FAR-tail crossover: graded position k>=20 h44 (0.24/0.17 @k20/k32) MATCHES/EXCEEDS FF9-norollout
  WINDOWED (0.29/0.11) -> the persistent bounded memory outlasts the latent window where it has decayed
  to chance. The relay's regime is FAR past the window.
- BUT saturates: even h44 (trained to depth 44) only holds useful position to ~k16-20 (modest
  ~0.2-0.25 graded), far short of k44 -> continuous-memory PRECISION cap. -> EXP-033 (M=16) tests
  capacity; discrete/VQ tests stability.

## EXP-033 (M=16 wider memory) result — CAPACITY IS A MAJOR LEVER (revises the conclusion)
Relay position M=16 vs M=4(h24) vs h44(M4): k8 0.83/0.20/0.45, k12 0.64/0.14/0.33, k16 0.34/0.06/0.20,
k32 0.19/0.02/0.13 (exact); graded k16 0.39/0.11/0.25. Color FLAT 0.89-1.0 to k32.
- Widening memory M=4 -> M=16 helps DYNAMIC position MORE than deepening training (h24 -> h44) did.
  => the M=4 dynamic cap was SUBSTANTIALLY a CAPACITY limit (M=4 too small to hold integrated position
  precisely), NOT purely continuous-drift representation (the P1 / method-architect capacity contingency).
- M=16 + rollout = a BOUNDED recurrent memory carrying static AND substantial dynamic state past the
  window (k12 0.64 exact), beating the no-rollout windowed model at the far tail. Markedly more positive
  than the M=4 picture. REVISED next lever: scale capacity (M=32, wide+deep) before discrete/VQ.
