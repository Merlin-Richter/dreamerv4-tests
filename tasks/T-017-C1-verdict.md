# T-017 C1 — Independent verification verdict

Verifier: critical-claim-verifier. Date: 2026-06-14. Status: pre-implementation audit.
Artifacts: `experiments/verify-T017-C1/` (probe_c1_contraction.py, probe_c1_gain.py, results*.json);
`EXPERIMENTS.md` row `V-T017-C1`.

## Restated neutral claims under test (proposer framing stripped)
The design proposes a config-gated additive loss `_multistep_loss(z1, actions, h)`: for each anchor
frame t, seed a context from real near-clean latents, then for j=1..h run a single finest-d forward
with a **τ=0 pure-noise target slot**, take flow loss `||ẑ − z1[t+j]||²` against the **real GT**
successor, **detach** the prediction, append it to the context (TBPTT-1), continue.

- **C-A.** This loss creates gradient pressure that REDUCES open-loop position-error compounding by
  training a contraction map, because the τ=0 target must be predicted from the model's OWN detached
  self-generated context. Is the mechanism sound, or is the loss minimizable without improving
  open-loop robustness?
- **C-B.** Are the named degenerate minimizers (copy-last, off-manifold drift, bounce mode-averaging,
  single-frame sharpness) closed by the stated guards, and are there OTHERS?
- **C-C.** Are the cheap approximations faithful: (i) single finest-d forward vs K=4 inference;
  (ii) TBPTT-1 detach — does it destroy the gradient that fixes compounding?
- **C-D.** With `multistep_h==0`, is the change byte-identical (no RNG-order change, no new params,
  inference/probe/FF7/FF9 untouched)?

---

## Verdicts

| Claim | Verdict |
|---|---|
| C-A | **TRUE UNDER CONDITIONS** — sound, but the stated *mechanism is mislabeled*. |
| C-B | **TRUE UNDER CONDITIONS** — named guards mostly hold; one un-named degenerate mode is open. |
| C-C(i) | **TRUE UNDER CONDITIONS** — acceptable proxy with a quantifiable, mostly-benign bias. |
| C-C(ii) | **REFUTED** (the worry) — detach does NOT destroy the needed gradient. |
| C-D | **PROVEN TRUE** conditional on copying the ff9 guard pattern exactly. |

---

## C-A — mechanism: contraction map vs on-policy distribution correction

**The "contraction map" framing is mechanistically wrong, but a *different*, sound mechanism makes
the loss useful.** Two distinct things are being conflated:

- *Mechanism-A (claimed): lower the learned map's local gain `|df/dx|` so drift shrinks.*
- *Mechanism-B (actual): make the map ACCURATE on the off-trajectory states the rollout visits
  (DAgger / scheduled sampling).*

These are different, and an anchored-GT-target loss **cannot** generically deliver A. The per-step
target is the *true* `z1[t+j]` = `f(true state at t+j)`. The GT successor of any state — on- or
off-manifold — is the true `f` of that state, whose gain is fixed by the data. So the loss drives the
map toward the **true** local Jacobian, not toward a contraction. If the true latent dynamics is
locally expanding (the bouncing ball's free flight is roughly neutral/expanding in latent space, and
your EXP-013 "open-loop chaos" finding implies gain ≥ 1), C1 will faithfully learn that expanding map
— it will *not* manufacture a contraction that damps its own error.

**Probe (Part 1, `probe_c1_gain.py`, seed 0):** true `f(x)=A·x`, `A=1.10`, single learnable scalar
gain `g`, noiseless. TF learns `g=1.1000`; **C1 learns `g=1.1000`** — identical. C1 does not push `g`
below `A`. There is no spurious contraction; the "pushes toward a contraction map" claim is false as
stated.

**What C1 *does* legitimately do (Mechanism-B):** it supervises the map on the model's own drifted
states, so prediction error at the states a rollout actually visits is reduced — the textbook
exposure-bias / DAgger fix. This *can* reduce open-loop compounding **iff the deficit is
off-manifold-accuracy** (the model is locally fine but visits states it was never trained on). It will
**not** help if the deficit is intrinsic high-gain dynamics (a small error is amplified by the true
Jacobian regardless of how accurate the map is) — that is irreducible and no loss fixes it.

Given your empirical context — teacher-forced per-step error is already ~1px FLAT for ff7/ff9, yet
open-loop diverges by h12-16 — the deficit is *exactly* the off-manifold case (the single-step map is
good; the rollout visits states where it is wrong). **So C1 targets the right failure for ff7/ff9.**
But for *vanilla* (teacher-forced 4.5px, a genuinely weak single-step map) the deficit is partly
intrinsic, and C1 will help less.

**Can it be minimized without improving open-loop robustness?** Yes, two ways: (1) if the term's
capacity is spent and the one-step term degrades (see C-B(d) below — confirmed in Part 2), and (2) the
unlearnable-context mode in C-B. So C-A is conditionally true.

## C-B — degenerate minimizers

**Named guards:**
- (a) copy-last: **closed.** Copy-last scores ~3.2px/step against a moving target; the GT-anchored
  flow loss strictly prefers the moving prediction. Sound. (Keep the displacement monitor.)
- (b) off-manifold drift: **acknowledged, not closed** — correctly flagged as out-of-scope. Real risk:
  over h steps the detached self-context can leave the frozen-tokenizer manifold, so the GT target is
  being chased from an invalid context (see un-named mode below). The decode-validity monitor is the
  right gate; keep h modest.
- (c) bounce mode-averaging: partially mitigated by small h; intrinsic to MSE at a bifurcation. OK to
  start at h=4 and watch bounce-frame error.
- (d) single-frame sharpness regression: **real and confirmed empirically.** In `probe_c1_gain.py`
  Part 2 (capacity-limited), the C1 map is *worse* one-step (0.024 vs 0.012) because the multi-step
  term competes for capacity, and it is worse at h24 (1.060 vs 1.040) even though it edges TF in the
  transient (h6 0.557 vs 0.601). This is precisely guard-(d)'s tension and it can dominate when the
  model is capacity-limited. The `λ_multistep` ramp + the val/diffusion≤0.003 tripwire are the correct
  mitigations — treat them as load-bearing, not optional.

**Un-named degenerate mode (the important one):** **context-ignoring prior emission under an
unlearnable drifted context.** When the detached self-context has drifted far from any real
trajectory, the GT target `z1[t+j]` is genuinely *not a function of* that context — there is no map
from a wrong context to the right answer. The MSE-optimal response is then to **ignore the context and
emit the conditional mean (a prior)** — i.e. blur toward the dataset-average trajectory. This both
(i) fails to improve open-loop robustness and (ii) can *leak into* the shared weights and soften
genuine context-reading. This is the time-axis analog of the V-T013 shortcut (loss minimized without
memory being load-bearing). It worsens monotonically with h, which is the real reason to keep h small
and ramp λ. **Add a monitor:** the multistep term's per-j loss should *not* be flat-or-rising toward a
context-independent floor; compare `L_j(self-context)` against `L_j(prior/zero-context)` — if the gap
collapses at large j, the model is emitting a prior there (mask or down-weight those j, mirroring the
FF9 horizon mask).

## C-C — faithfulness of approximations

**(i) single finest-d forward vs K=4 shortcut rollout: acceptable proxy.** The finest-d x-prediction
is the *same network output* the K-step sampler integrates; at finest d the single forward IS one
exact flow step. The train/infer gap is that inference composes K substeps with intermediate τ levels
and re-draws the pure-noise z, so the *self-context distribution* C1 trains on is slightly
narrower/cleaner than inference visits. This biases C1 toward a mildly optimistic self-distribution,
but in the same direction (on-policy), so it still corrects the dominant exposure bias. Low risk; the
falsifiable A/B will detect if it under-delivers. Cheaper and correct.

**(ii) TBPTT-1 detach: does NOT destroy the needed gradient. The worry is REFUTED.** Analytic probe
(`EXPERIMENTS.md` V-T017-C1; reproduced inline): the detached step-j forward `m(x_ctx_detached)→ẑ`
graded against GT produces a gradient **bit-identical** to teacher-forcing the map *at the model's own
visited state* `x_ctx_detached` (grad norm 4.71521 both ways). Detach removes ONLY the
"through-context" term `∂ẑ/∂θ via x_ctx` (the BPTT term) — which is exactly the term the V-T014 audit
demanded be cut for stability — and leaves the full distribution-correction (DAgger) signal intact.
So TBPTT-1 yields a coherent objective: "predict the true successor of the states your own rollout
visits." This is the crucial distinction from the T-014 relay (REFUTED): **there the carrier had no
GT anchor (a self-consistency condition → free to drift); here every step has a GT anchor, so detach
is safe.** C1 is not the T-014 failure mode.

## C-D — identity-when-off

**Byte-identical, conditional on copying the ff9 guard pattern exactly.** Verified
(`EXPERIMENTS.md` V-T017-C1): `model.loss(z1,a)` vs `model.loss(z1,a,ff9_k=0,ff7_k=0)` are
`torch.equal` (0.19189071655273438 both), confirming the additive-guard pattern draws RNG in the same
order. C1 must: (1) call `_multistep_loss` only inside `if multistep_h > 0`, (2) place ALL its RNG
draws (the pure-noise target z, the context noise) inside that branch and AFTER the existing
`sample_tau_d`/`z0`/diffusion + bootstrap draws (the bootstrap block runs unconditionally), (3) add no
`nn.Parameter`. It reuses the existing forward, so `generate*`, the frozen probe, FF7, FF9 are
structurally untouched. The single hazard is hoisting any `_multistep_loss` RNG before the guard or
reordering existing draws — the smoke test (`test_n_memory_zero_is_additive_identity` analog) must
assert seed-matched equality with multistep_h=0.

---

## Caveats & scope
- The probes are synthetic 1-D learning-rule isolations, NOT the production dynamics model. They
  decisively settle the *gradient mechanism* (contraction vs DAgger), the *detach* question, and the
  *capacity-tension* direction — they do NOT predict the magnitude of the open-loop gain on
  occluded.npy (that is what the proposed A/B measures). The latent dynamics' true local gain is
  unmeasured here; if it is ≥1 (likely, per EXP-013), expect C1 to help via Mechanism-B, not by
  damping.
- I did not run the production loss (C1 is unimplemented). C-D's identity claim is verified on the
  existing guard pattern, not on C1 code.

## If flawed: the defect & minimal fix
- **Defect (framing, load-bearing for expectations):** the design's stated mechanism — "pushes toward
  a contraction map" — is wrong; an anchored-GT loss recovers the true (possibly expanding) gain. The
  real mechanism is on-policy distribution correction. **Fix:** restate the rationale as DAgger /
  scheduled-sampling on the self-induced state distribution; this also reframes the falsifiable
  prediction: C1 helps to the extent the deficit is off-manifold accuracy (ff7/ff9 case), and the
  FALSIFIED-IF branch "deficit was velocity-state" should be widened to "deficit was intrinsic
  high-gain dynamics" (irreducible).
- **Defect (open degenerate mode):** prior-emission under unlearnable drifted context. **Fix:** add the
  context-vs-prior gap monitor per horizon j and mask/down-weight horizons where the self-context has
  decohered (the time-axis analog of FF9's load-bearing check), plus keep h small and ramp λ.
- **Defect (capacity tension, confirmed):** the multi-step term can degrade the one-step map. **Fix:**
  treat the val/diffusion ≤0.003 tripwire and the λ ramp as mandatory gates, not monitors; consider a
  stop-gradient-free *separate* small head only if regression bites.
