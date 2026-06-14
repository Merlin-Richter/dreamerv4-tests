# T-016 — Architect proposal: multi-step ball-motion prediction (no occlusion)

Independent mechanism-first analysis. Author: method-architect. Date: 2026-06-14.
Scope: improve **multi-step motion/position propagation, curtain-UP (no occlusion)**.
Not in scope: blind-occlusion position memory (that is op-3 / T-014, a different deficit).

Suggested fix demoted to candidate: orchestrator's "autoregressive error compounding /
exposure bias" hypothesis is treated as **candidate diagnosis D-EXP**, one of four, not the
answer.

---

## 1. Target restated (operational)

- **Y** = the model continues a coherent ball trajectory for H autoregressive steps,
  curtain-up, fed its own outputs as context.
- **Metric** = per-horizon position error `pos_err(h)` (probe-decoded x,y vs sim truth),
  curtain-up open-loop, the EXP-011/EXP-013 instrument. Baselines on record:
  copy-last = 3.2px@h1 (freeze ball); chance ≈ 20–23px.
- **"X works"** = single-step teacher-forced motion: FF7/FF9 models hit ~1.0px@h1
  (3× better than copy-last); vanilla ~4.5–4.7px (worse than copy-last).
- **Current Y (best model, FF9/FF7 ~1px-1-step):** 0.95@h1 → 4.6@h4 → 14.8@h12 →
  ≈chance by ~h16. Vanilla diverges by ~h6–13.
- **Success would read as:** `pos_err(h)` stays well below copy-last's *accumulated* drift
  out to ≥ h8–12; concretely, push the "crosses chance" horizon from ~16 to ≥ ~30, and keep
  `pos_err(h4) ≲ 2px` (vs 4.6 today). Threshold is a target, not a contract.

---

## 2. Diagnosis — which necessary condition is broken

The five necessary-condition links, checked against the numbers and the code.

**Link 1 (Representability) — NOT broken.** Latents linearly decode (x,y) at R²=0.96/2.7px
(EXP-011). The transformer has the latent window and causal attention to read the last N
frames. Capacity is present. *Do not propose architecture for position.*

**Link 2 (Information availability) — NOT broken for h1.** The single-step probe gets 1px
from the *aux-loss* models, proving the next position is computable from the context window
when the model is asked. So velocity/displacement IS recoverable from the window.

**Link 3 (Identifiability — does the objective reward motion?) — BROKEN for vanilla,
PARTIALLY broken for all.** This is the load-bearing finding. Reading `loss()`
(`dynamics_model.py:399-478`):

- Every frame `t` is independently noised at its OWN sampled τ
  (`sample_tau_d`, `:300`) and the target is its OWN clean latent `z1[t]` (`flow_loss`,
  `:457`). The bootstrap term (`:440-455`) distills along the **τ axis** (two d/2 *denoising*
  steps), NOT along the **time axis**. **There is no term anywhere that asks the model to
  predict frame t+h from a clean frame t.**
- At the high-signal end the ramp w(τ)=0.9τ+0.1 (`:463`) *up-weights*, the frame's own noised
  latent already almost determines z1[t] — the cheapest solution is "denoise the visible
  blob," i.e. effectively **copy/clean the present**, ignoring the context's motion cue. That
  is exactly the vanilla 4.5px "jitter, don't move" behaviour. V-T013 already proved this
  shortcut dominates the ramp-favored region (memory inert at high τ).
- The ONLY place a frame is required to *predict a successor* is `_ff7_loss` (`:480`,
  `z_hat[:,1:]` vs `zw[:,1:]`) and `_ff9_loss` (`:527`). EXP-014 is the clincher: the FF7
  weights reach 1px **via the plain windowed path, no relay** — i.e. the *successor-prediction
  loss itself* is what installs the 1-step motion model. So link 3 is the established cause of
  the vanilla single-step deficit, and the aux losses accidentally fix it.

So h1 is explained by link 3. But Y is **multi-step**, and h1≈1px does not transfer to h4–h12.
That residual is one of the next two links.

**Link 4 (Gradient dominance / shortcut) — contributes.** Even the aux loss only ever
trains prediction from a **near-clean / real** context frame (FF7 holds the source at
tau_ctx; FF9 at τ=0 but with the *memory carrier* supplying truth). The model is **never
trained on a context built from its own slightly-wrong predictions.** The lowest-loss
solution is a high-gain map "decode position from context, extrapolate one step." Under
iteration that map's small errors are not penalized because training context is never
self-generated → errors are free to compound. This is the orchestrator's exposure-bias
hypothesis, and it is real, but note it is a *special case of link 3*: the objective does not
contain the multi-step chain, so nothing rewards robustness-to-own-error.

**Link 4b (Context-distribution mismatch, τ) — contributes, cheap to test.** At training,
context frames carry per-frame random τ (`sample_tau_d` fills the whole (B,T) grid). At
rollout, every context frame is pinned at `context_signal=0.9` (`_denoise_next:603-607`).
The model has seen 0.9-context only as a measure-zero slice of training. Mismatched input
statistics degrade the readout map exactly in the regime rollout uses. Distinct from D-EXP
(content mismatch) and individually testable.

**Link 5 (Optimization reachability) — unlikely primary.** No evidence of grad pathology;
the 1px h1 result shows the basin is reachable. Deprioritize.

### Ranked diagnosis (by likelihood given the numbers)

1. **Link 3 — objective never trains time-axis multi-step prediction** (highest). The loss
   is a pile of independent single-frame denoisers + τ-axis distillation; the only
   successor-prediction signal is the aux loss, and it is 1-step from a clean source. The
   model is literally never optimized for "given your own h-step rollout, be right."
2. **Link 4 — exposure bias / no self-generated context in training** (high; a subcase of 3).
3. **Link 4b — τ context-distribution mismatch** (medium; cheap to rule in/out, may be a
   silent multiplier on 1 and 2).
4. **Link 1/2 for *velocity-as-state*** (low). Possible the model reads position but not a
   persistent velocity estimate, so it must re-infer motion from 2 frames each step and
   compounds. Cheap probe settles it; ranked low because the 1px-h1 result shows displacement
   is already recoverable from the window.

### Cheap probes to confirm/refute each (run before building anything)

All reuse the EXP-011/013 frozen probe + an existing trained FF9/FF7 ckpt; CPU/4070-min, no
training. Log under `experiments/EXP-018/` per protocol.

- **P1 (settles link 3 vs 4 — the decisive one): teacher-forced multi-step rollout.**
  Run the trained model autoregressively but at each step feed the **ground-truth** clean
  context window (not its own output), held at 0.9. Measure `pos_err(h)`.
  - If teacher-forced `pos_err(h)` stays ~1–2px out to h8–12 while open-loop diverges →
    **the per-step map is fine; the failure is compounding (D-EXP/link 4).** Fix = expose the
    model to its own context (scheduled rollout / multi-step loss).
  - If teacher-forced `pos_err(h)` ALSO climbs fast → **the per-step map is not a real motion
    model, it's a 1-step-only fit (link 3).** Fix = a genuine multi-step / horizon loss.
  This single probe discriminates the top two diagnoses. Run it first.
- **P2 (settles 4b): τ-context sweep.** Teacher-forced 1-step `pos_err` while sweeping the
  context τ the model is *fed* (0.5→1.0). If error is flat at the τ the model trained on but
  spikes near 0.9 → context-distribution mismatch is live. (Also: retrain-free — just changes
  the eval input.)
- **P3 (settles velocity-as-state, link 1/2): two-frame vs N-frame probe.** Linear-probe
  velocity (Δx,Δy) from (a) the last frame's latent alone, (b) last-2, (c) full window. If
  velocity needs ≥2 frames and the model's internal state doesn't carry it, an explicit
  velocity target helps; if 1px-h1 already implies it's there, skip.

**Confidence:** ~70% the dominant remaining multi-step deficit is link 3/4 (objective lacks
the multi-step chain + no self-context), with 4b a plausible 10–20% silent multiplier. P1
moves this to near-certain in one cheap run.

---

## 3. Proposals (ranked)

Hard constraints honored by all: **additive + config-gated, identity when off** (mirror the
`n_memory=0` / `ff7_k=0` / `ff9_k=0` guards already in `loss()`), single 4070 / modest
epochs, **tokenizer frozen**, eval probe + existing FF7/FF9 paths stay bit-identical.

### C0 — Cheapest-thing-that-could-work: train context frames at the rollout τ (fix link 4b)

- **Exact change:** new config `ctx_signal_train_frac: float = 0.0` (0 = identity). When >0,
  for that fraction of frames in the main diffusion loss, **override the sampled τ of frames
  used as context to `context_signal`** instead of random τ — i.e. make the training input
  statistics include the 0.9-pinned context the rollout actually uses. Implementation: after
  `sample_tau_d`, with prob `frac` set selected frames' `tau_idx` to `tau_ctx_idx`. Pure
  re-weighting of the existing loss; no new tensors, no new path. Guard: `frac==0` → untouched
  sampling → byte-identical.
- **Gradient pressure:** moves probability mass of the readout map onto the exact context
  statistics inference uses, sharpening the present→next map *in the regime that matters*.
- **Degenerate modes:** none new — it is a subset of the existing τ grid. Worst case it
  slightly reduces τ diversity → mild denoising regression; cap `frac ≤ 0.5` and monitor clean
  val/diffusion (the EXP-017 0.0017 anchor) for regression.
- **Interaction with X:** minimal; same loss, reweighted. Detect regression via val/diffusion.
- **Cost:** ~zero. No new params, no extra forward.
- **Cheap discriminator + falsifiable prediction:** if P2 showed a τ-mismatch cliff, C0 should
  flatten it and drop teacher-forced h1 toward the trained-τ value. **Falsified if** P2 shows
  no cliff (then 4b isn't the problem and C0 won't move open-loop error). Run only if P2 fires.

### C1 — TOP PICK: time-axis multi-step prediction loss (fix link 3 directly)

- **Exact change:** new config `multistep_h: int = 0` (0 = identity), `lambda_multistep: float
  = 1.0`. Add a term `_multistep_loss(z1, actions, h)` structurally a sibling of `_ff9_loss`
  (`:527`) — reuse its windowing (`idx`, fold n_t into batch). For each window
  `[t, t+1, ..., t+h]`: hold frame t (and any real prefix) at `context_signal` from the REAL
  clean latent; frames `t+1..t+h` are **denoised at finest d, but each successor is predicted
  with its context being the model's OWN prediction of the previous successor**, not the real
  latent. Loss = ramp-free `||z_hat[t+j] - z1[t+j]||²` summed over j=1..h. To keep it cheap and
  stable, use a **single forward over the h-window with the successor latent slots filled by a
  detached running prediction** (one inner rollout of length h with `torch.no_grad` on the
  context-build, grad only on the current target step) — i.e. TBPTT-1 across the time axis,
  exactly the discipline V-T014 demanded.
- **Gradient pressure:** introduces, for the first time, a gradient that says "your h-step
  *self-fed* trajectory must match the true trajectory." This penalizes the high-gain
  extrapolator whose 1-step error compounds, pushing toward a contraction map (errors shrink,
  not grow) — the exact missing force for Y.
- **Degenerate modes:**
  - (a) **Collapse to copy-last** (predict no motion → bounded error). Closed by: the term is
    on TOP of the existing per-frame diffusion loss which already rewards correct position;
    copy-last scores 3.2px/step and loses to the supervised target. Monitor that predicted
    inter-frame displacement matches sim (~3.2px), not 0.
  - (b) **Over-smoothing / mode-averaging** at the bounce (physics is near-deterministic given
    action, so low risk here, but bounces are bifurcations). Detect via error spikes at bounce
    frames; mitigate by keeping h modest (h=4) first.
  - (c) **Grad explosion through the chain.** Closed by TBPTT-1 (detach context build) — the
    V-T014 lesson. Start h=4, grad on terminal step only.
- **Interaction with X:** shares the backbone with the diffusion loss; possible tension if the
  multi-step term pulls toward over-smooth latents that hurt single-frame sharpness. Detect:
  watch clean val/diffusion (EXP-017 0.0017). Weight `lambda_multistep` ramped from small.
- **Cost:** one extra (k+1)-frame folded forward per batch, same shape as the FF9 term that
  already runs on the 4070 at bs32. Affordable.
- **Cheap discriminator + falsifiable prediction:** short run (≤30 ep) with `multistep_h=4`.
  **Predict:** open-loop `pos_err(h4)` drops from ~4.6 toward ~2px and the cross-chance horizon
  moves past ~20. **Falsified if** open-loop error is unchanged despite the term converging
  (would mean the deficit was link 4b or velocity-state, reorder to C0/C2).

### C2 — Explicit short-horizon self-rollout fine-tune (scheduled sampling; fix link 4)

- **Exact change:** config `rollout_ft_steps: int = 0`. A fine-tuning *phase* (not a loss
  term): periodically run the model's OWN `generate`-style rollout for r steps under
  `no_grad`, then take a gradient step asking step r+1 (with grad) to match truth given the
  self-generated context. Curriculum r: 1→4. This is DAgger/scheduled-sampling for the
  dynamics model. Fully gated; off → standard training.
- **Gradient pressure:** directly trains on the self-generated context distribution rollout
  actually visits — the textbook exposure-bias fix.
- **Degenerate modes:** (a) instability if r grows too fast (compounding garbage as target
  context) — curriculum + detach. (b) Can drift the latent distribution off the tokenizer
  manifold; monitor reconstruction. Slower/heavier than C1 because it needs an inner rollout
  loop in the training step.
- **Interaction with X:** higher risk of regressing single-frame denoising than C1 (it
  reweights toward a self-generated, slightly-OOD input distribution). Stage it AFTER C1.
- **Cost:** inner rollout per step → r× forwards; on 4070 keep r≤4, bs small. Moderate.
- **Discriminator:** same metric as C1; **predict** it mainly extends the *late* horizon
  (h8→h16) where C1's teacher-style term is weakest. **Falsified if** late-horizon error is
  unchanged.

### C3 — Velocity-as-target auxiliary (fix link 1/2 IF P3 says velocity isn't carried)

- **Exact change:** config `lambda_vel: float = 0.0`. Add a tiny head (gated, created only
  when on) that from frame t's latent-token outputs predicts the **latent displacement**
  `z1[t+1]-z1[t]` (or probe-space (Δx,Δy)); MSE term. Additive, identity off.
- **Gradient pressure:** forces the per-frame representation to encode instantaneous motion,
  not just position, so the next-step map needs less re-inference and compounds less.
- **Degenerate modes:** redundant if velocity is already linearly present (P3) → wasted
  capacity, possible mild interference. Only build if P3 shows velocity needs >1 frame AND
  isn't in the state.
- **Interaction with X:** small new head, low risk. **Discriminator:** P3 first; if velocity
  is already decodable, **skip C3 entirely** (do not build).
- Ranked low: most likely solves a non-problem.

### C4 — (mention, not recommended) longer context window / recurrence

Widening N or adding recurrence is link-1 machinery, and link 1 is NOT broken (R²=0.96).
Complexity theater for this deficit. Excluded.

---

## 4. Recommendation

**Run P1 first (one cheap teacher-forced rollout, no training).** It splits the top two
diagnoses in a single run:

- P1 says **per-step map is fine, compounding is the killer** → **C1 then C2** (C1 first: it
  installs robustness-to-own-error cheaply via a multi-step time-axis loss without a full
  inner-rollout training loop; C2 layered if late horizons still drift).
- P1 says **per-step map is not a true motion model** → **C1 is still the answer** (it is the
  first loss that trains multi-step motion at all), and C2 is unnecessary.

Either way **C1 (time-axis multi-step prediction loss, TBPTT-1, h=4) is the single thing I'd
build first**, because the diagnosis (link 3: the objective literally never contains a
multi-step time-axis term — confirmed by reading `loss()`) points at it directly, it is the
minimal additive/gated change that creates the missing gradient, and EXP-014 already proved
that *successor-prediction loss* — not architecture — is what produces the good per-step map.
Run **C0 only if P2 shows a τ-cliff** (near-free insurance).

**Strongest counterargument to C1:** if the real deficit is pure exposure bias (P1 shows
teacher-forced stays flat), then C1's partly-teacher-forced term may under-train the
self-generated distribution and C2 would be the more direct fix. Mitigation: C1's successor
context is the model's OWN detached prediction (not the real latent), so it already injects
self-generated context — it spans both diagnoses. If C1 plateaus on late horizons, escalate
to C2.

**Minimal-cost first step:** P1 (+P2) on an existing FF9/FF7 checkpoint, frozen probe, no
training, logged to `experiments/EXP-018/`. Then a ≤30-epoch C1 run on the 4070.

---

## 5. What would change this ranking

- **P1 teacher-forced error climbs fast** → demote "compounding," elevate "the 1px map is a
  1-step artifact"; C1 stays #1 but C2 becomes pointless.
- **P2 shows a sharp τ-context cliff** → C0 jumps to first (free) before C1.
- **P3 shows velocity needs >1 frame and isn't in the state** → C3 rises from low to a
  complement-to-C1.
- **C1 converges but open-loop error is unchanged** → diagnosis was wrong about link 3 being
  the multi-step cause; re-examine 4b/velocity, and check the tokenizer latent manifold isn't
  the bottleneck (self-generated latents leaving the C-manifold).
- **clean val/diffusion regresses past ~0.003** under C1 → tension between sharp single-frame
  and smooth multi-step; back off `lambda_multistep`, consider Pareto.

---

## 6. Open questions / undetermined

- Does the bounce (a near-deterministic but sharp bifurcation given action) induce
  mode-averaging under a multi-step MSE? Not resolved here; watch bounce-frame error.
- Do self-generated latents stay on the frozen tokenizer's manifold over h steps, or does
  off-manifold drift (not motion error per se) dominate the late horizon? A cheap check:
  decode self-generated latents through C and measure reconstruction validity vs h. If
  off-manifold drift dominates, no motion loss fixes it and the lever is a latent-space
  regularizer (out of this scope).
- Whether `context_signal=0.9` is even optimal for motion (vs color, where it was tuned in
  EXP-008) is unverified; P2 also informs this.
```