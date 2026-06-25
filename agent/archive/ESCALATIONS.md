# ESCALATIONS.md

> One entry per open question for the human. Resolutions are written back
> verbatim-in-substance; steering not written back here evaporates.


> Older resolved escalations (ESC-001 … ESC-013) are in `ESCALATIONS-archive.md` —
> grep there for a past resolution. This file keeps only OPEN entries + the most recent
> resolved one (ESC-015) for continuity.

---

## ESC-022 | 2026-06-25 | OPEN — FF9 rollout-training campaign (D-048)

### ⚠️ CORRECTED VERDICT (2026-06-25 afternoon, supersedes the overnight read below)
You were right to push on the inference. Re-evaluated under the ONE correct inference — the **normal
sliding-window rollout** (n_ctx=8, W=16, how the model is actually used) — **rollout-training does NOT
work; it HURTS.** Canonical figure: **experiments/EXP-030/compare_windowed.png**. position_acc:

  k    vanilla   FF9 no-rollout(M4)   FF9+rollout M16   FF9+rollout M4
  8     0.33        **1.00**            0.78             ~0.8
  12    0.08        **0.81**            0.33             ~0.33
  16    0.08        **0.44**            0.09             ~0.05
  colour k16: FF9-norollout **0.67** > rollout-M16 0.48 > vanilla 0.31.

- The plain **FF9-no-rollout (M4, EXP-028) is the BEST memory model** (position AND colour). Both
  rollout-trained models (M4 and M16) are WORSE — barely above the vanilla floor past k12. Wider memory
  (M16) does NOT rescue it.
- The overnight headline ("rollout-training works / static-memory win / capacity is a lever") was an
  **ARTIFACT of a W=2 noise-source 'relay' inference I invented**, which handicapped the FF9-no-rollout
  baseline to ~chance and made the rollout model look good. That inference + its results are **deleted**
  (per your cleanup ask); only the normal sliding-window inference exists now.
- **Honest mechanism guess:** the rollout loss trains the relay in an isolated 2-frame, memory-only,
  noise-source regime — mismatched to the windowed inference (full latent context) — so it shifts the
  model away from the windowed latent dead-reckoning that carries position best. Net regression.
- **Lesson (mine):** I over-claimed from a self-built inference that wasn't how the model is used. Always
  evaluate under the real-use inference first. The credit-assignment code was correctly implemented +
  independently verified — the METHOD/training-regime is what doesn't deliver, not a bug.

### Where this leaves H3 / the decision for you
- **No method improvement.** FF9-no-rollout (within-window memory) remains the best; rollout-training as
  designed is a NEGATIVE result. Options: (a) **redesign so training matches inference** — train the
  relay WITHIN the normal windowed rollout (not the isolated 2-frame memory-only regime), which is the
  obvious fix to the mismatch; (b) try DISCRETE/VQ memory; (c) accept FF9-no-rollout as the memory
  result + log this negative and move on. My lean: (a) is the cheapest test of whether the *idea* (op-3
  credit) has any legs once the train/inference mismatch is removed; if it still doesn't beat
  FF9-no-rollout, drop it.
- EXP-031 (deep) / EXP-033 (M16) are spent; nothing else running. cluster free.

--- (overnight brief below is SUPERSEDED by the corrected verdict above; kept for the trajectory) ---

Context: you went to bed and said to work autonomously through the night — "implement and train and
evaluate a few different ideas for the memory training... work empirically... reversibly." I did. This
is the consolidated present-then-stop for the whole campaign (D-048). Everything is on branch
feat/ff9-rollout-training (pushed); state files + figures committed.

### What I did (one line each)
1. **P1 probe (EXP-029)** — settled capacity-vs-credit + min depth for the relay BEFORE building.
2. **Implemented FF9 rollout-training** — the memory->memory relay (op-3) on the gradient path
   (TBPTT-k, contiguous-tail hiding, deep-clip decoupling). 7/7 self-tests; independent
   critical-claim-verifier pass (all 4 correctness claims SUPPORTED). Updating-memory inference for eval.
3. **Trained + evaluated** EXP-030 (h24), EXP-031 (h44 deep), EXP-032 (vanilla w32 control), EXP-033
   (M=16 capacity) on the clean GridWorld bench, env-direct recall A/B vs the EXP-027/028 baselines.

### The decisive read (honest; favorable on static + far-tail dynamic, modest absolute dynamic)
**Rollout-training WORKS as a credit fix and gives a persistent bounded memory that (a) carries STATIC
hidden state FLAT beyond the window and (b) with DEEP training sustains residual DYNAMIC position FAR
past the window (k>=20) where the latent-window inference has decayed to chance. Depth is a real lever
(h44 > h24). BUT absolute dynamic precision is modest and saturates short of the training depth (a
continuous-memory PRECISION cap), and training the relay TRADES OFF near-window windowed
dead-reckoning. Net: credit-assignment + depth get a real bounded dynamic memory, but to lift dynamic
PRECISION the lever is representation stability/capacity (discrete/VQ or wider memory), not more credit.**
- **STATIC colour (the win):** FF9+rollout under the updating-memory relay holds colour FLAT ~0.8 to
  k=32 (well past the 16-window), vs FF9-no-rollout windowed decaying to 0.34 and the UNTRAINED relay
  at chance. A clean beyond-window bounded-memory result (the DreamerV4 h-state goal, for static state).
- **DYNAMIC position, same-inference win:** trained relay >> untrained relay (k=6 0.52 vs 0.02; useful
  memory k~3 -> k~8). The credit-assignment fix is real and large. In-window (k<=4) near-perfect.
- **DYNAMIC position — DEPTH IS A REAL LEVER (EXP-031 h44 > EXP-030 h24).** Deeper rollout training
  pushes the knee right: graded position k8 0.48 vs 0.27, k12 0.36 vs 0.19, k16 0.25 vs 0.11. And in
  the FAR-past-window regime (k>=20) the deep relay MATCHES/EXCEEDS the no-rollout windowed model
  (graded k32: h44 0.17 vs FF9-windowed 0.11) — because the windowed latent dead-reckoning has decayed
  to chance there while the persistent bounded memory sustains residual position. So GridWorld's
  DISCRETE state extrapolates BETTER than P1's continuous probe (depth helps), but it still SATURATES
  well short of the h=44 training depth (useful only to ~k16-20, modest absolute numbers ~0.2-0.25
  graded) -> a continuous-memory PRECISION cap. EXP-033 (M=16) [RUNNING, ~12:30] tests if more
  continuous capacity lifts it; if not, the cap is representational -> discrete/VQ.
- **The trade-off (D-048 tripwire, partial):** rollout-training HURT the windowed path (position k12
  0.33 vs no-rollout 0.81) — the relay and the latent-window dead-reckoning compete for capacity. So
  rollout-training does NOT advance BEST-achievable dynamic position (no-rollout windowed still best to
  ~k16); it shifts the model toward the bounded-memory relay.
- **Controls:** vanilla w32 (bigger window) ALSO fails past ~k16 — growing the window doesn't beat
  blind dead-reckoning (the limit is ~14-16 dead-reckon steps, not window size). So there's no
  brute-force shortcut either.

### Access points
- **Figure (open first):** experiments/EXP-030/compare_rollout.png — position (exact+graded) + colour
  vs k; 5 curves (vanilla w16, FF9 no-rollout windowed [best position], FF9+rollout windowed
  [regressed], FF9 untrained relay, FF9+rollout relay [trained]) + copy-last/oracle/chance/window line.
- Numbers: experiments/EXP-030/recall_env_*.json. Reconciliation: experiments/EXP-030/NOTES.md.
- Reasoning/idea space: experiments/EXP-029-design/orchestrator_analysis.md. P1: experiments/EXP-029/.
- EXPERIMENTS rows EXP-029..033. Decision D-048. Verifier: experiments/EXP-030/verify/.

### The questions for you (my recommendation first)
1. Accept the read — rollout-training is a working credit fix that delivers a beyond-window STATIC
   bounded memory, but dynamic position is drift-capped and the relay competes with windowed
   dead-reckoning, so credit-assignment alone is NOT sufficient for dynamic state?
2. **My recommended next method: DISCRETE / quantized (VQ) memory** — a finite-state memory cannot
   drift continuously, directly attacking the EXP-030 dynamic-drift root cause (P1 + EXP-030 both point
   here). It's an architecture change worth your sign-off before I build (codebook + commitment loss +
   straight-through on the memory tokens). Alternative: consolidate the static-memory relay win + the
   honest dynamic negative as a contribution and move on. Which?
3. EXP-031 (deep) + EXP-033 (M=16) land this morning — I'll append their numbers here. If deeper/wider
   does NOT push the dynamic knee (as P1 predicts), that closes the "it's just depth/capacity" door and
   firms the discrete-memory recommendation.
Urgency: present-then-stop. Nothing new starting beyond the already-queued EXP-031/033; cluster
otherwise free. I did NOT build VQ (a real architecture change — your call). Note: ESC-020 (FF9
corrected inference 2nd seed/verifier) and ESC-021 (rollout design sign-off) are largely SUBSUMED by
this campaign (built + ran C1; the corrected inference is now exercised across many evals).

UPDATE 09:43 — **AUTH_DEAD**: socket expired overnight; RESOLVED (you re-auth'd ~12:18). EXP-033 pulled.

UPDATE 12:30 — **EXP-033 (M=16) RESULT CHANGES THE CONCLUSION (in a good way).** Widening the memory
M=4 -> M=16 (rollout otherwise identical to EXP-030) DRAMATICALLY lifts DYNAMIC position, and helps
MORE than deepening training did:
- relay position (exact / graded), M16 vs M4(h24) vs h44(M4): k8 0.83/0.84 vs 0.20/0.27 vs 0.45; k12
  0.64/0.67 vs 0.14/0.19 vs 0.33; k16 0.34/0.39 vs 0.06/0.11 vs 0.20; k32 0.19/0.24 vs 0.02/0.07 vs 0.13.
- color FLAT 0.89-1.0 to k32 (even cleaner than M=4).
=> **The dynamic-position cap was substantially a CAPACITY limit (M=4 too small to hold integrated
position precisely), NOT purely continuous-drift representation.** M=16 sustains REAL dynamic position
well past the window (k12 0.64 exact, k16 0.34) and at the far tail beats the no-rollout windowed model
— as a BOUNDED recurrent state. This was the P1 contingency the method-architect flagged ("if it's
capacity, widen memory, not a credit fix") — and it bites: BOTH levers matter (credit makes the relay
work at all; capacity sets how precisely/far it carries dynamic state).

**REVISED recommendation (supersedes "discrete/VQ first"):** the cheapest high-value next step is to
SCALE MEMORY CAPACITY further (M=32, and the wide+deep combo M=16 x h44) — capacity is clearly not yet
saturated at M=16. Discrete/VQ becomes the move IF capacity scaling saturates (precision/stability),
not the immediate first step. So the dynamic-memory result is markedly more POSITIVE than the M=4
picture implied: rollout-training + adequate memory width gives a bounded memory that carries static
AND substantial dynamic hidden state past the window. Figure updated (compare_rollout.png, darkred=M16).
Your call: (a) scale capacity (M32 + wide-deep) [my new lean], (b) discrete/VQ, (c) consolidate.

---

## ESC-021 | 2026-06-24 | OPEN — FF9 rollout-training design review (method-architect) + gating probe
Context: you agreed to a method-architect design pass on FF9_IDEA.md (op-3 / memory→memory training).
It's back, code-grounded, and productively DISAGREES with my doc on build order. Design note:
experiments/EXP-029-design/method_architect.md; gating probe (written + smoke-tested):
experiments/EXP-029-design/probe_dynamic_relay.py.

### The findings (my decisive read: I agree on all major points)
1. DIAGNOSIS (high confidence): the deficit is CREDIT-ASSIGNMENT, not architecture. _ff9_loss
   (dynamics_model.py:576) fills the intermediate frames with the learned-init placeholder, so the
   "write mem_{t+1} ← mem_t" map is on NO gradient path. The relay is representable + the info is
   present; the loss simply never trains it. EXP-028's decay-to-chance is the production echo of the
   V-T014 untrained-carry drift.
2. BUILD ORDER (overturns my doc): rank C1 > C2 > C3.
   - C1 (recommended): extend _ff9_loss to run the intermediate memory chain for REAL (model writes
     each intermediate memory token), retain graph k hops (TBPTT-k). Same gradient as my doc's scheme,
     far less surface area. One knob: depth k (measured, not guessed).
   - C2 = my doc's cached-grad streaming scheme. Same gradient IN KIND but high correctness risk
     (grad through RoPE-rotated K/V while evicting; must assert latent K/V detached). It's the
     implementation you GRADUATE to for open-ended training, not the first build.
   - C3 = cheap tbptt-2 patch to _ff9_loss as a control/floor (likely insufficient per V-T014, but free).
3. OPEN QUESTIONS: newest-frame-only flow loss (agree; older frames are teacher-forced near-clean →
   trivially low, dilutive). Per-step Bernoulli(p≈0.5) latent-hiding, mixture not all-hidden (re-anchor).
   DON'T hard-code 4·N — MEASURE min k. Butterfly = non-issue in deterministic GridWorld (an asset);
   defer credit-curving to stochastic envs.
4. GATING EXPERIMENT (P1): extend the V-T014 harness to a DYNAMIC secret (position+velocity, integrate
   each hop) + tbptt-k sweep. Settles the two things that gate the whole build, in ~30–60 min, NO
   GridWorld training: (Q1) can ANY trained relay carry dynamic state — if even full BPTT fails, it's a
   CAPACITY problem (M=4 too small) → pivot to widening memory, NOT a credit fix; (Q2) the minimum k =
   C1's truncation horizon. Script ready + smoke-tested.
5. DEPENDENCY FLAG: the diagnosis premise (FF9 carries dynamic position in-window) rests on EXP-028's
   corrected inference, whose 2nd-seed + verifier pass is ESC-020 Q2 (still open).

### Recommendation (mine)
Two cheap, independent de-risks BEFORE any GridWorld build, both present-then-stop:
- Run P1 (dynamic-secret tbptt-k probe) → capacity-vs-credit + the depth k.
- Close ESC-020 Q2 (2nd FF9 seed + critical-claim-verifier on the corrected inference) → firm the premise.
Then a build decision = C1 at the measured k (newest-only flow + per-step Bernoulli hide + wall-clock
warmup), for your sign-off. I am NOT building or running until you weigh in.
Urgency: present-then-stop. Nothing in flight; cluster idle.

## ESC-020 | 2026-06-24 | OPEN — EXP-028 FF9 v2 memory method on GridWorld [CORRECTED] (present-then-stop)
Context: you said train FF9. Trained budget-matched to vanilla (job 409625); env-direct recall A/B vs
vanilla, frozen scorer (D-045), N=64/k. CORRECTION: my first read ran FF9 through the FROZEN-snapshot
inference (generate_full_state_memory: one snapshot, 2-frame [noise|new] window, no dead-reckoning) →
k=1 position=0, period-10 spikes. You flagged it ("laughably wrong"). The INTENDED FF9 inference is the
normal autoregressive rollout with per-frame memory tokens carried in the sliding window via temporal
attention (added generate_cached plain=True). Numbers below are the corrected inference. §5 present-then-stop.

### The result (decisive read, CORRECTED)
**With the right inference, FF9 is a real DYNAMIC-memory win — it tracks the hidden square's POSITION
through occlusion far past where vanilla fails, not just static colour.** (This reverses my earlier
"static-only / position needs op-3" read, which was the wrong inference.)
- POSITION (the hard, dynamic metric): FF9 in-window 0.94 vs vanilla 0.52; past-window (k≥16) 0.20 vs
  0.05. Per-k: FF9 1.00 through k8, 0.89@k10, 0.73@k14, 0.44@k16, smooth-decaying to chance by ~k28.
  The curve is a SMOOTH DECAY (no period-10 spikes) → genuine motion integration through the blind run,
  not a frozen snapshot. k=1=1.0 sanity holds.
- COLOUR/BG (static): FF9 also beats vanilla past window (colour 0.48 vs 0.30, bg 0.62 vs 0.26 weighted)
  but DECAYS rather than staying flat at 1.0 — so static retention is helped but not perfect at this budget.
- Vanilla cliffs at its window edge on everything (as EXP-027).

### Access points
- compare.png (open first): experiments/EXP-028/compare.png — vanilla vs FF9, 4 panels.
- Numbers: experiments/EXP-028/recall_env_{vanilla,ff9}.json. Reconciliation: experiments/EXP-028/NOTES.md.

### The questions for you
1. Accept the corrected read — FF9 carries BOTH dynamic position (smooth decay, ≫ vanilla) and static
   colour past the no-memory window; the earlier frozen-snapshot result was an inference artifact?
2. Given this is a stronger-than-expected claim and I already got the inference wrong once, I propose a
   critical-claim-verifier pass on the corrected plain-rollout inference + a 2nd seed BEFORE we treat it
   as a headline. Agree? And regenerate the FF9 sheets with the corrected inference to show it qualitatively?
3. Then direction: push horizon further (op-3/relay for the k>24 decay), or consolidate FF9 as the
   GridWorld memory result first?
Urgency: present-then-stop per §5 — nothing started beyond this. Cluster idle.

## ESC-019 | 2026-06-24 | RESOLVED — EXP-027 vanilla GridWorld dynamics baseline recall (present-then-stop)
RESOLUTION (Merlin, 2026-06-24): baseline ACCEPTED ("we will use this as baseline for now. Maybe later
a longer trained one. But for now this is already very good"). Also corrected my window-8 caveat: a
shorter inference context is IN-distribution for a causally-masked transformer (memory:
causal-prefix-in-distribution). Directed: (1) train FF9 next → D-047/EXP-028 (job 409625, in flight);
(2) raised that evals should use the env directly, not the val set (headline.png used the val set) →
to convert (memory: evals-use-env-directly). Sheets accepted after the env-based occlusion rework.

## ESC-019-orig | 2026-06-24 | (superseded by resolution above)
Context: you green-lit training the vanilla baseline + building the rollout-recall adapter. Trained
(job 409479, val diffusion 0.0146→0.00139, stable). Rollout protocol audited by critical-claim-verifier
→ SUPPORTED on action-alignment / reveal-decode / no-leak (V-EXP027). Recall run (job 409559) on the
150 held-out val episodes through the frozen recall core (D-045). §5 present-then-stop.

### The result (decisive read)
**The no-memory FLOOR is established, textbook-clean: vanilla retains hidden state exactly up to its
16-frame temporal window and nothing beyond.** Three findings, all from one figure:
1. REAL within-window memory+reasoning on POSITION: model exact-cell acc 0.573 (k≤14) vs copy-last
   0.118 (~5× chance-adjusted) — it remembers the last-seen square AND dead-reckons the bounce through
   up to ~14 hidden steps (imperfect: integrating motion blind is hard).
2. HARD cliff at the window edge (k=15): past the window position → 0.015 (below chance AND below
   copy-last 0.191 — it actively hallucinates), and even the STATIC colour collapses to chance (0.265)
   while copy-last trivially holds it at 1.0. The cleanest possible "no memory past the window."
3. The cliff is MEMORY LOSS, not weak dynamics: the matched-horizon control (curtain held UP) tracks
   position ~0.70 FLAT at every k incl. past the window — the model CAN propagate motion arbitrarily
   far given observations; it fails under occlusion only because it can't retain hidden state past the
   window. Deficit cleanly isolated.
Both D-046 tripwires clear (not ≈oracle past window → env not too easy; in-window+control good → no
tokenizer/budget confound). oracle=1.0 (instrument valid).

### Access points
- headline.png (open first): experiments/EXP-027/headline.png — position graded/exact + ball/bg colour
  vs k; model (blue) vs control (green, flat) vs copy-last (red, spiky) vs oracle; purple line=window edge.
- Numbers: experiments/EXP-027/results.json. Reconciliation: experiments/EXP-027/NOTES.md. Rows EXP-027,
  V-EXP027.

### The questions for you
1. Accept this as the GridWorld no-memory baseline floor (memory to the window, cliff past it, isolated
   as memory loss by the control)?
2. This is the H-gridworld milestone the whole pivot was building toward — the clean bench is live
   (frozen tokenizer + frozen eval + audited rollout + vanilla floor). Ready to move to the MEMORY
   method on GridWorld (the FF7/FF9/op-3 line, now on the clean discrete bench)? If so I'll bring a
   method proposal + decision for your sign-off before training.
3. Minor housekeeping for your awareness (not blocking): (a) dynamics_vanilla.pt not yet archived under
   experiments/EXP-027/ (the staging cp was skipped when the job tripped on a missing-matplotlib at the
   very end; it's safe on ferranti); (b) I'll fix the run.sh convention so checkpoints save into the
   pullable run dir. Both folded into the next dynamics run.
Urgency: present-then-stop per §5 — not starting the memory method until you weigh in. Nothing in flight;
cluster idle.

## ESC-018 | 2026-06-24 | RESOLVED — EXP-026 tokenizer-roundtrip recall CEILING (present-then-stop)
RESOLUTION (Merlin, 2026-06-24): "Yes this all makes perfect sense." Ceiling accepted (latent not the
bottleneck → build dynamics on the frozen tokenizer); proceed to training the vanilla dynamics baseline.
IMPORTANT CORRECTION to my framing: **colour DOES require memory** (static retention through occlusion —
a model can still hallucinate the wrong colour, so it's a real, if easy, memory test), while **position
requires reasoning INSIDE memory** (retain + simulate the bounce forward through hidden steps). So it's
NOT "position is the only memory metric": colour = static-retention memory, position = memory+reasoning.
Report BOTH (see memory project_gridworld_metric_semantics). Next: vanilla GridWorld dynamics baseline
on the cluster (D-046). Q1/Q2/Q3 all = yes.

## ESC-018-superseded-framing | 2026-06-24 | (kept for the record)
Context: you approved freezing the eval (D-045) + running the ceiling check. EXP-026 ran the FROZEN
recall core on tokenizer encode→decode of the TRUE frames — the upper bound on what any dynamics model
on this frozen latent can recall (it predicts latents decoded the same way). 500 episodes, ~13k reveal
events. This is the §5 present-then-stop for the experiment.

### The result (decisive read)
**The frozen tokenizer is NOT the bottleneck — its representational ceiling equals the oracle.**
tokenizer-roundtrip recall == oracle == **1.0000 at EVERY k** for graded position, exact position
(chance 1/36), ball colour, and bg colour; there is no (k, metric) where the roundtrip falls below the
oracle. The square's exact cell + colour are read back bit-perfect from the recon at every reveal,
even at k=42. So decode(z) is faithful for the true latent z — any downstream recall failure will be
the dynamics model's MEMORY, not the tokenizer's capacity. D-044 tripwire cleared.

Second, load-bearing finding: **in GridWorld, position is the ONLY memory metric.** copy-last (freeze
last-seen square) already scores color_acc = bg_acc = 1.0 at every k because the square's colour is
static — colour cannot discriminate memory here (it could in the occluded line, where colour was the
hidden attribute). The position copy-last curve behaves exactly as predicted: spikes to 1.0 at k≡9
(mod 10) and sits at chance otherwise → judge memory by beating copy-last PER-K (D-045). This
vindicates D-033's position-first headline.

Honest caveat: the ceiling proves the decoder faithfully renders a square the encoder SAW; it does not
test whether the latent manifold is smooth/predictable enough for a dynamics model to hit the right z
— that's a dynamics question, out of scope for a representational ceiling.

### Access points
- **Headline (open first):** experiments/EXP-026/headline.png — 4 panels (position graded/exact, ball
  colour, bg colour) vs k; tokenizer line sits exactly on the oracle line, copy-last decays/spikes.
- Numbers: experiments/EXP-026/results.json. Reconciliation: experiments/EXP-026/NOTES.md. Row EXP-026.

### The questions for you
1. Agree the latent ceiling is clear (roundtrip == oracle everywhere) → green-light building dynamics
   on this frozen tokenizer without revisiting it?
2. Agree that in GridWorld position is THE memory metric (colour/bg are static identity checks copy-last
   already aces) → headline = position vs k beaten against copy-last per-k?
3. Next phase = vanilla GridWorld dynamics baseline on the cluster (record a decision first; grad-clip
   fix + batch re-profile), then wire the dynamics-rollout frame source + matched-horizon control into
   the adapter and run real recall curves. Go?
Urgency: present-then-stop per §5 — not starting the dynamics run until you weigh in. Nothing in flight;
cluster idle.

## ESC-017 | 2026-06-23 | RESOLVED — EXP-025 GridWorld tokenizer WORKS (present-then-stop, recommend freeze)
RESOLUTION (Merlin, 2026-06-23): "We now have the working tokenizer. Lets get the eval down."
= accepts the tokenizer (Q1 yes) and the freeze (Q2). I copied it → checkpoints/gridworld/tokenizer.pt
(D-044, provenance run 70k76148 @ f1e3d6c). He redirected to nailing down the eval next; the eval-
design freeze + periodicity + W&B-wiring questions remain his call under ESC-016 Q1 (still open).

Context: you flagged the failed tokenizer had a LOSS EXPLOSION (loss went really low, then exploded,
recovered worse) and asked for logging + a fix, run on the cluster with W&B, and to report back when
the tokenizer actually encodes the colored square (not fooled by low MSE). I pulled EXP-024's W&B curve
— it confirmed the explosion and RE-DIAGNOSED the failure (D-043): val/mse hit 6e-4 @ep9 (ball being
learned) → exploded @ep10 (62×) → recovered to a worse plateau; the single-checkpoint overwrite
discarded the good ep9 model. So the failure was instability, NOT sparse-target collapse. Fix shipped
(adam-beta2 0.95 + grad-spike skip + best-checkpoint by val/fg_mse + per-step grad-norm logging + modest
fg-weight). One pre-existing DataLoader bug (persistent_workers caching a stale per-epoch index) crashed
the first launch at ep2; fixed (persistent_workers=False) and rerun.

### The result (decisive read)
**Success — we have a usable GridWorld tokenizer.** EXP-025 (job 408760, run 70k76148) trained the full
30 epochs with NO explosion: val/mse decreased monotonically 0.0056 → 4.7e-6 (130× below EXP-024's
pre-explosion peak), latent_cos 0.08–0.11 (no collapse), grad_norm bounded (the spike-guard caught &
skipped ~6 spike steps — the EXP-024 failure mode, now neutralised). The reconstruction strips VISIBLY
show the colored square at the correct cell in a colour distinct from the background (blue/red/green
balls on red/purple/blue grids), tracking ball position AND colour — not a background-only cheat. The
corrected diagnosis is vindicated: with training stable, the ball is learned easily.
Honest caveat: the foreground mask flags ~32% of pixels (it catches the moving curtain, not just the
~1% ball), so val/fg_mse is a curtain-diluted ball guard — the recon VISUAL is the real confirmation,
and it's unambiguous. Not blocking; flagged for the record.

### Access points
- **Recon strips (open first):** `experiments/gridworld-tok-v3/recon.png` — GT (top of each pair) vs
  reconstruction (bottom); colored square present + correct colour in the recon rows.
- **W&B run:** transformer-C-tokenizer / gridworld-tok-v3 (id 70k76148) — val/mse, val/fg_mse,
  grad_norm_epoch, skipped_epoch, latent_cos curves.
- Checkpoints staged: `experiments/gridworld-tok-v3/{tokenizer.pt (best=ep29), tokenizer_last.pt}`.
- Full reconciliation: `experiments/EXP-025/NOTES.md`; decision `D-043`.

### The questions for you
1. Agree this is a usable tokenizer (ball position + colour encoded, no collapse, no explosion)?
2. **FREEZE it as the GridWorld backbone** → `checkpoints/gridworld/tokenizer.pt` (provenance run
   70k76148 @ f1e3d6c)? (I've staged it locally; not copied to the frozen path until you bless it.)
3. Then the next phase is the vanilla GridWorld dynamics — which also needs **ESC-016 Q1** resolved
   (formally freeze the recall eval + decide periodicity handling + where the during-training W&B eval
   lives). Want to settle Q1 now so the dynamics run is unblocked?
Urgency: present-then-stop per §5 — I am NOT freezing or starting the dynamics until you weigh in.
Nothing in flight; cluster idle.

## ESC-014 | 2026-06-14 | OPEN — relay-training credit design (V-T014 REFUTES pure detach) — present-then-stop
Context: you specified Mode B (op-3 relay) with the memory carry DETACHED ("can and should be detached"). I
wrote it up (T-014) and ran it past critical-claim-verifier, which built a decisive synthetic probe. This
is a §5 present-then-stop / §7 escalation: the result is surprising AND reopens a design choice you set.

### The result (decisive read)
**Detached carry does NOT give a stable deep relay — it preserves state only up to the trained rollout
depth, then drifts to chance.** Synthetic relay (GRU writer, a hop-0 secret never re-supplied, per-step
recover loss; train depth 32, eval to 200), recovery MSE (chance ≈0.98):
```
depth   no_relay  detached(ModeB)  tbptt1   bptt
 ≤31      0.98       ~0.0003        0.0004   0.0001   <- in-window: detach looks PERFECT (= BPTT)
 100      0.98        0.149         0.077    0.0018
 199      0.98        1.08          0.451    0.0135   <- past train depth: detached ≈ chance; BPTT flat
```
deep-avg(100/150/199): detached **0.587** vs BPTT **0.007** (84× worse) vs tbptt1 0.255. Detached drift
d199/d16 = 3589×. **The trap:** within-window FF9 loss→0 *green-lights* detached, but it's blind to the
drift that happens exactly in the deep regime op-3 exists to serve. Mechanism (your Bellman analogy, now
qualified): the detached carrier is a *consistency* fixed point with **no per-step content anchor** (unlike
Q-learning's observed reward), so it's free to slowly rotate/shrink its code while every step's loss stays
satisfied. **Only BPTT-through-time extrapolates; 1 hop (tbptt-1) only partially helps.**
Caveat: synthetic GRU + STATIC secret — proves the *credit mechanism* fails (architecture-general), not a
production hop-count; the real DYNAMIC-state relay is strictly harder (no copy attractor).

### The reconciliation (so we're precise about what's refuted)
Refuted = "detach gives a relay sufficient across many hops *without training that deep*." NOT refuted =
"train the per-step loss across the full N-step rollout, detached, and it holds within that N." Your
"~200 steps" reading IS the latter and is viable to depth N — but it costs a depth-N sequential rollout
every iteration, gives ZERO extrapolation beyond N, and is only validated for static state.

### Access points
- Curve (open first): `experiments/verify-T014/probe_curve.png` (MSE vs depth, 4 arms, train-depth line).
- Numbers: `experiments/verify-T014/results_v2.json`; probe `probe_detached_relay_v2.py`. Row V-T014.

### The decision (gradient design for the relay — your call; I recommend a cheap probe first)
The probe tested the extremes: detach (0 grad hops) fails to extrapolate, BPTT (∞) works, tbptt-1
insufficient. The sweet spot — the **minimum BPTT depth k that buys stable extrapolation** — is unknown and
**cheap to find** (~30 min: sweep tbptt-k k∈{2,4,8,16} + a norm/projection on the carry, same harness).
- **P-a (recommended): run that tbptt-k sweep before building.** One number (k) determines the whole Mode B
  loop (how much graph to keep, how deep to roll, memory cost). Then build with the winning k + guardrails.
- **P-b: accept train-to-depth detach now** — run the full deep rollout, per-step loss, detached; cheapest
  gradient, bounded memory, but no extrapolation + deep-rollout cost + static-only-validated. Build now.
- **P-c: also add a DYNAMIC-state probe variant** (secret = a moving quantity to integrate) — the real
  unknown the static probe doesn't cover; cheap, and arguably the more important de-risk than k.
My lean: **P-a + P-c together** (one short probe session settles both the cheapest k AND whether the relay
works for dynamic state) BEFORE the expensive Mode B build. Guardrails to fold in regardless (V-T014):
gate on a DEEP-HOP sufficiency metric not within-window FF9 loss; norm/projection on the relayed activation;
detach the committed K/V (+assert) in the cache; strict-FF9-fraction a tuned knob.
Urgency: blocking per §5 — not recording D-025 or building Mode B until you weigh in. Nothing in flight;
4070 idle. (Process note: the verifier edited EXPERIMENTS.md (V-T014 row) and CLAUDE.md (a `-u` tip) despite
the no-canonical-files instruction; both are correct so I kept them and reconciled as writer.)

## ESC-015 | 2026-06-14 | RESOLVED — present EXP-017 (FF9 v2 memory-token baseline, full eval) — present-then-stop
RESOLUTION (Merlin, 2026-06-14): "great findings. Consider ESC-015 resolved." The EXP-017 decisive read is
accepted — the FF9 v2 memory-token line is a validated architectural baseline for STATIC hidden state
(color retained flat at ceiling past the window, strictly beating FF7/vanilla; position not retained → op-3).
Same turn he asked to make the interactive viewer support the new memory rollouts → D-025/T-015 (done).
Next frontier remains the op-3 sequential relay for DYNAMIC state (ESC-014, still open).

Context: EXP-017 (D-024) is the FF9 v2 memory-token ARCHITECTURAL BASELINE you chose (ESC-013): a distinct
MEMORY token type + the leak-free FF9 v2 objective (path frames τ=0, memory injected, loss on the target),
trained 100 ep on occluded at the EXP-010/012 budget. Training finished overnight; this is the full eval.
Per §4 I settled the (genuinely hard) beyond-window inference design with `critical-claim-verifier` BEFORE
building it — verdict SUPPORTED for **A1+B1** (write a full-state memory snapshot ONCE from the prefix,
inject it static at a τ=0 source frame each step; the re-extract relay B2 is the untrained op-3 and drifts;
near-clean source A2 gave identical recall). EXPERIMENTS rows EXP-017 + V-T013-eval. §5 present-then-stop.

### The result (decisive read)
**FF9 v2 is a clean baseline that OVER-DELIVERED: it retains static hidden COLOR PERFECTLY and FLAT through
arbitrary occlusion — strictly better than FF7, not the ≈FF7 we predicted.** Three findings:
1. **Beyond-window color: flat at ceiling to n_occ=48 (6× the window).** color ΔRGB stays 12–14 (ceiling
   ~13, chance ~105, T-004 bar 63) at EVERY n_occ from 2 to 48, and occluded ≈ matched-horizon drift at
   every point (12.2 vs 12.3 @48 → occlusion adds ZERO color loss). FF7 k3 decays 17→85 (crosses the bar
   ~n_occ 22); vanilla cliffs to chance at the window edge (n_occ 8). FF9 dominates both everywhere past
   the window. **Why flatter than FF7:** A1+B1 carries a written-once snapshot that CANNOT drift, so a
   static attribute is held forever; FF7 re-extracts its register each step (one-hop relay) and drifts.
   Each model uses its own faithful inference, so the comparison is fair.
2. **Within-window memory sufficiency (PRIMARY) — strongly load-bearing.** With the whole path at τ=0
   (memory is the ONLY carrier), memory-only prediction of t+j: L(mem) 0.018/0.025/0.033 vs L(no-mem)
   0.27 (j=1/2/3), chance 0.41, copy-last 0.38/0.63/0.69 → closes 88–93% of the gap, ~20× below chance,
   and ≪ copy-last (copy-last climbs to 0.69 as the ball moves while L(mem) stays ~0.03) → memory captures
   MOTION within the window, it is not a static frame copy.
3. **No base-dynamics regression — improved.** Clean held-out val diffusion 0.00172 vs vanilla_s0 ~0.0066
   (~3.8× sharper) — the same dynamics-regularizer effect FF7 showed (EXP-014).

**The honest caveat (the half it does NOT solve):** dynamic POSITION is not retained — posErr ~20–30px at
all n_occ for FF9, the same as vanilla and FF7; latent-MSE stays near chance (position-dominated, the
T-004 confound). The frozen snapshot cannot integrate motion. So FF9 v2 perfectly carries STATIC hidden
state; carrying DYNAMIC state needs the memory to UPDATE across steps — exactly op-3 / the sequential
relay (T-014, ESC-014), which this working write+read substrate now de-risks and motivates.

All three D-024 tripwires checked and clear (memory load-bearing ✓; no regression ✓; color not worse than
FF7 — it's better ✓). No HALT condition; the surprise is favorable.

### Access points (low-friction view)
- **Headline (open first):** `experiments/EXP-017/headline_color.png` — color recall vs n_occ, 3 models +
  ceiling/chance/T-004-bar; FF9 flat line vs FF7 decay vs vanilla cliff, obvious at a glance.
- **Primary readout:** `experiments/EXP-017/memory_sufficiency.png` — L(mem) vs L(no-mem) vs copy-last/chance.
- **Qualitative:** `experiments/EXP-017/sheet_ff9.png` — GT(top)/prediction(bottom); the predicted reveal
  ball matches GT COLOR at every n_occ incl. 48 (position off — color held, position not).
- Numbers: `experiments/EXP-017/frozen_color.json`, `primary.json`. Full reconciliation: `NOTES.md`.
  Inference-design audit: `tasks/T-013-eval-inference.md` + `experiments/verify-T013-eval/` (V-T013-eval).

### The question for you
1. Agree with the read — FF9 v2 cleanly + perfectly retains static hidden COLOR beyond the window (flat at
   ceiling to n_occ=48), strictly beating FF7's drifting relay and vanilla's cliff; the mechanism is a
   non-drifting written-once full-state snapshot; position (dynamic) is unsolved and needs op-3?
2. Is this enough to call the **memory-token architecture baseline a success** and the H3 memory line
   validated for static state (color)? (It exceeded the "≈FF7" bar you set in D-024.)
3. Direction — my recommendation: this is the green light for the dynamic-state extension. The blocker is
   **ESC-014** (still OPEN): the op-3 relay gradient design — P-a (cheap tbptt-k sweep to find min BPTT
   depth that extrapolates) [my lean] / P-c (dynamic-state probe) / P-b (train-to-depth detach). I lean
   **P-a + P-c together** (one short probe session: cheapest k AND whether a relay can carry dynamic state
   at all) BEFORE the Mode B build. Your call on ESC-014 unblocks the relay; or redirect (e.g. a 2nd FF9
   seed to firm the flat-color claim first; or fold FF9 v2 into the writeup and pause).

Urgency: blocking per §5 — I am not starting the relay build, ESC-014 probes, or any follow-up until you
weigh in. Nothing is in flight; the 4070 is idle. Code committed (FF9 eval @ 0f02f18); gates green
(FF9 9/9, FF7 5/5, KV 5/5, stream 9/9).

## ESC-016 | 2026-06-16 | GridWorld pivot milestone: eval sign-off + compute-tier steer | OPEN
Context: Merlin live-steered a pivot (D-032) to a discrete GridWorld memory env and a clean eval,
then "structure checkpoints by env" and "start a vanilla-model smoke test (10 epochs) on the new data."
Done this session (all committed): GridWorldEnv + datagen (curtain schedule 90/5/5) + **gridworld.npy
3000x200 generated**; recall eval (`src/evals/gridworld/`) + instrument validation; checkpoints
reorganized to `checkpoints/<env>/` (D-034); gate tests green; docs synced.

Two things for your review:

1. **Eval scoring design — sign-off requested (D-033, the "vital decision").** HEADLINE = **position
   recall accuracy vs occlusion length k** (exact 8x8 cell) + **color recall (4-way)**; diagnostics =
   reflection split (learned the walls vs ballistic) + readout margin; references = oracle ceiling
   (=1.0), **copy-last/no-memory** baseline, chance (1/64). Readout is closed-form & provably exact on
   true frames (oracle=1.0; copy-last decays 0.08@k1->0; random~1/64). **Key choice:** I PROMOTED
   position to the headline (the fluid env had to demote it to a drift-confounded non-metric) because
   it is the only attribute that CHANGES under occlusion — the genuine dynamic-memory test. Is
   position-first right, or do you want color-first continuity with the H2/H3 line? Not frozen until
   you bless it.

2. **Compute tier.** The 4070 runs this tokenizer at ~9 s/it (GPU-bound; EXP-006's real tokenizer was
   trained on galvani, not locally). So a full 10-epoch run on the 6.9GB set is ~25h locally. I started
   a reduced local smoke (300-ep subset, 10 epochs, ~2.7h, running) to validate the gridworld pipeline
   end-to-end (tokenizer -> then vanilla dynamics). **Question:** for the REAL gridworld pipeline
   (tokenizer + vanilla baseline on full data), do you want it on the cluster (proper, per §6), or keep
   iterating locally at reduced scope? The local smoke continues either way as a pipeline check.

Resolution (partial, 2026-06-18):
- **Q2 (compute tier) — ANSWERED = cluster.** Merlin directed "Its time to work on the cluster
  interface scripts" → the GridWorld pipeline goes to the cluster. Built T-003 (`scripts/`, D-035).
  Correction he gave: two clusters **ferranti (H100)** and **galvani (A100)**, no default — pick per
  live fairshare/queue. Code sync = remote git fetch+checkout. (Local smoke already completed: W&B
  zjvhcn4s, val/mse 0.00216, latent_cos 0.37 — pipeline validated end-to-end on the 300-ep subset.)
- **Q1 (GridWorld eval design sign-off, D-033 position-first headline) — STILL OPEN.** Not frozen;
  model adapter not wired. Tokenizer training on the cluster does not need this; the downstream
  dynamics RECALL eval does.

Still awaiting: (a) Q1 eval sign-off; (b) Merlin to fill `scripts/cluster.env` + open the master
socket so the cluster wrappers can be live-tested before the first real cluster job.

Update (2026-06-18, later): (b) DONE — cluster live, T-003 validated end-to-end, first real tokenizer
run in flight (job 405629). (a) Q1 PROGRESSED — Merlin gave a refined eval spec, built as D-040
(graded position_score + ball/bg colour + per-k stats; validated). STILL OPEN for Q1: formally
FREEZE the eval + decide periodicity handling (copy-last spikes to 1.0 at k≡9 mod 10 on the 6×6 env →
judge per-k; periodic-W&B eval should use off-grid k like {3,6,12,16}) + where to wire the during-
training W&B eval. Tokenizer training does not need this; the dynamics recall eval does.

RESOLUTION — Q1 (2026-06-24): Merlin approved the freeze ("you can continue with that"). Eval CORE
frozen (D-045): readout.py + scoring/aggregation/baselines in recall.py. Periodicity = judge PER-K,
periodic W&B eval uses off-grid k {3,6,12,16}. Model-based frame sources go in a separate adapter that
imports the frozen core. ESC-016 fully resolved (Q1 + Q2 both answered).
