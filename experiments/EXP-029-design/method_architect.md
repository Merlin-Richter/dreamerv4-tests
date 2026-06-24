# EXP-029 — Design note: rollout training for FF9 memory→memory propagation (operation 3)

method-architect, 2026-06-24. Independent design + pressure-test of `FF9_IDEA.md`.
Status: design only. No canonical/frozen files touched. Probe scripts under this dir.

> Suggested fixes from `FF9_IDEA.md` treated as candidates, not the answer:
> rollout-mode at ~25–50% wall-clock, random window length {2,4,…,N}, flow + k-step sufficiency
> losses, KV+grad caching truncated at 4·N, per-step latent-hiding, teacher-forced GT context,
> stability guardrails. Demoted to candidate **C-DOC** below.

---

## 1. Target restated (operational)

- **Y (target capability):** *memory→memory propagation across context windows* — operation (3):
  read REAL previously-written memory tokens from context and write the next memory token, such
  that hidden state (square **position** = dynamic; **colour/bg** = static) is preserved past the
  point where the informative latent has evicted from the N-frame window.
- **Metric / threshold:** GridWorld env-direct recall vs occlusion length k (`src/evals/gridworld/`,
  frozen core D-045), judged **per-k vs copy-last**. Today (EXP-028, corrected inference): position
  1.00 through k8, 0.73@k14, 0.44@k16, → chance by ~k28. **Success = the decay knee moves right**:
  concretely, position recall at k=24 rising from ~chance toward the in-window level, and the curve
  flattening (slope of recall-vs-k near zero in the 16<k<32 band) WITHOUT regressing in-window (k≤8)
  or base 1-step motion. A weaker but still-publishable success: static colour/bg goes flat past the
  window (it currently decays — §1 EXP-028).
- **"X works":** FF9 already (a) writes memory from real latents and (b) reads memory to reconstruct
  the next k frames within a single mini-window where frame t+j attends DIRECTLY to the injected
  memory_t. That is operations (1)+(2). It works (EXP-028 in-window 0.94).

---

## 2. Diagnosis — which necessary condition is broken

**Broken link: identifiability / credit-assignment for the memory→memory transition (links 3+6).
The relay function is REPRESENTABLE and the information is AVAILABLE, but the objective exerts
zero gradient on the multi-hop memory-write chain.** Grounded in code:

1. **`_ff9_loss` never instantiates a chain of real memory writes.** (lines 543–601.) Per window it
   injects `mem0 = mem[:, :n_t]` (real, from the main pass) at frame 0, and fills frames 1..k with
   `self.memory_tokens.expand(...)` — the **learned init**, NOT a re-written memory (line 576,
   `mem_rest = self.memory_tokens.expand(BN, k, -1, -1)`). The terminal frame t+j attends, via the
   position-wise temporal channel, **directly back to frame-0's memory slot**. So the loss credits
   (1) writing mem0 from latents and (2) reading mem0 j hops later — but the memory at the
   intermediate frames is a constant placeholder. **The "write memory_t+1 from memory_t" map is on
   no gradient path.** This is the docstring's own admission ("trains memory as a sufficient
   full-state object, NOT the cross-window relay").

2. **The architecture CAN carry it; that is not the deficit.** Temporal attention is position-wise
   (`forward`, lines 159–207: temporal permute puts N on batch, T on sequence) → each memory slot is
   an independent causal channel through time. Spatial blocks (lines 167–171, unmasked) mix
   memory↔latents within a frame. So a function "read prev memory slot + this frame's
   action/latent → write next memory slot" exists in the hypothesis class. (Link 1 OK.)

3. **The information is available** the moment a real chain exists: the readout ceiling is met
   (EXP-026 roundtrip == oracle), the latent→position probe is R²=0.96 (EXP-011), and EXP-028 shows
   memory DOES contain position in-window. (Link 2 OK.)

4. **Why it decays to chance specifically (the V-T014 mechanism).** At inference the model carries
   memory via temporal attention over many hops, a relay it got **zero gradient** for. V-T014 proved
   the failure mode of an untrained/within-window-only carry precisely: in-window aux-loss→0
   green-lights it, but past the trained horizon the carrier is a *consistency* fixed point with no
   per-step content anchor and drifts monotonically to chance (detached d199 84× worse than BPTT;
   tbptt-1 only partial). EXP-028's smooth decay-to-chance-by-k28 is the production echo of that
   synthetic curve. **Confidence: high** that this is the dominant cause; the diagnosis is the same
   mechanism already proven in two places (V-T014 synthetic, h3-line memory).

**Caveat that sharpens the whole design:** V-T014 used a STATIC secret. GridWorld position is
DYNAMIC — the relay must not just *preserve* a code but *apply the transition* (integrate the bounce
under the known action) each hop. That is strictly harder than the proven-failing static case, and
it changes what "minimum BPTT depth" means (§4). The static-colour decay in EXP-028 is the V-T014
static-drift; the position decay is static-drift PLUS un-trained dynamics-in-memory.

**So: the fix family is OBJECTIVE + CREDIT (put real memory chains on the gradient path with enough
through-time gradient to anchor the write), NOT architecture.** Do not add modules.

---

## 3. Ranked design options for the rollout-training loss + gradient scheme

All options share: teacher-force GT context latents near-clean (isolate the memory relay from latent
drift — agreed, this is correct and matches the V-T014 reader-anchor logic). They differ in *where
the real memory chain comes from* and *how deep the gradient flows*. Ranked best-first.

### C1 (RECOMMENDED) — Unrolled-window sufficiency: extend `_ff9_loss` to a TRAINED intermediate chain, TBPTT-k

**What:** keep the existing FF9 mini-window structure but instead of placing `self.memory_tokens`
at frames 1..j−1, **run the model forward frame-by-frame so each intermediate memory slot is the
REAL memory the model just wrote**, carried via the temporal channel, with the autograd graph
retained for k hops then detached. Concretely: a short teacher-forced rollout of length up to the
window, GT latents held near-clean at every frame, the memory channel is the only recurrent element,
loss = the memory-sufficiency target (predict t+j from a τ=0 / hidden slot) AND/OR flow on the newest
frame. The single knob is the **TBPTT depth k** (how many hops of memory-write graph to keep).

- **Gradient pressure:** the loss at frame t now backprops through `write(mem_{t}) ← mem_{t-1}` for k
  hops, directly rewarding the *construction* of an intermediate memory token from the previous one —
  the exact map that is currently un-gradiented. This is the one force that points at Y.
- **Degenerate optima & how each is closed:**
  - *Memory ignored when latents present.* Closed by hiding the path latents (τ=0) on the
    memory-sufficiency branch (as `_ff9_loss` already does) so memory is the only carrier on that
    branch — there is no latent to copy.
  - *Memory collapses to a constant / identity copy (the V-T014 consistency fixed point).* This is
    the real danger and is **NOT closed by within-window loss** (that's the V-T014 trap). Closed by
    (i) a DYNAMIC target — position must be *updated* each hop, so a frozen code is wrong at t+1, not
    just at t+30; and (ii) gating the run on a **deep-hop** metric (recall at k>window), never on the
    rollout loss. A constant-memory solution gets a non-zero per-hop loss immediately because the
    square moved — this is why GridWorld's determinism is an ASSET here (see below).
  - *Trivial within-graph shortcut: keep grad only 1 hop and let drift hide past it.* That is exactly
    tbptt-1, which V-T014 showed is insufficient. Closed by choosing k from the §4 sweep, not k=1.
- **Interaction with X (operations 1+2, base dynamics):** shares the FF9 substrate; risk is the
  unrolled branch's gradient fighting the within-window sufficiency gradient. Mitigated by the
  wall-clock warmup (learn to *contain* before *propagate*) and by keeping the plain diffusion +
  current `_ff9_loss` as the majority of steps. Detect regression via in-window recall (k≤8) and base
  val-diffusion as tripwires (both already logged).
- **Cost:** k sequential forwards per anchor with graph retained → ~k× the FF9 mini-forward memory.
  k≈4–8 (§4) is affordable. No new modules, no new hyperparameter beyond k + the warmup schedule +
  the rollout-step fraction.
- **Cheap discriminating experiment:** the §4 synthetic tbptt-k sweep (DYNAMIC secret) gives k BEFORE
  any GridWorld training; then one short GridWorld run at that k. **Falsifiable prediction:** at the
  k the synthetic says extrapolates, GridWorld position recall at k=24 rises above its EXP-028 chance
  floor and the 16<k<32 slope flattens; if it still decays identically to EXP-028, the mechanism is
  wrong (or k too small / capacity-bound at M=4).

### C2 — Cached cross-step rollout with retained memory-K/V graph (the C-DOC scheme, made precise)

**What:** the doc's proposal — extend `stream_rollout_init/step` so the memory-token K/V keep their
autograd graph across committed frames (detach the latent K/V), truncate at ~4·N, random window
length, per-step latent hiding. This is C1's mechanism but realised through the streaming cache
instead of a fresh short unroll.

- **Gradient pressure:** identical in KIND to C1 (through-time gradient on the memory write chain).
- **Why it ranks BELOW C1:** higher engineering surface + correctness risk for the SAME gradient.
  Specifically (grounded in the cache code):
  - `stream_rollout_step` is `@torch.no_grad()` and commits memory K/V *pre-rotated at absolute
    positions* (lines 1113, 1149–1153). Retaining grad means removing no_grad, keeping the graph on
    only the memory columns while `_evict_oldest` (line 1056) slices away old columns — eviction must
    drop graph nodes cleanly or memory leaks the graph. This is exactly the "detach the committed
    frame-latent K/V, keep grad only on memory" surgery the doc asks for; it is correct in principle
    but every step is a place to silently attach the wrong tensor (e.g. RoPE-rotated K depends on the
    memory activation → grad path must survive rotation).
  - **Degenerate/failure mode unique to C2:** if the latent K/V detach is imperfect, gradient leaks
    through the (teacher-forced, near-clean) latent channel and the loss is satisfied by the latent
    copy, not the memory relay — silently re-creating the "memory ignored" degenerate the whole
    exercise is meant to avoid. Detect with an assert that no latent-K/V tensor `requires_grad`.
  - 4·N truncation is a guess; C1's k comes from a measured sweep.
- **When C2 wins:** only if you need open-ended (k ≫ 4·N, interactive) memory imagination during
  TRAINING, or if C1's per-anchor k-unroll is too memory-heavy at the real model size. Until C1's k
  is known, C2 is premature optimization of an unproven loss.
- **Cheap experiment:** none cheaper than C1's; C2 is the *implementation* you graduate to after C1's
  k and sign are established.

### C3 (cheapest-that-could-work) — TBPTT-2 patch to `_ff9_loss`, no rollout loop at all

**What:** the minimal intervention. In `_ff9_loss`, replace the placeholder `mem_rest` at frame 1
only: do a 2-frame sub-forward [t, t+1] that WRITES mem_{t+1} from mem_t (real, injected), then use
that real mem_{t+1} as the injected memory for the t+1.. mini-window. One extra hop of real memory
write, full grad (k=2). Everything else unchanged.

- **Gradient pressure:** trains exactly one memory→memory hop with gradient — the smallest possible
  dose of Y.
- **Honest prediction (likely INSUFFICIENT):** V-T014 showed tbptt-1 only partially helped and did
  not extrapolate; tbptt-2 is one hop more and may still not anchor a long relay. **Its value is as a
  control / floor**, not the answer: if even tbptt-2 visibly bends the EXP-028 curve, the mechanism
  is confirmed cheaply; if it does nothing, it tells us the depth must be larger and C1 is justified.
- **Cost:** ~1 extra forward per anchor; a few lines in `_ff9_loss`; no rollout infra, no cache
  surgery. Config-gated identity-when-off (mirror the `ff9_k=0` guard) per the codebase convention.
- **Cheap experiment:** it IS the experiment — run FF9 with this patch at the existing budget vs
  EXP-028. Discriminates "is one trained hop enough" for free.

### C-DOC (the doc as written) — folded into C2

C-DOC = C2 + per-step latent hiding + random window + warmup. The per-step latent-hiding and warmup
are good and I keep them (§5). The objection is ordering: C-DOC builds the hardest piece (grad-cache
through the streaming relay) before establishing the loss has the right sign and the needed depth.
Build C1 (or even C3) first; graduate to C2/C-DOC for open-ended training only if needed.

---

## 4. Recommendations on the open questions

### (a) Flow loss: newest frame only vs all in-window
**Newest frame only — agree with the doc's lean, with a reason it understates.** Flow-matching all
in-window frames re-weights each frame ~window-size times AND, worse, the older frames are
teacher-forced near-clean GT, so their flow loss is *trivially low and uninformative* — it dilutes
the gradient with easy examples and competes for capacity. Inference only ever commits the newest
frame, so newest-only also removes a train/test mismatch. **Keep the memory-SUFFICIENCY loss
multi-frame** (it is where the memory signal lives), flow-match newest-only.

### (b) Per-step latent-hiding fraction + re-anchoring
**Agree with per-step (not per-rollout) hiding; recommend p_hide ≈ 0.5, and a mixture, never
all-hidden.** Mechanism: a hidden step (τ=0 latents) gives the memory-only gradient (the only signal
that forces memory to carry state); a visible step (GT latents near-clean) **re-anchors the rollout
to the true trajectory** so a single wrong memory guess cannot compound forever. p≈0.5 balances
"enough memory-only pressure" against "enough re-anchoring," and matches inference better than
all-hidden (open question 2: inference HAS latents present near-clean). Concrete scheme: i.i.d.
Bernoulli(p_hide) per step; on visible steps still take the memory-sufficiency loss too (memory must
agree with the visible latent) so visible steps aren't free of memory pressure. **Treat p_hide as a
tuned knob (0.3–0.7) gated on the deep-k recall**, not in-window loss.

### (c) Gradient-truncation horizon (the ESC-014 min-BPTT-depth question)
**Do NOT hard-code 4·N. Measure the minimum k first; it is one cheap synthetic sweep (§5/P1).**
Reasoning: (i) the doc's 4·N is unmotivated and, if the relay is a contraction once trained, a much
smaller k extrapolates (BPTT's whole point); if it is NOT a contraction, no finite k extrapolates and
we need a different anchor (see below) — either way 4·N is a guess. (ii) **The dynamic-state caveat
matters**: V-T014's k was measured on a static secret; position needs the relay to *apply the bounce
transition* each hop, which may require deeper k OR may actually be EASIER to anchor because each hop
has an action-conditioned, deterministically-correct target (no consistency-fixed-point freedom — see
butterfly note). So measure k on a DYNAMIC secret. **Procedure:** sweep tbptt-k ∈ {2,4,8,16} on the
dynamic-secret harness, eval extrapolation to 4–6× train depth, pick the smallest k whose deep-hop
recovery stays flat (matches full BPTT within, say, 1.5×). My prior: **k≈4–8** will suffice for the
deterministic transition; commit to that band and confirm with the sweep before the GridWorld build.

### The "butterfly effect" / over-punishing valid-but-wrong guesses
**In GridWorld this is largely a NON-issue, and that is a real asset — say so explicitly.** GridWorld
is *deterministic given actions* (`src/envs/gridworld.py`: square steps 1 cell/tick under the action,
wall-reflects). With teacher-forced GT context + known actions, the next state is a *deterministic
function* of (prev state, action) — there is no random branch, so penalising the full downstream
rollout is CORRECT signal, not noise: a wrong prediction is genuinely a memory/dynamics error, never a
valid alternative future. The doc's "don't over-punish butterfly" guardrail is therefore **belt-and-
suspenders here** (keep it cheap; don't spend design effort on credit-curving for GridWorld). It only
becomes load-bearing in a *stochastic* env, where the right construction is: take the loss only where
the context *determines* the answer (the deterministic part of the state) and re-anchor frequently —
exactly what teacher-forcing + per-step re-anchor already buys. **Recommendation: ship the simple
full-downstream loss for GridWorld; defer butterfly-credit machinery to the stochastic-env future.**

---

## 5. The single most informative cheap experiment to run BEFORE building (P1)

**Extend `experiments/verify-T014/probe_detached_relay_v2.py` to a DYNAMIC secret + a tbptt-k sweep.**
This settles the two unknowns that gate the whole build — (i) does a trained relay extrapolate for
DYNAMIC state at all (the real GridWorld question the static probe never answered), and (ii) the
minimum k — in ~30–60 min, no GridWorld training, reusing the validated harness and its four arms.

Design (script provided: `experiments/EXP-029-design/probe_dynamic_relay.py`):
- **Dynamic secret:** replace the static `s` with a 2-D position + velocity that integrates each hop
  with reflecting walls (a 1-D/2-D bounce), and supply the per-hop "action" (or make it deterministic)
  as input — the reader must output the CURRENT position, so the relay must apply the transition each
  hop, not just preserve a code. Secret observed at hop 0 only, then hidden.
- **Arms:** no_relay (floor), tbptt-{1,2,4,8,16}, full bptt (ceiling). Train depth 32, eval to 200.
- **Reads:**
  - If **even full BPTT fails** on the dynamic secret → the deficit is capacity/representability of
    the relay (M=4 too small, §FF9 future caveat), NOT credit — that *reorders everything*: pivot to
    widening memory before any rollout loss. (High-value negative result.)
  - If **BPTT works and tbptt-k matches it for k≥k\*** → k\* is the number for C1's truncation; build
    C1 at k\*. Prediction: k\* ∈ {4,8}.
  - If **tbptt-k never matches BPTT for any finite k** → the relay is not a contraction; need a
    content anchor every hop (which the GridWorld action+GT-latent re-anchoring provides) → favors
    high p_visible and frequent re-anchor over deep BPTT.

This probe is strictly more informative than starting the GridWorld rollout loop, because it
discriminates capacity-vs-credit AND fixes k, and a wrong answer on either would waste the big run.

**Logging:** append EXP-029 row to `EXPERIMENTS.md` (id, date, question, hypothesis, command, seed,
outcome); artifacts + script under `experiments/EXP-029-design/`. Fixed SEED=0 reported. Note
explicitly: a passing synthetic relay proves the *credit mechanism* can carry dynamic state, NOT the
production hop count on the real tokenizer latent (M=4 capacity is the separate, downstream unknown).

---

## 6. What would change this ranking
- **P1 shows full BPTT fails on the dynamic secret** → C1/C2/C3 all premature; widen memory (M>4) or
  add a transition-friendly memory parameterization first. (Capacity, link 1, not credit.)
- **P1 shows tbptt-2 already extrapolates** → C3 (the cheap patch) becomes the recommendation; skip
  the rollout loop.
- **C3 run alone already bends the EXP-028 curve** → one trained hop was the missing dose; stop.
- **In-window recall (k≤8) or base val-diffusion regresses** under any option → the unrolled branch is
  fighting operations 1+2; raise the warmup, lower the rollout fraction, or lower p_hide.
- **EXP-028's corrected inference fails critical-claim-verifier / 2nd seed** (ESC-020 Q2 still open)
  → the whole premise (FF9 carries dynamic position in-window) is shakier and the diagnosis target
  shifts; settle ESC-020 before committing the big run.

## 7. Open questions / undetermined
- **M=4 capacity vs full-scene memory-only imagination** — not resolved here; P1's BPTT-ceiling arm
  is the cheap first read on whether M=4 can hold integrated position; the pixel-detail north-star
  (memory-only imagination) is a separate, larger capacity question (doc §future).
- **Exact warmup curve (0→50% by wall-clock)** — I endorse the shape (contain-then-propagate) but the
  rate is a tuning knob gated on deep-k recall; not derivable a priori.
- **Whether newest-only flow + multi-frame sufficiency is the right split under per-step hiding** —
  leant yes (§4a/b); confirm cheaply by an ablation once C1 is wired (newest-only vs all-frame on one
  short run), not before.
