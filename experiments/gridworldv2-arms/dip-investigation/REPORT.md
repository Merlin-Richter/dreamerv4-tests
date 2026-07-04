# D_sparse_n8 k=4-6 recall dip — root-cause report

**Date:** 2026-07-05. Independent probe investigation (all scripts + JSONs in this directory).
Checkpoints probed: `checkpoints/gridworldv2/dynamics_sparse_n8.pt` (D, `DynamicsModelSparseWS`),
`checkpoints/gridworldv2/dynamics_vanilla_tau0.pt` (A, control), tokenizer
`checkpoints/gridworld/tokenizer.pt`. 256 rollouts per condition (±0.03 binomial at p≈0.5),
branch at EVERY k, oracle self-test = 1.000 in every run, and every analyzed rollout's recorded
trajectory was re-simulated from its seed and matched exactly (driver self-check).

## Verdict

**Not H2, H3, or H4. H1 in a sharpened, structural form:** the dip is a deterministic
**train/eval mismatch of the write-aligned sliding-window training scheme**
(`experiments/sparse-write-slots/rollout_sparse.py`), not an eval bug and not a corrupted write.

Behavioral law (established causally): **at branch positions after a memory-write slot, the
model's dominant belief is the true trajectory with the write-slot frame's own action
DELETED.** The write itself is fine — it integrates its own action token correctly. The wrong,
one-move-behind belief is produced by the **non-memory pathway** (latent-anchor + action-token
integration through register/latent channels), which learned during training to *never apply the
write-phase frame's action on top of an anchor located before the write phase* — because in
training such a configuration cannot exist: window starts are always write-aligned
(`s % 8 == 0`), so the temporal history of every non-memory channel is severed exactly at write
phases, and a write-phase frame is always either the window-initial frame (whose own action's
effect is already baked into its visible content, so its token must NOT be re-applied) or the
in-pass fresh write. The incremental KV-cache rollout at eval hands the model a *continuous*
history across write phases; anchors then lie before the write slot, the learned
"start integrating after the boundary" program fires, and the write-frame's move is dropped.
Near the write this lagged belief outweighs the correct write content -> accuracy below the
memoryless vanilla. The apparent "recovery" by k=10 is (w8) the anchor frames evicting +
(both windows) wall-clamping coalescence of the one-move-deleted trajectory — at w16 the model
still follows the deleted-move trajectory at 77-88% through k=11 whenever the two disagree.

## Evidence (probe -> result)

### 1. H4 (noise) — dead
256-rollout fine grid reproduces the dip tightly.
Exact-position acc (`d_w8.json`, `d_w16.json`, `a_w8.json`, `a_w16.json`):

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|----|----|----|----|----|
| D w8  | .980 | .945 | **.652** | .680 | .734 | .754 | .809 | .867 | .941 | .762 | .469 | .531 | .559 |
| D w16 | .988 | .996 | **.613** | .629 | .648 | .695 | .727 | .738 | .723 | .746 | .441 | .477 | .500 |
| A w8  | .980 | .969 | .973 | .957 | .297 | .293 | .246 | .238 | .215 | .215 | .152 | .188 | .172 |
| A w16 | .996 | .961 | .938 | .934 | .922 | .871 | .859 | .836 | .816 | .773 | .789 | .359 | .305 |

D sits 0.30+ below vanilla exactly at k=4-6 while revealed context is in-window. D w8 color acc
is 1.0 through k=10, then collapses to ~0.44-0.50 from k=11 (w16 keeps color 1.0 — the w8 relay
chain is broken by the forced window, see footnote 1).

### 2. H3 (eval-driver / action off-by-one) — dead
* Fully-revealed teacher-forced probe (`a_nohide_tf_w8.json`): **A scores 0.977-1.000 at every
  k** through this exact driver -> action/truth alignment, branch semantics, decode path all
  correct. Oracle 1.000 everywhere; seed->trajectory re-simulation matched in all runs.
* D's k=1-3 (before any in-rollout write exists) = 0.945-1.000 with zero off-by-one signature.
* But: **D dips even fully revealed, teacher-forced** (`d_nohide_tf_w8.json`: 0.750/0.812 at
  k=5/6 = branches 1-2 after position 8), overriding the *visible* true frame at distance 1.
  The dip is a property of the model at write-adjacent positions, not of occlusion or the eval.

### 3. H2 (corrupted first in-rollout write) — dead
`mem_only` branch ablation (memory channel only; register/latent/action history masked,
`d_w8_memonly.json`): the branch renders the **write-time position p4 at 0.68-0.74, flat for
k=4..10** (while acc vs the moving truth decays 0.72->0.19, exactly a frozen-but-correct belief).
The write@8 content is the correct post-action belief. Spoof confirmation: feeding A_STAY as the
model's action token at the write commit (env still moves; `d_w8_memonly_spoofwrite.json`) flips
the write content to p3 (del-o4 match 0.570 vs 0.099 truth) -> **the write reads and correctly
integrates its own action token.**

### 4. The deleted-move signature (analyze2.py, discriminative subsets only)
`del_m` = true trajectory with occluded move m deleted; scored only where it disagrees with truth.

* `d_w16.json`: P(pred==del4 | del4!=truth) = **0.810 / 0.871 / 0.865 / 0.882 / 0.851 / 0.797 /
  0.800 / 0.766** for k=4..11 (truth: 0.09-0.19). The belief IS the o4-deleted trajectory the
  whole time the pre-write anchors are in-window; raw-acc "recovery" is clamping coalescence.
* `d_w8.json`: same signature, decaying with anchor eviction: 0.603/0.545/0.348/0.250/... —
  drops sharply at k=6, exactly when the last revealed frame (pos 3) leaves the w8 cache.
* At the **next** write the deletion repeats: k=12 -> del12 matched 0.647 (w16).
* `no_mem_read` ablation (write keys masked, `d_w8_nomem.json`): dip and del4 signature
  **persist** (k=4: 0.605, del4 0.636) -> the lagged belief is carried by the non-memory
  channels; normal branch (0.652) sits between no_mem_read (0.605) and mem_only (0.715) —
  fusion near the write trusts the wrong channel.

### 5. Phase-shift probe — dip is locked to write slots, not to the hide tick
`n_ctx=8` puts the hide tick ON write slot 8 (its "move" is a no-op) and the first occluded
write at 16 (committed at k=8). Result (`d_nctx8_w16.json`): k=1..7 = 0.86-0.99 (**no dip at
k=4-6**); dip appears at k=8 with del8 dominance **0.737 vs truth 0.143**, persisting 0.61-0.69
through k=15; at k=16 the next boundary takes over (del16 = 0.508). A write whose own action is
a no-op produces no deleted-move dip — the effect scales with the write-frame action's
displacement, as the mechanism predicts.

### 6. Spoof causality (action-token surgery at eval)
* `at_write` spoof (model told A_STAY at write commits; env unchanged, `d_w16_spoofwrite.json`):
  branch belief at k=4 becomes p3 at **0.992** — unchanged trajectory-class vs normal (0.906),
  i.e. consistent with the write-phase move being absent from the dominant belief either way.
* `after_write` control (spoof position 9 instead, `d_w16_spoofafter.json`): beliefs shift to
  the **double-deletion del(4,5) trajectory at 0.810-0.867** (truth 0.05-0.08) -> non-write
  action tokens ARE faithfully integrated; only the write-phase token is dropped from the
  non-memory belief.

## Mechanism -> root cause location

`experiments/sparse-write-slots/rollout_sparse.py` trains ONLY on write-aligned windows:
`s` starts at `half` (=8) and advances by `half`, with `assert s % n == 0`; the carried write is
always injected at window index 0 (`build_mem_in`: `out[:, 0:1] = carried`). Consequences:

1. Non-memory temporal channels (latents/registers/action) never span a write phase: at every
   position = 0 (mod 8) their history restarts (window-initial frame). No gradient ever teaches
   "anchor before a write slot + integrate the write slot's action across it".
2. In training, the window-initial (write-phase) frame's own content already contains its
   action's effect, so the *correct* trained program is to NOT re-apply that frame's action
   token — the program the eval branch then wrongly executes on a continuous cache.
3. Readers see injected-style write K/V only at relative distances 8-15 (old-half write);
   distances 1-7 only ever serve in-pass fresh writes. The eval cache serves injected-style
   K/V at all distances (also visible as the w16 `mem_only` short-distance degradation).

The eval (`src/evals/gridworldv2/recall.py`, `src/models/dynamics_model.py` rollout path) is
correct: it is the training scheme that never visits the eval's data distribution around write
slots. **No code fix in `src/` is warranted; do not "fix" the eval.**

## Training-side fix (recommended) + confirm experiment

**Randomize the window phase in `sparse_rollout_loss`:** keep write slots at absolute
`p % 8 == 0` (the mask is already absolute-position-keyed) but start windows at
`s = r (mod 8)` with a random per-batch offset `r in {0..7}` (slides still advance by 8; inject
the carried write set at window index `(8 - r) % 8`, i.e. the unique write position inside the
old half; carry out the new half's write slot). This exposes, with loss, (a) anchors BEFORE a
write phase with continuous history across it — putting gradient directly on the deleted-move
program — and (b) injected-style write K/V at all relative distances 1..15. A stronger (dearer)
variant: train a fraction of steps through the actual incremental commit path
(`rollout_init`/`rollout_step`).

**Confirm experiment** (one ~35-min job, same protocol as arm D): retrain with random window
phase, rerun this battery. Predictions if the diagnosis is right: k=4-6 >= vanilla (~0.93+) at
both windows; del4 discriminative share ~0 (from 0.81-0.88); k=2 / k=10 peaks unchanged
(~0.94-0.98); teacher-forced no-hide dip at k=5-6 gone; likely improvement of the k>=12 relay
plateau too (relayed writes read boundary-adjacent context as well).

## Footnotes (real but not the dip's cause)

1. **w8 violates the model's own invariant**: `DynamicsModelSparseWS` asserts
   `max_temporal_length >= 2*n_sparse` ("a write always sees the previous write"), but the
   eval's `--window 8` (`max_ctx=7`) silently bypasses it: write@16 is committed with cache
   = positions 9..15 — no previous write visible. This breaks the write->write relay at w8 for
   k>=12 (visible as the w8-only color collapse from k=11). Affects the plateau, not the dip.
2. **Untrained d-embedding at eval**: `train_sparse.py` trains d_min only (`n_d_unlocked=1`,
   no bootstrap), but `rollout_step` conditions every pass on `d_idx = K.bit_length()-1 = 2`
   (K=4) — an untrained embedding input. Uniform across k (k=2/k=10 are fine), so not the dip;
   arm A trained all-d, so this is also an arm asymmetry worth cleaning up.
3. **Registers are an unrestricted memory side-channel**: the sparse mask constrains only the
   memory slots' queries; register/latent/action slots relay causally at every frame. E.g. D w8
   k=11 scores 0.762 with NO write key in the cache at all. The "only writes carry state"
   interpretation of arm D is not enforced by the architecture.

## File inventory

* `common.py` — instrumented batched driver (fine k-grid, trajectory recording, no-hide /
  teacher-forced modes, branch-only sparse-mask ablations, action-token spoof), loaders.
* `run.py` — CLI runner; `run_all.sh` — the battery; `run_all.log`, `spoof.log` — full tables.
* `analyze.py` — error/lag classification; `analyze2.py` — deleted-move hypothesis test with
  seed-exact stream re-simulation (self-checks the driver per record).
* JSONs: `{a,d}_w{8,16}.json` (main, 256), `d_*_nomem/memonly` (ablations),
  `{a,d}_nohide_tf_*`, `d_nohide_free_w8` (driver validation), `d_nctx8_w{8,16}` (phase shift),
  `d_w16_spoof{write,after}`, `d_w{8,16}_memonly_spoofwrite` (token-surgery causality).
