# GridWorldV2 arms campaign: vanilla / dense mem2mem / SPARSE write-slots (overnight, autonomous)

Ordered by Merlin 2026-07-04 (~22:00): "work on this branch and train vanilla, dense mem2mem and
sparse mem2mem"; ~10h autonomous window granted while he sleeps. Design: v3 write-slots
(`tasks/drafts/sparse-memory-write-slots.md`) with Merlin's confirmed role encoding — memory init
params become (2, n_memory, dim): one init for WRITE slots, one for SCRATCH slots. TBPTT stays
~2x window; old write K/V beyond the horizon are read-only (read path trains, writes don't).

## Arms (all GridWorldV2, frozen v1 tokenizer `checkpoints/gridworld/tokenizer.pt`, 50ep, seed 0)

| arm | recipe | ckpt |
|---|---|---|
| A vanilla-τ0 | train_dynamics + DynamicsModelTau0Anchor, bs256 | checkpoints/gridworldv2/dynamics_vanilla_tau0.pt |
| B dense m2m +FF9 | train_mem2mem --mem2mem-frac 1.0 --no-bootstrap --ff9 3 --n-memory 4 bs64 clip64 (v1 winner recipe) | checkpoints/gridworldv2/dynamics_m2m_ff9.pt |
| C dense m2m no-FF9 | = B minus FF9 (--no-ff9) | checkpoints/gridworldv2/dynamics_m2m_noff9.pt |
| D sparse write-slots n=8 | NEW experiment code (this campaign's build), no-FF9, W=16 (>= 2n), bs64 clip64 | checkpoints/gridworldv2/dynamics_sparse_n8.pt |

**Agent decision (reversible, flagged):** arm C added beyond Merlin's three because D cannot use
FF9 cleanly (FF9 assumes a write every frame) — C is D's matched control; it also replicates the
"FF9 not necessary" finding on the new env (the v2 scrutiny flagged that 415143 doesn't gate this).

## Pipeline

1. Datagen job on ferranti: 5000 eps x 200 -> data/gridworldv2.npy + latent-cache build
   (train_dynamics --build-latent-cache-only) so the arms don't race.
2. Launch A/B/C as soon as datagen lands (B/C trainer compat checked first).
3. Build D locally meanwhile: experiments/sparse-write-slots/ — SparseMemAttention (slot-dependent
   temporal mask: memory slots attend only to write-slot positions, pos % n == 0, no diagonal for
   scratch), DynamicsModelSparseWS ((2,n_memory,E) init, phase-aware memory_in interception so
   rollout_init/rollout_step/recall work unchanged), rollout_sparse.py trainer variant (carry =
   write-slot memories only; phase from absolute positions; W=16 half=8 n=8 -> slide-aligned).
   Smokes: mask correctness (scratch rows never attended), causality, n=1 (all-write) ==
   dense-equivalent sanity, cache-equivalence, 2ep local train. Then launch D.
4. Overnight: process memmaze landings (415205 vanilla-tau0 ~02:40; 415104 dense m2m ~05:00):
   pull, verify, sheets. 415143 lands after the window (~13:00) — leave a watcher.
5. Morning: pull v2 arms, recallv2 native W16 max_k 64 + w8 variant, compare plot (incl. k mod n
   decomposition for D), NOTES/EXPERIMENTS/ORIENT, full report for Merlin.

## Provenance
- (jobs recorded below as launched)
