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
- prep: 415220 FAILED (bash -c quoting; fixed via prep.sh) -> **415221 rc=0** (5000 eps + latent cache, 2 min) @ b3db367.
- **A 415222** gwv2-vanilla-tau0 @ b3db367 (startup verified: Tau0Anchor, n_actions=7).
- **B 415223** gwv2-m2m-ff9 @ b3db367 (verified: frac 1.0, no-bootstrap, use_ff9=True).
- **C 415224** gwv2-m2m-noff9 @ b3db367 (verified: use_ff9=False).
- **D 415226** gwv2-sparse-n8 @ a85eed7 (impl: experiments/sparse-write-slots/, smoke 6/6 PASS
  incl. zero attention leak on non-write keys + grads to both inits; 2ep local train green).
- Eval runner ready: experiments/gridworldv2-arms/eval_all.py (4 arms x w16/w8, staleness view for D).

## Result (2026-07-05, overnight autonomous window)
DONE (gwv2 portion; memmaze continues under its own tasks). 7 arms trained+evaled (incl. mid-
campaign root-cause->fix->retrain cycle on the sparse write-aligned-window artifact, found by an
independent dip-investigation agent with causal probes). HEADLINE (256 rollouts, w16): dense
mem2mem 1.00 FLAT to k=64 (lossless, FF9==noFF9); sparse write-slots best arm (m16+phase-fix)
0.69-0.73 flat (m4-fix ~0.58; bugged 0.50); NEW B1 exact-Bayes no-memory floor 0.45-0.61 — v2
memory claims must clear B1, not chance. Bottleneck localized to the write-update op (storage/
reach lossless over 7+ relays). Design findings: registers = unrestricted memory side-channel;
w8 eval violates sparse W>=2n. Full record: experiments/gridworldv2-arms/{NOTES.md,NIGHTLOG.md,
compare_w16_r256.png,dip-investigation/REPORT.md}.
