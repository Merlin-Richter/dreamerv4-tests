# Vanilla honest-baseline A/B: step-size curriculum vs τ0-anchor (Merlin, 2026-07-04)

Follow-up to `tasks/done/diagnose-vanilla-inwindow-position-failure.md`. Merlin ordered two
parallel GridWorld experiments to fix the vanilla baseline's in-window position failure, both
vanilla (n_memory=0), both changing only the training τ/d allocation via `--model-module`:

- **Arm C (Merlin's spec):** step-size curriculum — d_min only for the first 33% of training,
  unlock the rest gradually and evenly until 66%, all d unlocked after.
- **Arm D (agent's design, free rein):** sustained (τ_idx=0, d=d_min, GT flow) point mass at
  p=0.5 per frame — the mem2mem noise-mode pressure transplanted into plain diffusion forcing.

**Done means:** both 50ep ferranti runs complete, ckpts pulled, teacher-forced probe + recall +
sheets run, verdict vs the pre-registered predictions written to
`experiments/vanilla-honest-baseline/NOTES.md` + EXPERIMENTS.md.

Design/provenance/predictions: `experiments/vanilla-honest-baseline/NOTES.md`.

## Result (2026-07-04)
Both arms completed (415190/415191 @ fae4e8b) and evaluated. PREDICTIONS CONFIRMED: Arm D (tau0-anchor) teacher-forced 1-step ~1.0 (old vanilla 0.09), free-run flat 0.98-1.0, recall w8 perfect in-window + chance past eviction = the honest no-memory baseline; Arm C (curriculum) marginal (<=0.25). Sustained tau0-GT pressure is the active ingredient. Full numbers: experiments/vanilla-honest-baseline/NOTES.md.
