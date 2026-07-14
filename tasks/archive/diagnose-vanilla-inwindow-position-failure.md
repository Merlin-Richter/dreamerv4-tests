# Diagnose: vanilla can't predict square positions even in-window (no occlusion)

**Source:** direct instruction from Merlin, 2026-07-04. Observation: `outputs/sheets/sheet_normal.png`
(vanilla, 4 ctx + 12 free-run, curtain up) gets colors right but square positions wrong, while
`outputs/sheets/mem2mem_w8/sheet_occlusion.png` (mem2mem, occluded!) tracks the square. Same
compute. The repo's goal is showing memory tokens preserve info across sliding windows — but
vanilla failing IN-window is unrelated to memory and "just bad". Why?

**Done means:** a verified mechanistic answer, written down.

## Result (2026-07-04)

Answered, evidence in `experiments/vanilla-inwindow-diagnosis/` (NOTES.md + 2 probes + JSONs):
vanilla never LEARNED the transition map — teacher-forced 1-step from all-real revealed context is
~chance at every context length while ff9/mem2mem/no-ff9 controls are ~1.0; architecture is
innocent (ff9 plain forward, no carried memory, 1.0 @ tau=0). Root cause is training-signal
allocation in diffusion forcing: only (GT flow target AND tau<=0.1) forces position-from-context
= 1.3% of frames = 0.4% of ramp-weighted loss; the other tau=0 frames get bootstrap self-distill
targets; ~75% of frames let the model read position from its own noisy latent (denoise shortcut).
FF9/mem2mem terms are ~100% tau=0-vs-GT, which is why they learn it. Flagged: this confounds the
vanilla-vs-memory comparison (incl. the memmaze vanilla arm 415103); honest-baseline fix options
in NOTES §Implications — spec edit needed, Merlin decides.
