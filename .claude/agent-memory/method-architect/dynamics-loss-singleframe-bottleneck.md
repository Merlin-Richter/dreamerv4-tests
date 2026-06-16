---
name: dynamics-loss-singleframe-bottleneck
description: The D_dynamics_model shortcut-forcing loss never trains time-axis multi-step prediction; this is the recurring identifiability cause of weak motion/rollout
metadata:
  type: project
---

`DynamicsModel.loss()` in `src/D_dynamics_model/dynamics_model.py` (~line 399) is a pile of
**independent single-frame denoisers**: each frame t is noised at its own sampled tau and the
target is its own clean latent z1[t]. The bootstrap term (~:440-455) distills along the **tau
(noise) axis** (two d/2 denoising steps), NOT the **time axis**. There is **no term that asks
the model to predict frame t+h from a clean frame t.** The only successor-prediction signal in
the whole file is the aux losses `_ff7_loss` (~:480) and `_ff9_loss` (~:527), which compare
`z_hat[:,1:]` to `zw[:,1:]`.

**Why this matters (load-bearing):** EXP-014 proved vanilla gets ~4.5px single-step motion
(worse than copy-last 3.2px) while FF7/FF9 get ~1px — and the FF7 gain comes via the **plain
windowed path, no relay**, i.e. the successor-prediction *loss itself* installs the motion
model. So "learned X (denoise present) not Y (propagate motion)" is an **identifiability /
link-3** failure: the objective is minimized without modelling motion. The ramp w=0.9tau+0.1
up-weights the high-signal region where the frame's own noised latent already determines it,
favoring a "clean the visible blob" shortcut (V-T013 showed memory/context inert at high tau).

**Also:** training context frames carry random per-frame tau; rollout pins context at
`context_signal=0.9` (`_denoise_next` ~:603). That tau-distribution mismatch is a separate
cheap-to-test multiplier (link 4b).

**Why:** localizing a "model won't do multi-step Y" complaint here saves re-deriving it. The
fix family is objective (add a time-axis multi-step/self-context term), NOT architecture —
representation is fine (linear probe R²=0.96 for position, EXP-011).

**How to apply:** when asked about weak rollout/motion/dynamics in D, read `loss()` first and
check whether the proposed target appears as a *time-axis* term anywhere. If not, that's the
gap. Additive fixes must be config-gated identity-when-off (mirror `n_memory=0`/`ff7_k=0`/
`ff9_k=0` guards). See [[probes-dynamics-motion-memory]] and [[h3-line-failures]].
