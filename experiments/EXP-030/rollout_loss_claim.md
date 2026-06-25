# Claim for independent verification — `_ff9_rollout_loss` (FF9 rollout-training, D-048)

Target code: `src/models/dynamics_model.py` — `DynamicsModel._ff9_rollout_loss` and its wiring in
`DynamicsModel.loss` (the `ff9_rollout_h>0` branch); the clip-windowing at the top of `loss`;
`encode_frames` chunking in `src/training/train_dynamics.py`. Design rationale:
`experiments/EXP-029-design/method_architect.md` (C1). Self-tests: `src/tests/test_ff9_rollout.py`.

## Background (what it is supposed to do)
FF9 v2 memory tokens (a distinct token type, carried per-frame; temporal attention is position-wise so
each memory slot is its own causal channel) contain hidden state in-window, but the FF9 sufficiency
loss `_ff9_loss` fills the intermediate frames of its mini-window with the **learned-init placeholder**
(`self.memory_tokens.expand(...)`), so the map "write memory_{t+1} from memory_t" (operation 3) is on
**no gradient path**. `_ff9_rollout_loss` is meant to fix exactly that: seed a memory token from a real
near-clean prefix, then roll `h` differentiable hops where the WRITTEN memory at each hop (the
final-layer memory activation, `forward(..., return_memory=True)`) is carried (injected at the source
of a 2-frame `[source|new]` window) into the next hop, with the autograd graph kept for `tbptt_k` hops.
Per step, source latents are HIDDEN (tau=0 pure noise => memory is the only carrier) on a contiguous
tail (`hide_mode="tail"`) or i.i.d. Bernoulli; the new frame's latent slot is pure noise (tau=0) and is
the x-prediction flow target. GT context is teacher-forced (the source on a VISIBLE step is the true
previous frame held near-clean).

## CLAIM (falsifiable) — the implementation is correct in these specific senses
**C1 (relay on the gradient path).** The loss backprops through the chain of memory writes:
`mem_carry_{j} = (written memory at hop j-1)` depends on `mem_carry_{j-1}` through the model, and the
hop-j loss depends on `mem_carry_j`. So gradient from a later hop's loss reaches the construction of an
earlier memory token, for up to `tbptt_k` hops, and is cut beyond it by the `.detach()` every
`tbptt_k` hops. (Not merely that *some* gradient reaches the memory tokens, but that it flows *through
the carried-memory chain* and that `tbptt_k` controls the depth.)

**C2 (no teacher-forcing leak).** On a HIDDEN step the source latent is pure noise, so the new-frame
prediction cannot be solved by copying a visible latent — memory is genuinely the only scene carrier.
The loss is not silently satisfiable by a latent shortcut on hidden steps. (Check the `torch.where`
hide logic and tau assignment: hidden => source replaced by noise AND tau_src=0.)

**C3 (identity when off).** `ff9_rollout_h=0` adds no term and draws no RNG => byte-identical to a
pre-D-048 model. And the `loss()` clip-windowing (`z_full, z1 = z1, z1[:, :W]`) is byte-identical when
the clip length T == the model window W (the usual case), only diverging when a longer clip is fed
deliberately (deep variant).

**C4 (train/inference consistency).** The rollout's per-hop structure (2-frame `[source|new]`, memory
injected at source, learned-init at new, source hidden=noise or near-clean, new=pure-noise target,
carry = the new frame's written memory) is the SAME computation an updating-memory inference must run,
so a model trained by this loss is exercised by such an inference (this inference is being built
separately; verify the training side is self-consistent and matches `full_state_rollout_step`'s op
semantics except for the memory UPDATE).

## What would FALSIFY each
- C1 false if: the carried memory is silently detached every hop (no multi-hop credit), or `tbptt_k`
  does not actually bound the graph depth, or the "written memory" taken as the carry is not on the
  path from the loss.
- C2 false if: on a hidden step some real-latent information still reaches the prediction (e.g. the new
  frame's noised target leaks GT at the tau used, or the source isn't actually zeroed).
- C3 false if: enabling the args changes the off-path RNG stream or the windowing changes T==W results.
- C4 false if: the training rollout's op semantics (which frame the memory is injected at, which frame
  is the target, the action alignment src=prev/new=current) cannot be mirrored at inference.

## Notes for the verifier
- The `_ff9_rollout_loss` tau for the new (target) frame is **0 (pure noise)** — a pure flow/
  x-prediction target — to maximise the memory signal (V-T013 showed high-tau targets make memory
  inert). Confirm this is what the code does and reason about whether it is the right call.
- The deterministic GridWorld (square steps 1 cell/tick, wall-reflect, under known curtain actions)
  means a wrong prediction is always a genuine error (no valid-but-wrong branch) — so the
  full-downstream loss is correct signal (design note 4). You may take this as given (env fact).
- An empirical probe is welcome (e.g. confirm the seed-write gradient grows with tbptt depth on a
  *trained* tiny model, or that hidden-step prediction is at chance for an untrained model). Log any
  probe under `experiments/EXP-030/` and add an EXPERIMENTS row if you run one.
