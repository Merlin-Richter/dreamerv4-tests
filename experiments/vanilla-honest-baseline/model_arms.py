"""Two vanilla (n_memory=0) training-objective arms for the honest-baseline A/B.

Background (experiments/vanilla-inwindow-diagnosis/): vanilla diffusion forcing starves the
"predict-from-context" signal — (GT flow target AND tau<=0.1) is ~1.3% of frames, ramp-weighted
~0.4% of loss — so the model learns denoising, never the transition map. These arms change ONLY
`sample_tau_d` (training-time tau/d allocation); the loss formula, ramp, bootstrap, model params
and checkpoint format are untouched, so all eval tooling (recall, sheets, probes) works unchanged
with the base DynamicsModel class.

Both arms sample the DEFAULT distribution in eval mode, so val/loss stays comparable across arms
and to the original vanilla baseline.

Arm C — DynamicsModelDCurriculum (Merlin's design, 2026-07-04):
    step-size curriculum on training progress: d_min only for the first 33% of epochs, then unlock
    the remaining steps gradually and evenly until 66%, from where every d is unlocked. Progress is
    tracked by counting eval->train mode transitions of loss() (the trainer runs one val pass per
    epoch); total epochs from $CURR_TOTAL_EPOCHS (default 50 — set it if you change --epochs!).

Arm D — DynamicsModelTau0Anchor (agent's design):
    with prob P_ANCHOR=0.5 per frame, force (tau_idx=0, d_idx=d_min): the frame's own latent is
    pure noise and the loss is the ground-truth flow term — i.e. plain next-frame prediction from
    context, sustained for the WHOLE run. This transplants exactly the pressure that provably
    teaches the map in the mem2mem trainer (its noise mode: 50% of new-half frames at tau=0 vs GT,
    ramp applied — see mem2mem-rollout-noff9-fair: that pressure alone reaches recall ~1.0) into
    plain diffusion forcing, minus memory tokens, minus rollout training. One knob, magnitude
    matched to a validated recipe. Weighted-loss share of the anchored slice ~= 0.5*w(0) /
    (0.5*w(0) + 0.5*E[w]) ~= 18% (vs 0.4% in default vanilla).

Usage (train_dynamics.py):
    --model-module experiments/vanilla-honest-baseline/model_arms.py:DynamicsModelDCurriculum
    --model-module experiments/vanilla-honest-baseline/model_arms.py:DynamicsModelTau0Anchor
"""
from __future__ import annotations

import math
import os

import torch

from models.dynamics_model import DynamicsModel


class DynamicsModelDCurriculum(DynamicsModel):
    """Arm C: finest-first step-size curriculum (d_min-only 0-33%, even unlock 33-66%, full after)."""

    def __init__(self, config):
        super().__init__(config)
        self._total_epochs = int(os.environ.get("CURR_TOTAL_EPOCHS", "50"))
        self._epoch = 0          # completed-epoch counter (0 during the first epoch)
        self._saw_eval = False   # set by a val pass; the next train call increments _epoch
        self._last_k = None
        print(f"[DCurriculum] total_epochs={self._total_epochs} "
              f"(schedule: k=1 to 33%, even unlock to 66%, then k=n_d={self.n_d})")

    def _n_unlocked(self) -> int:
        frac = self._epoch / max(1, self._total_epochs)
        if frac < 1.0 / 3.0:
            return 1
        if frac >= 2.0 / 3.0:
            return self.n_d
        # middle third: unlock the remaining n_d-1 steps evenly
        return 1 + min(self.n_d - 1, math.ceil((frac - 1.0 / 3.0) * 3.0 * (self.n_d - 1)))

    def sample_tau_d(self, B: int, T: int, device):
        if not self.training:              # val pass: default distribution (comparable val/loss)
            self._saw_eval = True
            return super().sample_tau_d(B, T, device)
        if self._saw_eval:                 # first train batch after a val pass -> new epoch
            self._epoch += 1
            self._saw_eval = False
        k = self._n_unlocked()
        if k != self._last_k:
            print(f"[DCurriculum] epoch {self._epoch}/{self._total_epochs} -> n_d_unlocked={k}")
            self._last_k = k
        # d ~ U over the k FINEST steps; tau ~ U on the grid implied by d (base formula, restricted).
        off = torch.randint(0, k, (B, T), device=device)
        d_idx = (self.n_d - 1) - off
        K = torch.pow(2, d_idx)
        step = torch.minimum((torch.rand((B, T), device=device) * K).long(), K - 1)
        tau_idx = step * torch.pow(2, self.n_d - 1 - d_idx)
        return tau_idx, d_idx


class DynamicsModelTau0Anchor(DynamicsModel):
    """Arm D: sustained (tau=0, d_min, GT-flow) point mass at P_ANCHOR per frame (train only)."""

    P_ANCHOR = 0.5

    def __init__(self, config):
        super().__init__(config)
        print(f"[Tau0Anchor] P_ANCHOR={self.P_ANCHOR} -> forced (tau_idx=0, d_idx={self.n_d - 1})")

    def sample_tau_d(self, B: int, T: int, device):
        tau_idx, d_idx = super().sample_tau_d(B, T, device)
        if not self.training:              # val pass: default distribution (comparable val/loss)
            return tau_idx, d_idx
        anchor = torch.rand((B, T), device=device) < self.P_ANCHOR
        tau_idx = torch.where(anchor, torch.zeros_like(tau_idx), tau_idx)
        d_idx = torch.where(anchor, torch.full_like(d_idx, self.n_d - 1), d_idx)
        return tau_idx, d_idx
