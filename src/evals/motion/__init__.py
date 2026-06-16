"""Motion-prediction eval (working, non-frozen).

`motion.py` holds the curve primitives (open-loop compounding, teacher-forced per-step map,
tau-context sweep). `MotionEval` exposes the headline scalars through the common interface;
rich curve charts come from `evals.rollout_view`.
"""
from __future__ import annotations

from evals.base import Eval, EvalConfig, register


class MotionEval(Eval):
    name = "motion"
    frozen = False
    compatible_envs = ("occluded_bouncing",)

    def _episodes(self, cfg: EvalConfig):
        from evals.probe_env import make_probe_episode
        # Curtain-up throughout (k=0): pure motion-tracking, no occlusion.
        return [make_probe_episode(seed=20000 + i, P=cfg.prefix_P, k=0, R=cfg.horizon)
                for i in range(cfg.episodes)]

    def score(self, tok, dyn, cfg: EvalConfig, *, device) -> dict[str, float]:
        from evals.motion.motion import open_loop_curve, teacher_forced_curve, cross_chance_h

        eps = self._episodes(cfg)
        ol = open_loop_curve(tok, dyn, eps, device, cfg.K, cfg.horizon)
        tf = teacher_forced_curve(tok, dyn, eps, device, cfg.K, cfg.horizon)
        h = min(16, cfg.horizon)
        return {
            f"ol_pos_err@{h}": float(ol["model"][h]),
            f"tf_pos_err@{h}": float(tf["model"][h]),
            "cross_chance_h": float(cross_chance_h(ol["model"])),
        }


register(MotionEval(), midrun=True)
