"""Revisit-consistency eval — the FROZEN H2/H3 spine.

`probe.py` holds the frozen measurement logic (frozen @ commit 5503e75; any change is a logged
decision — GOAL.md §8). `RevisitEval` is a thin adapter exposing it through the common interface.
"""
from __future__ import annotations

from pathlib import Path

from evals.base import Eval, EvalConfig, EvalResult, register


class RevisitEval(Eval):
    name = "revisit"
    frozen = True
    compatible_envs = ("occluded_bouncing",)

    def score(self, tok, dyn, cfg: EvalConfig, *, device) -> dict[str, float]:
        from evals.probe_env import make_probe_batch
        from evals.revisit.probe import run_condition

        tok_win = cfg.tok_win or cfg.window_N
        out: dict[str, float] = {}
        for n_occ in (12, 16, 24):
            eps = make_probe_batch(k=n_occ, n_seeds=cfg.episodes, seed0=5000 + n_occ,
                                   P=cfg.prefix_P, R=1)
            agg = run_condition(tok, dyn, eps, device, cfg.K, tok_win)
            out[f"color_dRGB@{n_occ}"] = float(agg["color_dRGB"])
        return out

    def report(self, tok, dyn, cfg: EvalConfig, out_dir, *, device) -> EvalResult:
        from evals.revisit.probe import render_sheet

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sheet = out_dir / "sheet.png"
        render_sheet(tok, dyn, device, cfg.K, str(sheet))
        return EvalResult(
            scores=self.score(tok, dyn, cfg, device=device),
            artifacts={"sheet": sheet},
            meta={"frozen": True, "spine_commit": "5503e75"},
        )


register(RevisitEval(), midrun=False)
