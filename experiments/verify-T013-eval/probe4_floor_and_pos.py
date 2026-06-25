"""V-T013-eval Probe 4 — (a) no-memory floor (chance for color recall) so probe3's ~13 dRGB is
contextualized; (b) does A2 (near-clean source) help POSITION recall vs A1, even though color is
equal? Position is dynamic state; if A2's near-clean curtain latent leaks velocity/position the
near-clean source could flatter dynamic recall (an OOD-pairing confound to flag).

Reuses probe3's step/rollout machinery but also returns pos_err at the reveal frame, and adds a
no_mem (learned-init memory carried) static rollout as the floor.

Seed 0. venv/Scripts/python.exe -u.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_SRC = _HERE.parents[1] / "src"
for _p in (_SRC, _SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import probe3_rollout as P3
from probe_env import make_probe_batch
from revisit_probe import detect_ball, _decode_frame
import numpy as np


@torch.no_grad()
def rollout_full(dyn, tok, ep, P, src_tau_idx, carry_mode, K, no_mem=False):
    z_prefix = P3.encode_clip(tok, ep.frames, 0, P)
    if no_mem:
        mem_carry = dyn.memory_tokens[None, ...].mean(0)  # placeholder, overwritten below
        mem_carry = dyn.memory_tokens.unsqueeze(0)        # (1,M,E) learned init
    else:
        mem_carry = P3.write_memory_from_prefix(dyn, z_prefix)
    src_lat = z_prefix[:, -1:]
    n_steps = ep.frames.shape[0] - P
    has_act = dyn.n_actions > 0
    last = None
    for i in range(n_steps):
        cur_idx = P + i
        src_act = ep.actions[cur_idx - 1] if has_act else None
        nxt_act = ep.actions[cur_idx] if has_act else None
        want = (carry_mode == "B2") and not no_mem
        z, new_mem = P3.step(dyn, mem_carry, src_lat, src_act, nxt_act, src_tau_idx, K, want)
        if want:
            mem_carry = new_mem
        src_lat = z
        last = z
    f, x, y, color = detect_ball(_decode_frame(tok, last[:, 0]))
    if not f:
        return float("nan"), float("nan")
    rev = ep.reveal_index
    gx, gy = ep.states[rev, :2]
    pos = float(np.hypot(x - gx, y - gy))
    cdr = float(np.abs(color.astype(np.float32) - ep.ball_color.astype(np.float32)).mean())
    return cdr, pos


def main():
    tok, dyn, dcfg = P3.load()
    K = dcfg.inference_steps
    tau_ctx_idx = round(dcfg.context_signal * dyn.K_max)
    P = 3
    n_ep = 32
    occ_grid = [2, 8, 16, 24]
    print(f"== Probe 4: floor + position (P={P}, K={K}, n_ep={n_ep}) ==")
    print(f"{'n_occ':>5} | {'A1B1 dRGB/pos':>16} | {'A2B1 dRGB/pos':>16} | {'NOMEM dRGB/pos':>16}")
    for n_occ in occ_grid:
        eps = make_probe_batch(k=n_occ, n_seeds=n_ep, P=P, R=1, seed0=5000 + n_occ)

        def agg(st, cm, nm=False):
            vc, vp = [], []
            for ep in eps:
                c, p = rollout_full(dyn, tok, ep, P, st, cm, K, no_mem=nm)
                if np.isfinite(c): vc.append(c)
                if np.isfinite(p): vp.append(p)
            return (np.mean(vc) if vc else float('nan'),
                    np.mean(vp) if vp else float('nan'))

        a1 = agg(0, "B1"); a2 = agg(tau_ctx_idx, "B1"); fl = agg(0, "B1", nm=True)
        print(f"{n_occ:>5} | {a1[0]:7.1f}/{a1[1]:6.1f}  | {a2[0]:7.1f}/{a2[1]:6.1f}  | "
              f"{fl[0]:7.1f}/{fl[1]:6.1f}")


if __name__ == "__main__":
    main()
