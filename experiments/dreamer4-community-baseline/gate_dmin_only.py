#!/usr/bin/env python3
"""Correctness gate for the community-Dreamer4 d_min-only arm, run against the pinned upstream
checkout itself (nicklashansen/dreamer4 @ b8abafbf) rather than a copy of it.

Run before spending H100 hours:

    python experiments/dreamer4-community-baseline/gate_dmin_only.py --dreamer4 "$D4_ROOT"

Claims under test:

  G1  --self_fraction 0 => B_self=0 => upstream's dynamics_pretrain_loss reduces EXACTLY to its
      finest-step flow term: loss == loss_emp, loss_self == 0, bootstrap_mse == 0.  This is why
      the arm needs no source patch.
  G2  the B_self=0 path runs, backwards, and puts gradient on the network.
  G3  under d_min-only training the step_embed rows for K < k_max receive EXACTLY zero gradient.
      Upstream keys the shortcut step size through nn.Embedding(log2(k_max)+1, d_model)
      (model.py step_embed), so those rows stay at their random init forever.  Row 2 is K=4 --
      the default eval schedule.  This is the whole reason the arm must be scored at K=8.
  G4  the vanilla control (self_fraction=0.25) DOES train those rows, so K=4 is in-distribution
      for it while K=8 is in-distribution for BOTH arms and is therefore the fair setting.
  G5  a d_min-only state dict still loads strict=True into the stock upstream Dynamics, and the
      two eval schedules resolve to the step indices claimed above.

Gradient probes must not be taken at initialization: upstream zero-inits flow_x_head, so a fresh
model has exactly zero gradient everywhere upstream of it and every probe reads a false zero.
build() breaks that degeneracy the way one optimizer step would.
"""
import argparse
import math
import sys
from pathlib import Path

import torch

K_MAX = 8
EMAX = int(round(math.log2(K_MAX)))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True,
                    help="pinned upstream checkout ($D4_ROOT from setup_upstream.sh)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(args.dreamer4 / "dreamer4"))
    from model import Dynamics
    from train_dynamics import dynamics_pretrain_loss, make_tau_schedule

    dev = args.device

    def build():
        torch.manual_seed(0)
        m = Dynamics(
            d_model=64, d_bottleneck=32, d_spatial=64, n_spatial=8,
            n_register=4, n_agent=1, n_heads=4, depth=2, k_max=K_MAX,
            dropout=0.0, mlp_ratio=4.0, time_every=1,
            space_mode="wm_agent_isolated", scale_pos_embeds=False,
        ).to(dev)
        with torch.no_grad():
            m.flow_x_head.weight.normal_(0.0, 0.02)
            m.flow_x_head.bias.normal_(0.0, 0.02)
        return m

    B, T = 8, 6
    torch.manual_seed(0)
    z1 = torch.randn(B, T, 8, 64, device=dev)
    actions = torch.zeros(B, T, 16, device=dev)
    actions[:, :, 0] = 1.0
    act_mask = torch.ones(B, T, 16, device=dev)

    # G1 / G2 -- the arm
    m = build()
    torch.manual_seed(1)
    loss0, aux0 = dynamics_pretrain_loss(
        m, z1=z1, actions=actions, act_mask=act_mask, k_max=K_MAX,
        B_self=0, step=10_000, bootstrap_start=5_000,
    )
    boot0, self0 = float(aux0["bootstrap_mse"]), float(aux0["loss_self"])
    emp0, tot0 = float(aux0["loss_emp"]), float(loss0.detach())
    check("G1 bootstrap_mse == 0", boot0 == 0.0, f"{boot0}")
    check("G1 loss_self == 0", self0 == 0.0, f"{self0}")
    check("G1 loss == loss_emp exactly", tot0 == emp0, f"{tot0} vs {emp0}")
    loss0.backward()
    check("G2 loss is finite", bool(torch.isfinite(loss0).item()))
    n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    check("G2 network receives gradient", n_grad > 0, f"{n_grad} tensors with nonzero grad")

    # G3 -- the dead step_embed rows
    g = m.step_embed.weight.grad
    rows = [float(g[i].abs().sum()) for i in range(g.shape[0])]
    check("G3 step_embed has log2(k_max)+1 rows", g.shape[0] == EMAX + 1, f"rows={g.shape[0]}")
    check(f"G3 row {EMAX} (K={K_MAX} = d_min) IS trained", rows[EMAX] > 0, f"|grad|={rows[EMAX]:.4e}")
    check("G3 rows 0..emax-1 get EXACTLY zero gradient",
          all(v == 0.0 for v in rows[:EMAX]),
          f"{[f'{v:.3e}' for v in rows[:EMAX]]}  (index 2 = K=4 = the default eval schedule)")

    # G4 -- the control
    m2 = build()
    torch.manual_seed(1)
    loss1, aux1 = dynamics_pretrain_loss(
        m2, z1=z1, actions=actions, act_mask=act_mask, k_max=K_MAX,
        B_self=int(round(0.25 * B)), step=10_000, bootstrap_start=5_000,
    )
    loss1.backward()
    rows2 = [float(m2.step_embed.weight.grad[i].abs().sum()) for i in range(EMAX + 1)]
    check("G4 control bootstrap term fires", float(aux1["bootstrap_mse"]) > 0,
          f"boot_mse={float(aux1['bootstrap_mse']):.4e}")
    check("G4 control DOES train the coarse rows", any(v > 0 for v in rows2[:EMAX]),
          f"{[f'{v:.3e}' for v in rows2[:EMAX]]}")

    # G5 -- checkpoint compatibility and schedule resolution
    build().load_state_dict(m.state_dict(), strict=True)
    check("G5 d_min state dict loads strict=True into stock Dynamics", True)
    s8 = make_tau_schedule(k_max=K_MAX, schedule="finest")
    s4 = make_tau_schedule(k_max=K_MAX, schedule="shortcut", d=0.25)
    check("G5 finest -> K=8, step_idx=3 (trained by both arms)",
          s8["K"] == K_MAX and s8["e"] == EMAX, f"K={s8['K']} e={s8['e']}")
    check("G5 shortcut d=0.25 -> K=4, step_idx=2 (untrained by the arm)",
          s4["K"] == 4 and s4["e"] == 2, f"K={s4['K']} e={s4['e']}")

    print()
    if FAILS:
        print("GATE FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("GATE PASSED: self_fraction=0 is exactly d_min-only, and K=4 reads an untrained "
          "step_embed row while K=8 is in-distribution for both arms.")


if __name__ == "__main__":
    main()
