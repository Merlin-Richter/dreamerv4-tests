"""V-EXP027 stronger CLAIM (c) probe: does the TEMPORAL tokenizer carry a visible square's
position forward into a following occluded frame's latent? That would be the actual leak path:
the adapter encodes the TRUE context window a..lv which can contain visible THEN occluded frames.
If the occluded frame's decoded latent reveals where the square WAS/WOULD-BE, feeding true context
latents leaks hidden position.

Test: visible frames showing the square, then an occluded run, all in ONE encode window.
Read the square out of the occluded frames' recon. Leak <=> recovery > chance / matches true cell.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from envs.gridworld import GridWorldEnv, make_grid_background, stamp_square  # noqa
from evals.gridworld.readout import read_square  # noqa
from evals.gridworld import adapter  # noqa

torch.set_grad_enabled(False)
DEV = "cpu"
tok, L = adapter.load_tokenizer(str(ROOT / "checkpoints/gridworld/tokenizer.pt"), DEV)

def reflect(p, v):
    n = p + v
    if n < 0 or n > 5:
        v = -v; n = p + v
    return n, v

n_eps = 30
occ_hit = occ_n = 0
occ_dist = []
for ep in range(n_eps):
    env = GridWorldEnv(); env.reset(seed=1000 + ep)
    bg, sq = env.bg_color, env.color
    base = make_grid_background(bg)
    c, r, dc, dr = env.col, env.row, env.dcol, env.drow
    # window: 8 visible frames (square shown), then 8 occluded frames -> L=16
    frames = np.zeros((L, 64, 64, 3), np.uint8)
    cells = []
    for t in range(L):
        c, dc = reflect(c, dc); r, dr = reflect(r, dr)
        cells.append((c, r))
        if t < 8:
            f = base.copy(); stamp_square(f, c, r, sq); frames[t] = f
        else:
            frames[t] = np.full((64, 64, 3), (128, 128, 128), np.uint8)
    z = tok.encoder(torch.from_numpy(frames.astype(np.float32) / 255).unsqueeze(0))
    rec = tok.decoder(z)[0].clamp(0, 1).cpu().numpy()
    for t in range(8, L):  # the occluded frames
        f8 = (rec[t] * 255).round().astype(np.uint8)
        rd = read_square(f8)
        tc, tr = cells[t]
        d = max(abs(rd["col"] - tc), abs(rd["row"] - tr))
        occ_dist.append(d)
        occ_hit += int(d == 0); occ_n += 1

print(f"Occluded frames AFTER visible frames in same temporal window:")
print(f"  exact-cell recovery: {occ_hit}/{occ_n} = {occ_hit/occ_n:.3f}  (chance 1/36 = {1/36:.3f})")
print(f"  mean Chebyshev dist: {np.mean(occ_dist):.2f}  (random uniform ~2.4)")
leak = occ_hit / occ_n > 3 * (1 / 36)
print(f"  CLAIM (c): {'POSSIBLE LEAK - investigate' if leak else 'NO LEAK (occluded latents do not carry square)'}")
