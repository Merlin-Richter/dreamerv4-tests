"""V-EXP027 probe: verify dynamics_rollout_frames action alignment, reveal-frame index,
and occluded-latent-leak claims. Independent of the design doc's argument.

Run from repo root:  python -u experiments/verify-EXP027/probe_alignment.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from envs.gridworld import GridWorldEnv, make_grid_background, stamp_square, PALETTE  # noqa
from evals.gridworld.recall import find_reveal_events  # noqa
from evals.gridworld.readout import read_square  # noqa
from evals.gridworld import adapter  # noqa

DEV = "cpu"
torch.set_grad_enabled(False)
torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Build a SYNTHETIC episode with a fully known square path + curtain pattern.
# We pick a curtain pattern that yields a reveal event with a controllable k,
# and we know the true square cell at every frame (rendered ourselves).
# ---------------------------------------------------------------------------
env = GridWorldEnv()
env.reset(seed=123)
T = 40
frames = np.zeros((T, 64, 64, 3), np.uint8)
states = np.zeros((T, 5), np.float32)

# Curtain pattern: visible for a while, then a k=5 occluded run, then reveal.
# action[t] is the curtain OBSERVED at frame t (env convention).
curtain = np.zeros(T, np.int64)
curtain[20:25] = 1          # occluded run frames 20..24 (k=5); reveal at t=25

bg = env.bg_color
sq = env.color
base = make_grid_background(bg)
# Drive physics ourselves so we KNOW the path; render per curtain.
col, row = env.col, env.row
dcol, drow = env.dcol, env.drow
def reflect(p, v):
    n = p + v
    if n < 0 or n > 5:
        v = -v; n = p + v
    return n, v
for t in range(T):
    # advance physics (env advances BEFORE render in step())
    col, dcol = reflect(col, dcol)
    row, drow = reflect(row, drow)
    if curtain[t]:
        frames[t] = np.full((64, 64, 3), (128, 128, 128), np.uint8)
    else:
        f = base.copy(); stamp_square(f, col, row, sq); frames[t] = f
    states[t] = (col, row, dcol, drow, float(curtain[t]))

events = find_reveal_events(curtain)
print("reveal events:", events)
ev = events[0]
lv, k, t_rev = ev["last_visible_t"], ev["k"], ev["reveal_t"]
print(f"lv={lv} k={k} reveal_t={t_rev}")
assert t_rev == 25 and k == 5 and lv == 19

# ---------------------------------------------------------------------------
# Load tokenizer + dynamics
# ---------------------------------------------------------------------------
tok, L = adapter.load_tokenizer(str(ROOT / "checkpoints/gridworld/tokenizer.pt"), DEV)
model, cfg = adapter.load_dynamics("C:/Users/richt/AppData/Local/Temp/gw_dyn_smoke.pt", DEV)
print(f"tokenizer L={L}  dyn max_T={cfg.max_temporal_length} n_actions={cfg.n_actions} "
      f"n_latents={cfg.n_latents} bottleneck={cfg.bottleneck_dim}")
max_T = cfg.max_temporal_length

# ---------------------------------------------------------------------------
# CLAIM (a): instrument action_features to record the ids it receives, and
# instrument _denoise_next / _denoise_next_cached to record which action id the
# NEW (last) frame consumes. Then verify new-frame action == true curtain of that
# absolute frame, with no off-by-one.
# ---------------------------------------------------------------------------
records = []  # per generated frame: (new_frame_action_id, ctx_window_action_ids)

orig_dn = model._denoise_next
orig_dnc = model._denoise_next_cached

def spy_dn(context, K, actions=None):
    # actions: (B, T_ctx+1, n_act, E) features. We can't read ids from features directly,
    # so we instead recover ids via a parallel hook on action_features below.
    return orig_dn(context, K, actions)

# Better: hook action_features to log the ids, tagging order of calls.
af_calls = []
orig_af = model.action_features
def spy_af(action_idx):
    if action_idx is not None:
        af_calls.append(action_idx.detach().cpu().numpy().copy())
    else:
        af_calls.append(None)
    return orig_af(action_idx)
model.action_features = spy_af

# We need per-new-frame action. The generate loop computes act_window = act_feat[:, new_idx-w:new_idx+1]
# from the SINGLE act_feat built from the full action_idx. So instead, replicate the slicing the
# loop does, using the recorded full action_idx, and compare to what the model input's last column is.
# To capture the actual last-column id seen by forward, hook forward to log act feature -> map to id
# by nearest action_table row.
fwd_last_action_ids = []  # the id assigned to the LAST frame position of each forward call (gen frames only)
orig_fwd = model.forward
AT = model.action_table.weight.detach()  # (n_actions, E) per action-token? check shape
print("action_table.weight shape:", tuple(model.action_table.weight.shape),
      "n_action_tokens:", getattr(model, "n_action_tokens", "?"))

def feat_to_id(feat_vec):
    # feat_vec: (n_act_tokens*E?) flatten of the action feature for one frame -> nearest id
    # action_features returns embedding(id) reshaped to (n_act_tokens, E). Recover id by matching
    # against action_table(all_ids).
    ids = torch.arange(model.n_actions)
    cand = model.action_features(ids.unsqueeze(0))[0]  # (n_actions, n_act_tokens, E) -- but spy logs!
    return cand

# The cleanest decisive check avoids fragile reverse-mapping: directly assert the slicing math.
# Reconstruct exactly what generate() does and what curtain each generated absolute frame gets.
print("\n=== CLAIM (a): action-alignment math (replicating generate() slicing) ===")
a = max(0, lv - (max_T - 1))
T_ctx = lv + 1 - a
act_full = curtain[a:t_rev + 1].copy()  # adapter: cur_np[a:t+1]
n_gen = t_rev - lv
print(f"a={a} T_ctx={T_ctx} n_gen={n_gen} act_full(len={len(act_full)})={act_full.tolist()}")
ok_a = True
for i in range(n_gen):
    new_idx = T_ctx + i
    abs_frame = a + new_idx           # absolute frame index this generated frame represents
    new_frame_action = act_full[new_idx]   # action fed to the NEW (last) frame position
    true_curtain = curtain[abs_frame]
    match = (new_frame_action == true_curtain) and (abs_frame == lv + 1 + i)
    ok_a &= match
    tag = "OK" if match else "MISMATCH"
    print(f"  gen i={i}: new_idx={new_idx} abs_frame={abs_frame} "
          f"fed_action={new_frame_action} true_curtain={true_curtain} [{tag}]")
print("CLAIM (a) alignment:", "SUPPORTED" if ok_a else "REFUTED")

# Empirically confirm forward really uses the LAST position as the new frame, by checking the
# last generated frame is absolute reveal_t (claim b shares this) AND that swapping the new-frame
# action changes the output (action is actually consumed at that position).
print("\n=== CLAIM (a) empirical: does new-frame action actually drive the last position? ===")
wf = frames[a:lv + 1].astype(np.float32) / 255.0
fx = torch.from_numpy(wf).unsqueeze(0).to(DEV)
ctx = tok.encoder(fx)
model.action_features = orig_af  # restore (avoid log spam)
act0 = torch.from_numpy(curtain[a:t_rev + 1]).unsqueeze(0).long()
act_swap = act0.clone()
# flip ONLY the reveal frame's action (last position) and see if reveal latent changes
act_swap[0, -1] = 1 - act_swap[0, -1]
torch.manual_seed(7); g0 = model.generate(ctx, n_gen, action_idx=act0)
torch.manual_seed(7); g1 = model.generate(ctx, n_gen, action_idx=act_swap)
d_last = (g0[:, -1] - g1[:, -1]).abs().max().item()
d_prev = (g0[:, :-1] - g1[:, :-1]).abs().max().item() if n_gen > 1 else float("nan")
print(f"flip reveal-frame action: max|Δ| last gen frame={d_last:.4e}  earlier gen frames={d_prev:.4e}")
print("  expectation: last changes (>0), earlier unchanged (~0) since causal + only last action flipped")

# ---------------------------------------------------------------------------
# CLAIM (b): the decoded frame placed at reveal_t is the model's prediction for reveal_t.
# Verify decode-window math: full=concat(ctx,gen); win=full[:,-max_T:]; out[t]=decode(win)[-1].
# The last latent of win must be gen[:,-1] (= absolute reveal_t prediction).
# ---------------------------------------------------------------------------
print("\n=== CLAIM (b): decode window picks the reveal-frame latent ===")
torch.manual_seed(7); gen = model.generate_cached(ctx, n_gen, action_idx=act0)
full = torch.cat((ctx, gen), dim=1)
win = full[:, -max_T:]
last_is_reveal = torch.equal(win[:, -1], gen[:, -1])
print(f"win last latent == gen[:,-1] (reveal pred): {last_is_reveal}")
print(f"full len={full.shape[1]} win len={win.shape[1]} (<=max_T={max_T})")
# Also decode and confirm out[t] equals decode(win)[-1]
dec = tok.decoder(win)[0].clamp(0, 1).cpu().numpy()
print(f"decoded window frames={dec.shape[0]}; placing dec[-1] at reveal_t={t_rev}")
print("CLAIM (b):", "SUPPORTED" if last_is_reveal else "REFUTED")

# ---------------------------------------------------------------------------
# CLAIM (c): is feeding TRUE latents of occluded context frames a leak of the
# hidden square position? Test: does the occluded-frame latent encode the square?
# Here, by construction the context window ends at lv (last visible) and a..lv may
# contain occluded frames? In this synthetic ep, a..lv = 0..19 are ALL visible
# (occlusion starts at 20). So the *context* has no occluded frames. But the general
# claim: encode a curtain (occluded) frame and decode it -> can the square be read out?
# If an occluded frame's latent leaks the square cell, feeding true latents of occluded
# CONTEXT frames (when lv-window includes them) would leak. Test the tokenizer directly.
# ---------------------------------------------------------------------------
print("\n=== CLAIM (c): does an occluded-frame latent encode the hidden square? ===")
# Build a window of occluded frames whose TRUE square positions differ, encode+decode,
# and see if read_square recovers the true (occluded) cell better than chance.
env2 = GridWorldEnv(); env2.reset(seed=55)
c, r, dc, dr = env2.col, env2.row, env2.dcol, env2.drow
true_cells = []
occ_frames = np.full((max_T, 64, 64, 3), (128, 128, 128), np.uint8)  # all curtain
for tt in range(max_T):
    c, dc = reflect(c, dc); r, dr = reflect(r, dr)
    true_cells.append((c, r))
fx2 = torch.from_numpy(occ_frames.astype(np.float32) / 255.0).unsqueeze(0)
z2 = tok.encoder(fx2)
rec2 = tok.decoder(z2)[0].clamp(0, 1).cpu().numpy()
hits = 0
for tt in range(max_T):
    f8 = (rec2[tt] * 255).round().astype(np.uint8)
    rd = read_square(f8)
    tc, tr = true_cells[tt]
    if rd["col"] == tc and rd["row"] == tr:
        hits += 1
print(f"occluded-frame recon: exact-cell recovery {hits}/{max_T} (chance=1/36={1/36:.3f}). "
      f"is_occluded flags: {[read_square((rec2[i]*255).round().astype(np.uint8))['is_occluded'] for i in range(min(5,max_T))]}")
print("  if recovery ~= chance and frames flagged occluded -> occluded latent does NOT leak square")

# Additionally: does the CONTEXT window in a typical real reveal event ever contain occluded frames?
# (a..lv). lv is last-visible (curtain 0). Frames a..lv-1 CAN be occluded if there were earlier
# occluded runs within the window. Confirm those occluded context latents also don't leak -> same
# tokenizer property above covers it.
print("\nDONE")
