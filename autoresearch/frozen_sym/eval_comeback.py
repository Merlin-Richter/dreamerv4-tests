"""The ColorField-SYM comeback eval — THE result-defining scalar of the
symbolic tier. FROZEN-LAYER-sym (see env.py header; spec:
tasks/in-progress/colorfield-sym-frozen-layer.md). This is the pixel tier's
frozen-v2.1 eval ported to symbols; scoring semantics are UNCHANGED, the
partial-visibility machinery is deleted because partial overlap is
structurally impossible on a cell viewport.

Shape: per episode, a REAL teacher-forced prefix (starts within 2 cells of a
corner — a corner anchors ALL borders on a fixed-size board) followed by a
long pure-IMAGINATION phase driven by a closed-loop policy, with the phase-5
rule throughout (policies consulted at phase-0 ticks only; STAY forced
off-phase). A cell tracker records every cell the grids show, with provenance:
  - real-observed  (first seen during the prefix)     -> comeback events scored
    against GROUND TRUTH (ungameable retention),
  - imagination-born (first seen during imagination)  -> comeback events scored
    for SELF-CONSISTENCY vs the previous visit's recorded value.
Comeback = in viewport -> fully left (not in viewport; NO partial-overlap
states exist) -> back in viewport, scored ONCE per event at the return visit.
The read is the grid value at the cell's viewport slot — exact by
construction, no majority voting. Registration is the path integral of taken
actions (the center may leave the board in imagination — extended coords).

Scoring is AGE-STANDARDIZED (v2.1): AGE = the MAX contiguous out-of-viewport
run in TICKS between visits (with no partial states this is the single
absence gap, but the max-run definition is kept verbatim — it is what the
brute-force cross-check verifies); events are binned by age and the headline
components are EQUAL-WEIGHT means over qualified bins. Ages are 5x dilated by
the phase rule: bin edges are kept from the pixel tier, bin [1,16] is expected
sparse (min leave-and-return ~ 2 effective moves ~ 10 ticks).

v2.1 scoring, verbatim from the pixel tier (post red-team):
  - per-bin CHANCE CORRECTION over IN-MAP events: acc_cc = max(0, (acc-.2)/.8);
  - BORDER (OUT-referenced) events EXCLUDED from the score, reported as the
    border_recall diagnostic (pure geometry, zero content-memory signal);
  - composite = real_cc * (0.7 + 0.3 * consistency_cc) — consistency can only
    AMPLIFY genuine ground-truth-anchored retention, never substitute for it.

Hard gates (score := 0.0 + flags on failure) — the Goodhart guards:
  1. action fidelity, EXACT on symbols: at a phase-0 tick the predicted grid
     must equal the previous grid shifted by the action (checked on the
     overlap; the newly revealed line is unconstrained — border/OUT fill or
     recalled content is the eval's business, not the gate's), AND at an
     off-phase tick the predicted grid must equal the previous grid UNCHANGED
     (the new, free gate). Pooled fraction >= 0.90. NB: unlike the pixel tier
     a constant-color grid PASSES this gate (a uniform grid is
     shift-invariant) — the entropy gate is what kills it, by design.
  2. color-marginal entropy: imagination-born first-seen IN-MAP colors ~
     uniform over the 5 palette colors (KL <= 0.2, >= 20 samples else fail).
  3. (driver-level) frozen-layer hash check — outside this module.
"""

import json

import numpy as np

from .adapters import make_adapter  # noqa: F401  (re-export convenience)
from .env import (BOARD, ColorFieldSymEnv, DELTAS, OUT_IDX, PHASE_PERIOD, STAY,
                  VIEW_CELLS, VIEW_HALF, apply_action, out_bands)
from .eval_policies import EVAL_SUITE

# --- Frozen eval configuration (defaults; the driver must not override the
# --- scoring semantics, only sizes for calibration) ---------------------------
PREFIX_LEN = 192                        # ticks (~38 effective moves)
IMAG_LEN = 768                          # ticks (~153 effective moves)
N_SEEDS = 8
BIN_EDGES = (1, 17, 33, 65, 129, 257)   # bins: [1,16] [17,32] [33,64] [65,128] [129,256] [257,inf)
                                        # ages in TICKS (5x dilated); bin1 expected sparse
MIN_EVENTS_PER_BIN = 30
W_REAL, W_IMAG = 0.7, 0.3               # composite = real_cc * (W_REAL + W_IMAG * consistency_cc)
OUT_TILE_WEIGHT = 0.1                   # kept in the EVENT LOG for analysis only;
                                        # border cells are excluded from the score
CHANCE_IN_MAP = 1.0 / 5.0               # uniform guess over the 5 map colors
FIDELITY_THRESHOLD = 0.90
ENTROPY_KL_MAX = 0.20
ENTROPY_MIN_SAMPLES = 20


def gt_color(map_arr, cell):
    ci, cj = cell
    if 0 <= ci < BOARD and 0 <= cj < BOARD:
        return int(map_arr[ci, cj])
    return OUT_IDX


class CellTracker:
    """Comeback bookkeeping over a grid stream (real prefix + imagination).

    on-screen := the cell is one of the 25 viewport slots at the registered
    center; fully-left := not in the viewport. There are NO partial-overlap
    states (the pixel tier's hovering exploits are structurally impossible), so
    a visit is a maximal contiguous run of in-viewport ticks and every tick
    between two visits is a full absence.

    AGE (v2.1, kept verbatim): the LONGEST contiguous out-of-viewport run since
    the previous visit — with no partial states this equals the single gap
    length, but the max-run bookkeeping is retained so the semantics (and the
    independent brute-force cross-check) are identical to the pixel tier.

    The visit's read = the grid value at the cell's slot on the RETURN tick
    (the moment memory is tested), exact by construction; it is used both as
    the scored comeback color and as the record the next comeback's
    consistency reference reads. A comeback event fires when a visit starts
    for a cell that (a) has a previous recorded value and (b) has been fully
    absent for >= 1 tick since its previous visit closed."""

    def __init__(self, map_arr, prefix_len):
        self.map = map_arr
        self.prefix_len = prefix_len
        self.cells = {}      # key (extended board coords) -> state dict
        self.events = []
        self.first_imag_colors = []   # first-seen colors of imagination-born cells

    def observe(self, t, grid, pos, is_real):
        r, c = int(pos[0]), int(pos[1])
        visible = {}
        for i in range(VIEW_CELLS):
            for j in range(VIEW_CELLS):
                visible[(r - VIEW_HALF + i, c - VIEW_HALF + j)] = int(grid[i, j])

        for key, st in self.cells.items():
            if key not in visible:                    # fully left: absence grows
                st["in_visit"] = False
                st["cur_off"] += 1
                if st["cur_off"] > st["max_off"]:
                    st["max_off"] = st["cur_off"]

        for key, color in visible.items():
            st = self.cells.get(key)
            if st is None:
                st = {"record": None, "provenance": "real" if is_real else "imag",
                      "last_onscreen": t, "cur_off": 0, "max_off": 0,
                      "in_visit": False}
                self.cells[key] = st
            if not st["in_visit"]:                    # a visit STARTS at t
                st["in_visit"] = True
                if st["record"] is None:
                    if st["provenance"] == "imag" and color != OUT_IDX:
                        self.first_imag_colors.append(color)
                elif st["max_off"] > 0:               # comeback: score ONCE, now
                    ref = (gt_color(self.map, key) if st["provenance"] == "real"
                           else st["record"])
                    self.events.append({
                        "cell": key,
                        "provenance": st["provenance"],
                        "t": t,
                        # age (v2.1) = longest contiguous full-absence run survived
                        "age": st["max_off"],
                        "age_onscreen": t - st["last_onscreen"],   # diagnostic (v2.0 def)
                        "color": int(color),
                        "ref": int(ref),
                        "correct": bool(color == ref),
                        "weight": OUT_TILE_WEIGHT if ref == OUT_IDX else 1.0,
                        "phase": "imag" if t >= self.prefix_len else "prefix",
                    })
                st["record"] = int(color)             # return-tick read = the record
                st["max_off"] = 0
            st["cur_off"] = 0
            st["last_onscreen"] = t

    def finalize(self):
        """No-op: events fire at visit start and nothing is pending at episode
        end. Kept for pixel-tier API parity (the pixel tracker closes visits)."""


def corner_start(rng):
    """Start within 2 cells of a random corner: the corner anchors ALL borders
    on the fixed-size board (Merlin)."""
    m = BOARD - 1
    cr, cc = rng.integers(0, 2, size=2)
    orow = int(rng.integers(0, 3))
    ocol = int(rng.integers(0, 3))
    return (orow if cr == 0 else m - orow, ocol if cc == 0 else m - ocol)


def fidelity_ok(prev, nxt, action, t):
    """Gate 1, exact on symbols. Off-phase ticks (t % 5 != 0) and STAY: the
    predicted grid must equal the previous grid UNCHANGED. Phase-0 moves: the
    predicted grid must equal the previous grid SHIFTED by the action — checked
    on the 4x5/5x4 overlap; the newly revealed line is unconstrained (it may be
    OUT fill at a border or recalled/imagined map content — the comeback eval
    scores that, not the gate)."""
    if t % PHASE_PERIOD != 0 or action == STAY:
        return bool(np.array_equal(nxt, prev))
    dr, dc = DELTAS[action]
    i0, i1 = max(0, -dr), VIEW_CELLS - max(0, dr)
    j0, j1 = max(0, -dc), VIEW_CELLS - max(0, dc)
    return bool(np.array_equal(nxt[i0:i1, j0:j1],
                               prev[i0 + dr:i1 + dr, j0 + dc:j1 + dc]))


def _band_abs_err(grid, pos):
    """Diagnostic: |imagined OUT band - band the TRUE board geometry would show
    at the path-integral position| averaged over the 4 sides, in cells
    (border-drift measure). Handles extended (off-board) centers."""
    b = out_bands(grid)
    rows = np.arange(pos[0] - VIEW_HALF, pos[0] + VIEW_HALF + 1)
    cols = np.arange(pos[1] - VIEW_HALF, pos[1] + VIEW_HALF + 1)
    mask = (((rows < 0) | (rows >= BOARD))[:, None]
            | ((cols < 0) | (cols >= BOARD))[None, :])
    exp = out_bands(np.where(mask, OUT_IDX, 0).astype(np.uint8))
    return float(np.mean([abs(b[k] - exp[k]) for k in b]))


def run_episode(adapter_factory, policy, map_seed, ep_seed,
                prefix_len=PREFIX_LEN, imag_len=IMAG_LEN, privileged=False):
    """One eval episode. Returns (events, fidelity_matches, first_imag_colors,
    band_abs_err, positions) — positions is the per-TICK registration (true
    positions during the prefix, action path-integral during imagination).

    privileged=False (the DEFAULT, and the only mode the harness driver may use
    for candidate models): adapter_factory receives None — a model must work
    from prefix_grids/prefix_actions alone (red-team S4: handing out env is an
    instant-oracle hole). privileged=True is for the frozen baselines only."""
    rng = np.random.default_rng(ep_seed)
    env = ColorFieldSymEnv()
    grid, _ = env.reset(seed=map_seed, start=corner_start(rng))
    adapter = adapter_factory(env if privileged else None)
    policy.reset(rng)
    tracker = CellTracker(env.map, prefix_len)

    # --- real prefix (teacher-forced): the policy sees real grids' bands, and
    # --- is consulted only at phase-0 ticks (STAY forced off-phase) -----------
    grids = [grid]
    actions = [STAY]
    positions = [env.pos]
    tracker.observe(0, grid, env.pos, is_real=True)
    for t in range(1, prefix_len):
        if t % PHASE_PERIOD == 0:
            a = policy.act(out_bands(grids[-1]), rng)
        else:
            a = STAY
        if a not in env.valid_actions():   # frozen-layer bug if this ever fires
            raise AssertionError(f"eval policy emitted invalid action {a} at {env.pos}")
        grid, _ = env.step(a)
        grids.append(grid)
        actions.append(a)
        positions.append(env.pos)
        tracker.observe(t, grid, env.pos, is_real=True)
    adapter.begin(np.stack(grids), np.asarray(actions, dtype=np.int64))

    # --- imagination phase: registration = path integral of taken actions ------
    pos = env.pos
    cur = grids[-1]
    fidelity = []
    band_err = []
    for t in range(prefix_len, prefix_len + imag_len):
        if t % PHASE_PERIOD == 0:
            a = policy.act(out_bands(cur), rng)
        else:
            a = STAY
        nxt = adapter.step(a)
        if t % PHASE_PERIOD == 0:
            pos = apply_action(pos, a, check=False)  # may leave the board: allowed
        fidelity.append(fidelity_ok(cur, nxt, a, t))
        band_err.append(_band_abs_err(nxt, pos))
        tracker.observe(t, nxt, pos, is_real=False)
        positions.append(pos)
        cur = nxt
    tracker.finalize()
    return tracker.events, fidelity, tracker.first_imag_colors, band_err, positions


def aggregate(events, fidelity, first_imag_colors,
              bin_edges=BIN_EDGES, min_events=MIN_EVENTS_PER_BIN,
              fidelity_threshold=FIDELITY_THRESHOLD):
    """Age-standardized components + gates + composite. Pure function of the
    collected statistics — the driver calls THIS for the loop's number.
    Identical math to the pixel tier's frozen-v2.1 aggregate."""
    edges = list(bin_edges) + [np.inf]
    imag_phase = [e for e in events if e["phase"] == "imag"]

    def component(prov):
        evs = [e for e in imag_phase if e["provenance"] == prov]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [e for e in evs if lo <= e["age"] < hi]
            # SCORED accuracy uses IN-MAP events only. OUT (border) cells are
            # pure geometry — a zero-content-memory model gets them right at any
            # age — and they dominate far bins where in-map events thin out, so
            # any nonzero weight lets them set the long-range score (pixel-tier
            # red-team S3). Border recall is a separate diagnostic instead.
            sel_in = [e for e in sel if e["ref"] != OUT_IDX]
            sel_out = [e for e in sel if e["ref"] == OUT_IDX]
            if sel_in:
                acc = sum(e["correct"] for e in sel_in) / len(sel_in)
                acc_cc = max(0.0, (acc - CHANCE_IN_MAP) / (1.0 - CHANCE_IN_MAP))
            else:
                acc = acc_cc = None
            bins.append({"lo": lo, "hi": (None if hi == np.inf else int(hi)),
                         "n": len(sel_in), "acc": acc, "acc_cc": acc_cc,
                         "n_border": len(sel_out),
                         "border_recall": (sum(e["correct"] for e in sel_out) / len(sel_out)
                                           if sel_out else None),
                         "qualified": len(sel_in) >= min_events})
        accs = [b["acc_cc"] for b in bins if b["qualified"]]
        return (float(np.mean(accs)) if accs else None), bins, len(evs)

    real_score, real_bins, n_real = component("real")
    imag_score, imag_bins, n_imag = component("imag")

    fid = float(np.mean(fidelity)) if len(fidelity) else 0.0
    gates = {"fidelity": {"value": fid, "passed": fid >= fidelity_threshold}}
    if len(first_imag_colors) >= ENTROPY_MIN_SAMPLES:
        counts = np.bincount(first_imag_colors, minlength=5)[:5].astype(float)
        p = counts / counts.sum()
        kl = float(np.sum([pi * np.log(pi / 0.2) for pi in p if pi > 0]))
        gates["entropy"] = {"kl_to_uniform": kl, "n": int(counts.sum()),
                            "passed": kl <= ENTROPY_KL_MAX}
    else:
        gates["entropy"] = {"kl_to_uniform": None, "n": len(first_imag_colors),
                            "passed": False, "reason": "insufficient imagination-born cells"}

    reasons = [k for k, g in gates.items() if not g["passed"]]
    flags = []
    composite = None
    if real_score is not None:
        # Multiplicative: consistency AMPLIFIES real retention, never substitutes.
        amp = W_REAL + W_IMAG * (imag_score if imag_score is not None else 0.0)
        composite = real_score * amp
        if imag_score is None:
            flags.append("no_qualified_imag_bins")   # not gating: amp floor 0.7
    else:
        reasons.append("no_qualified_real_bins")

    ages = [e["age"] for e in imag_phase]
    return {
        "composite": composite,
        "composite_gated": (composite if not reasons and composite is not None else 0.0),
        "gates_passed": not reasons,
        "fail_reasons": reasons,
        "flags": flags,
        "real_anchored": {"score": real_score, "bins": real_bins, "n_events": n_real},
        "consistency": {"score": imag_score, "bins": imag_bins, "n_events": n_imag},
        "gates": gates,
        "scoring": {"formula": "real_cc * (0.7 + 0.3 * consistency_cc)",
                    "chance_in_map": CHANCE_IN_MAP,
                    "border_cells": "excluded from score; border_recall diagnostic",
                    "age_units": "ticks (phase-5 dilated)"},
        "age_stats": {"n": len(ages),
                      "mean": (float(np.mean(ages)) if ages else None),
                      "median": (float(np.median(ages)) if ages else None)},
    }


def run_eval(adapter_factory, suite=EVAL_SUITE, n_seeds=N_SEEDS,
             prefix_len=PREFIX_LEN, imag_len=IMAG_LEN, seed0=0,
             min_events=MIN_EVENTS_PER_BIN, privileged=False, verbose=False):
    """Full eval: suite x seeds episodes, pooled aggregation. Returns the result
    dict (aggregate() output + per-policy breakdown + config + raw event log).
    privileged: see run_episode — False for candidate models, True for baselines."""
    all_events, all_fid, all_colors, all_band_err = [], [], [], []
    per_policy = {}
    for si, (name, factory) in enumerate(suite):
        pol_events = []
        for ki in range(n_seeds):
            map_seed = 100003 * (seed0 + 1) + 1009 * si + 2 * ki
            ep_seed = map_seed + 1
            ev, fid, colors, berr, _pos = run_episode(
                adapter_factory, factory(), map_seed, ep_seed, prefix_len, imag_len,
                privileged=privileged)
            for e in ev:
                e["policy"] = name
                e["episode"] = (si, ki)
            all_events += ev
            pol_events += ev
            all_fid += fid
            all_colors += colors
            all_band_err += berr
        imag_ev = [e for e in pol_events if e["phase"] == "imag"]
        wsum = sum(e["weight"] for e in imag_ev)
        per_policy[name] = {
            "n_events": len(imag_ev),
            "weighted_acc": (sum(e["weight"] * e["correct"] for e in imag_ev) / wsum
                             if wsum > 0 else None),
        }
        if verbose:
            print(f"[eval-sym] {name}: {per_policy[name]}", flush=True)

    result = aggregate(all_events, all_fid, all_colors, min_events=min_events)
    result["per_policy"] = per_policy
    result["border_drift_cells"] = (float(np.mean(all_band_err)) if all_band_err else None)
    result["config"] = {"suite": [n for n, _ in suite], "n_seeds": n_seeds,
                        "prefix_len": prefix_len, "imag_len": imag_len,
                        "seed0": seed0, "bin_edges": list(BIN_EDGES),
                        "min_events_per_bin": min_events, "privileged": privileged}
    result["events"] = [dict(e, cell=list(e["cell"])) for e in all_events]
    return result


def save_json(result, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
