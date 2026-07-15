"""The ColorField comeback eval — THE result-defining scalar of the autoresearch
harness. FROZEN LAYER (see env.py header). Spec: tasks/*/colorfield-env-and-eval.md
(eval v2, design rounds of 2026-07-06).

Shape: per episode, a REAL teacher-forced prefix (starts near a corner — a corner
anchors ALL borders on a fixed-size board) followed by a long pure-IMAGINATION
phase driven by a closed-loop policy. A cell tracker records every tile the frames
show, with provenance:
  - real-observed  (first seen during the prefix)     -> comeback events scored
    against GROUND TRUTH (ungameable retention),
  - imagination-born (first seen during imagination)  -> comeback events scored for
    SELF-CONSISTENCY vs the previous visit's recorded color.
Comeback = on-screen -> fully-left (zero overlap) -> center back on-screen.

Scoring is AGE-STANDARDIZED: events are binned by age (steps since the cell was
last on-screen) and the headline components are EQUAL-WEIGHT means over qualified
bins, so shifts in the age distribution (e.g. early imagined borders making
recalled information younger) move bin populations, not the score.

Hard gates (score := 0.0 + flags on failure) — the Goodhart guards:
  1. action fidelity: imagined frame-to-frame shift must match the commanded 4px
     move (catches 'actions do nothing' models),
  2. color-marginal entropy: imagination-born first-seen colors ~ uniform over the
     5 in-map palette colors (catches collapse-to-one-color; pure self-consistency
     alone has a degenerate optimum — an all-one-color world is perfectly
     consistent),
  3. (driver-level) frozen-layer hash check — outside this module.

SCORING (v2.1 — revised after the 2026-07-06 adversarial red-team,
experiments/colorfield-redteam/REPORT.md, which showed the original additive
0.7*real + 0.3*consistency with raw bin accuracies was Goodhart-exploitable: a
plain 64-frame window buffer scored 0.62 and a zero-retention 'consistent liar'
outscored honest short memory):
  - per-bin CHANCE CORRECTION over IN-MAP events: acc_cc = max(0, (acc-0.2)/0.8).
    Long-range failure now scores 0 in its bins, not the 0.2 chance floor — a
    bounded-window model's score is ~ the fraction of age bins its window covers.
  - AGE = longest contiguous zero-overlap absence survived (see CellTracker —
    weaker definitions fell to partial-visibility hovering / multi-gap bridging).
  - BORDER (OUT-referenced) tiles are EXCLUDED from the scored accuracy and
    reported as a separate border_recall diagnostic: they are pure geometry
    (zero content-memory signal) and dominate far bins at any nonzero weight
    (measured ~78% of bin [65,129) events). This supersedes the earlier 0.1
    weight — flagged to Merlin with the evidence.
  - composite = real_cc * (0.7 + 0.3 * consistency_cc): consistency can only
    AMPLIFY genuine ground-truth-anchored retention, never substitute for it.
    A zero-retention liar now scores 0.0 (was 0.43); oracle stays exactly 1.0.
"""

import json
from collections import Counter

import numpy as np

from .adapters import make_adapter  # noqa: F401  (re-export convenience)
from .env import (ColorFieldEnv, DELTAS, LATTICE, N_CELLS, OUT_IDX, PITCH_PX,
                  STAY, TL_OFFSET, VIEW_PX, WORLD_PX, apply_action)
from .eval_policies import EVAL_SUITE
from .readout import border_bands, estimate_shift, read_cells

# --- Frozen eval configuration (defaults; the driver must not override the
# --- scoring semantics, only sizes for calibration) ---------------------------
PREFIX_LEN = 192
IMAG_LEN = 768
N_SEEDS = 8
BIN_EDGES = (1, 17, 33, 65, 129, 257)   # bins: [1,16] [17,32] [33,64] [65,128] [129,256] [257,inf)
                                        # NB structural min comeback age is 6 (audit N1)
MIN_EVENTS_PER_BIN = 30
W_REAL, W_IMAG = 0.7, 0.3               # composite = real_cc * (W_REAL + W_IMAG * consistency_cc)
OUT_TILE_WEIGHT = 0.1                   # kept in the EVENT LOG for analysis only;
                                        # border tiles are excluded from the score
CHANCE_IN_MAP = 1.0 / 5.0               # uniform guess over the 5 map colors
FIDELITY_THRESHOLD = 0.90
ENTROPY_KL_MAX = 0.20
ENTROPY_MIN_SAMPLES = 20


def gt_color(map_arr, cell):
    ci, cj = cell
    if 0 <= ci < N_CELLS and 0 <= cj < N_CELLS:
        return int(map_arr[ci, cj])
    return OUT_IDX


class CellTracker:
    """Comeback bookkeeping over a frame stream (real prefix + imagination).

    Visit = maximal contiguous run of on-screen frames for a cell. A visit's read
    color = majority color among its max-visibility frames. A comeback event fires
    when a visit STARTS for a cell that (a) has a previous recorded color and
    (b) has had at least one ZERO-overlap frame since its previous visit closed.

    AGE (v2.1): the LONGEST contiguous zero-overlap stretch since the previous
    visit — the full-absence length the model demonstrably survived. Two weaker
    definitions fell to the bounded-window exploit while fencing it:
      - "steps since last on-screen" (v2.0): a cell can hover at 1-5px partial
        overlap for dozens of steps, keeping its color alive in any window model;
      - "steps since first zero-overlap": several short absences can be CHAINED
        via partial-visibility refreshes between them (multi-gap bridging), so
        the total elapsed span overstates the retention actually demonstrated.
    A window-W model cannot bridge a single absence > W, so with max-gap age the
    beyond-window bins are chance by construction — exactly the semantics the
    harness needs. "Steps since last on-screen" is kept as `age_onscreen`
    (diagnostic only).
    """

    def __init__(self, map_arr, prefix_len):
        self.map = map_arr
        self.prefix_len = prefix_len
        self.cells = {}      # key -> state dict
        self.events = []
        self.first_imag_colors = []   # first-seen colors of imagination-born cells

    def observe(self, t, frame, pos, is_real):
        reads = read_cells(frame, pos)
        onscreen = {k for k, r in reads.items() if r.on_screen}

        for key, st in self.cells.items():
            if key not in reads:                      # zero overlap: absence grows
                if st["visit"] is not None:
                    self._close_visit(key, st)
                st["cur_zero"] += 1
                st["max_zero"] = max(st["max_zero"], st["cur_zero"])
            else:
                st["cur_zero"] = 0                    # any overlap ends the absence
                if key not in onscreen and st["visit"] is not None:
                    self._close_visit(key, st)        # partial: visit ends, no gap

        for key in onscreen:
            r = reads[key]
            st = self.cells.get(key)
            if st is None:
                st = {"record": None, "provenance": "real" if is_real else "imag",
                      "last_onscreen": t, "cur_zero": 0, "max_zero": 0, "visit": None}
                self.cells[key] = st
            if st["visit"] is None:
                st["visit"] = {"start": t,
                               "came_back": st["record"] is not None and st["max_zero"] > 0,
                               "prev_last": st["last_onscreen"],
                               "gap_len": st["max_zero"],
                               "reads": []}
                st["max_zero"] = 0
            st["visit"]["reads"].append((r.ov_y * r.ov_x, r.color))
            st["last_onscreen"] = t

    def finalize(self):
        for key, st in self.cells.items():
            if st["visit"] is not None:
                self._close_visit(key, st)

    def _close_visit(self, key, st):
        v = st["visit"]
        max_area = max(a for a, _ in v["reads"])
        color = Counter(c for a, c in v["reads"] if a == max_area).most_common(1)[0][0]
        if st["record"] is None and st["provenance"] == "imag" and color != OUT_IDX:
            self.first_imag_colors.append(color)
        if v["came_back"]:
            ref = gt_color(self.map, key) if st["provenance"] == "real" else st["record"]
            self.events.append({
                "cell": key,
                "provenance": st["provenance"],
                "t": v["start"],
                # age (v2.1) = longest contiguous zero-overlap absence survived
                "age": v["gap_len"],
                "age_onscreen": v["start"] - v["prev_last"],   # diagnostic (v2.0 def)
                "color": int(color),
                "ref": int(ref),
                "correct": bool(color == ref),
                "weight": OUT_TILE_WEIGHT if ref == OUT_IDX else 1.0,
                "phase": "imag" if v["start"] >= self.prefix_len else "prefix",
            })
        st["record"] = int(color)
        st["visit"] = None


def corner_start(rng):
    """Start within 10 lattice steps of a random corner: the corner anchors ALL
    borders on the fixed-size board (Merlin)."""
    m = LATTICE - 1
    cr, cc = rng.integers(0, 2, size=2)
    orow = int(rng.integers(0, 10))
    ocol = int(rng.integers(0, 10))
    return (orow if cr == 0 else m - orow, ocol if cc == 0 else m - ocol)


def run_episode(adapter_factory, policy, map_seed, ep_seed,
                prefix_len=PREFIX_LEN, imag_len=IMAG_LEN, privileged=False):
    """One eval episode. Returns (events, fidelity_matches, first_imag_colors,
    band_abs_err, positions) — positions is the per-frame registration (true
    positions during the prefix, action path-integral during imagination).

    privileged=False (the DEFAULT, and the only mode the harness driver may use
    for candidate models): adapter_factory receives None — a model must work from
    prefix_frames/prefix_actions alone (red-team S4: handing out env is an
    instant-oracle hole). privileged=True is for the frozen baselines only."""
    rng = np.random.default_rng(ep_seed)
    env = ColorFieldEnv()
    frame = env.reset(seed=map_seed, start=corner_start(rng))
    adapter = adapter_factory(env if privileged else None)
    policy.reset(rng)
    tracker = CellTracker(env.map, prefix_len)

    # --- real prefix (teacher-forced): the policy sees real frames' bands -------
    frames = [frame]
    actions = [STAY]
    positions = [env.pos]
    tracker.observe(0, frame, env.pos, is_real=True)
    for t in range(1, prefix_len):
        a = policy.act(border_bands(frames[-1]), rng)
        if a not in env.valid_actions():   # frozen-layer bug if this ever fires
            raise AssertionError(f"eval policy emitted invalid action {a} at {env.pos}")
        frame = env.step(a)
        frames.append(frame)
        actions.append(a)
        positions.append(env.pos)
        tracker.observe(t, frame, env.pos, is_real=True)
    adapter.begin(np.stack(frames), np.asarray(actions, dtype=np.int64))

    # --- imagination phase: registration = path integral of taken actions ------
    pos = env.pos
    cur = frames[-1]
    fidelity = []
    band_err = []
    for t in range(prefix_len, prefix_len + imag_len):
        a = policy.act(border_bands(cur), rng)
        nxt = adapter.step(a)
        pos = apply_action(pos, a, check=False)   # may leave the true lattice: allowed
        dy, dx, _ = estimate_shift(cur, nxt)
        cdy, cdx = PITCH_PX * DELTAS[a][0], PITCH_PX * DELTAS[a][1]
        fidelity.append((dy, dx) == (cdy, cdx))
        band_err.append(_band_abs_err(nxt, pos))
        tracker.observe(t, nxt, pos, is_real=False)
        positions.append(pos)
        cur = nxt
    tracker.finalize()
    return tracker.events, fidelity, tracker.first_imag_colors, band_err, positions


def _band_abs_err(frame, pos):
    """Diagnostic: |imagined band - band the TRUE lattice would show at the
    path-integral position| averaged over sides (border-drift measure)."""
    b = border_bands(frame)
    tly = PITCH_PX * pos[0] + TL_OFFSET
    tlx = PITCH_PX * pos[1] + TL_OFFSET
    exp = {
        "up": max(0, -tly),
        "left": max(0, -tlx),
        "down": max(0, tly + VIEW_PX - WORLD_PX),
        "right": max(0, tlx + VIEW_PX - WORLD_PX),
    }
    return float(np.mean([abs(b[k] - min(exp[k], 64)) for k in b]))


def aggregate(events, fidelity, first_imag_colors,
              bin_edges=BIN_EDGES, min_events=MIN_EVENTS_PER_BIN,
              fidelity_threshold=FIDELITY_THRESHOLD):
    """Age-standardized components + gates + composite. Pure function of the
    collected statistics — the driver calls THIS for the loop's number."""
    edges = list(bin_edges) + [np.inf]
    imag_phase = [e for e in events if e["phase"] == "imag"]

    def component(prov):
        evs = [e for e in imag_phase if e["provenance"] == prov]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [e for e in evs if lo <= e["age"] < hi]
            # SCORED accuracy uses IN-MAP events only. OUT (border) tiles are pure
            # geometry — a zero-content-memory model gets them right at any age —
            # and they DOMINATE far bins where in-map events thin out (measured:
            # ~78% of bin [65,129) events on box-loop policies), so any nonzero
            # weight lets them set the long-range score. Border recall is reported
            # as a separate diagnostic instead. [Deviation from the original 0.1
            # weight — evidence in tests/test_eval.py::test_bounded_window_*]
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
                    "border_tiles": "excluded from score; border_recall diagnostic"},
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
            print(f"[eval] {name}: {per_policy[name]}", flush=True)

    result = aggregate(all_events, all_fid, all_colors, min_events=min_events)
    result["per_policy"] = per_policy
    result["border_drift_px"] = (float(np.mean(all_band_err)) if all_band_err else None)
    result["config"] = {"suite": [n for n, _ in suite], "n_seeds": n_seeds,
                        "prefix_len": prefix_len, "imag_len": imag_len,
                        "seed0": seed0, "bin_edges": list(BIN_EDGES),
                        "min_events_per_bin": min_events, "privileged": privileged}
    result["events"] = [dict(e, cell=list(e["cell"])) for e in all_events]
    return result


def save_json(result, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
