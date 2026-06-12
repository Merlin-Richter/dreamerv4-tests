"""EXP-010 presentation builder: FF7 arms vs the EXP-009 baseline on the frozen probe.

Reads results.json files (probe schema 5503e75), writes:
  comparison.md   — headline tables (color dRGB primary, latent-MSE secondary)
  comparison.html — self-contained SVG chart (no dependencies, open in any browser)

Run from repo root:
  python experiments/EXP-010/compare.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
BAR = 63.0  # T-004: H3 success = color dRGB below this at n_occ in {12,16,24}
TEST_POINTS = (12, 16, 24)

SERIES = [
    # (label, results.json path, css color)
    ("baseline (EXP-009)", HERE.parent / "EXP-009" / "results.json", "#888888"),
    ("FF7 k=1", HERE / "k1" / "results.json", "#d62728"),
    ("FF7 k=3", HERE / "k3" / "results.json", "#1f77b4"),
]


def load(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt_row(label, by_occ, grid, bold_points=()):
    cells = []
    for n in grid:
        v = by_occ.get(str(n))
        s = "-" if v is None else f"{v:.1f}"
        if n in bold_points and v is not None:
            s = f"**{s}**"
        cells.append(s)
    return f"| {label} | " + " | ".join(cells) + " |"


def svg_chart(series, grid, ceiling, chance, metric, title, lo, hi):
    W, H, ML, MB, MT, MR = 760, 420, 60, 50, 40, 20
    pw, ph = W - ML - MR, H - MT - MB

    def x(n):
        return ML + pw * grid.index(n) / (len(grid) - 1)

    def y(v):
        v = min(max(v, lo), hi)
        return MT + ph * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'style="background:#fff;font-family:sans-serif">',
             f'<text x="{W/2}" y="20" text-anchor="middle" font-size="15">{title}</text>']
    # axes + reference lines
    for v, lab, col in [(ceiling, "ceiling", "#2ca02c"), (chance, "chance", "#999"),
                        (BAR if metric == "color" else None, "T-004 bar", "#e377c2")]:
        if v is None:
            continue
        parts.append(f'<line x1="{ML}" y1="{y(v)}" x2="{W-MR}" y2="{y(v)}" '
                     f'stroke="{col}" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{W-MR-4}" y="{y(v)-4}" text-anchor="end" font-size="11" '
                     f'fill="{col}">{lab} {v:.0f}</text>')
    for n in grid:
        parts.append(f'<text x="{x(n)}" y="{H-MB+18}" text-anchor="middle" font-size="11">{n}</text>')
    parts.append(f'<text x="{W/2}" y="{H-8}" text-anchor="middle" font-size="12">n_occ '
                 f'(occluded frames; window N=8, prefix P=3)</text>')
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        parts.append(f'<text x="{ML-8}" y="{y(v)+4}" text-anchor="end" font-size="11">{v:.1f}</text>')
    parts.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H-MB}" stroke="#000"/>')
    parts.append(f'<line x1="{ML}" y1="{H-MB}" x2="{W-MR}" y2="{H-MB}" stroke="#000"/>')
    # series
    ylegend = MT + 14
    for label, by_occ, col, dash in series:
        pts = [(x(n), y(by_occ[str(n)])) for n in grid if str(n) in by_occ]
        if not pts:
            continue
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2" '
                     f'{"stroke-dasharray=\"4 3\"" if dash else ""}/>')
        for px, py in pts:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{col}"/>')
        parts.append(f'<text x="{ML+10}" y="{ylegend}" font-size="12" fill="{col}">'
                     f'{"- - " if dash else "—— "}{label}</text>')
        ylegend += 16
    parts.append("</svg>")
    return "".join(parts)


def main():
    loaded = [(lab, load(p), col) for lab, p, col in SERIES]
    missing = [lab for lab, d, _ in loaded if d is None]
    loaded = [(lab, d, col) for lab, d, col in loaded if d is not None]
    base = loaded[0][1]
    grid = base["occ_grid"]
    ceil_c = base["controls"]["ceiling"]["color_dRGB"]
    chance_c = base["controls"]["chance"]["color_dRGB"]
    ceil_m = base["controls"]["ceiling"]["latent_mse"]
    chance_m = base["controls"]["chance"]["latent_mse"]

    md = ["# EXP-010 — FF7 v1 vs baseline (frozen probe 5503e75)", ""]
    if missing:
        md.append(f"*Missing (not yet run): {', '.join(missing)}*\n")
    md.append(f"T-004 bar: color dRGB < {BAR:.0f} at n_occ in {TEST_POINTS} "
              f"(ceiling {ceil_c:.1f}, chance {chance_c:.1f}).\n")
    md.append("## Color dRGB at the reveal frame (HEADLINE)\n")
    md.append("| series | " + " | ".join(str(n) for n in grid) + " |")
    md.append("|" + "---|" * (len(grid) + 1))
    for lab, d, _ in loaded:
        md.append(fmt_row(lab + " occluded", d["color_dRGB_by_occ"], grid, TEST_POINTS))
        md.append(fmt_row(lab + " drift-ctrl", d["matched_horizon_drift"]["color_dRGB"], grid))
    md.append("\n## latent-token MSE (secondary)\n")
    md.append("| series | " + " | ".join(str(n) for n in grid) + " |")
    md.append("|" + "---|" * (len(grid) + 1))
    for lab, d, _ in loaded:
        md.append(fmt_row(lab + " occluded", d["latent_mse_by_occ"], grid))
        md.append(fmt_row(lab + " drift-ctrl", d["matched_horizon_drift"]["latent_mse"], grid))
    md.append("\n## Controls per series (own rollout path — checks base-quality tripwire D-014)\n")
    md.append("| series | ceiling dRGB | chance dRGB | ceiling MSE | chance MSE | "
              "ball_lost max | detector gate |")
    md.append("|---|---|---|---|---|---|---|")
    for lab, d, _ in loaded:
        c = d["controls"]
        lost = max(d["ball_lost_rate_by_occ"].values())
        md.append(f"| {lab} | {c['ceiling']['color_dRGB']:.1f} | {c['chance']['color_dRGB']:.1f} "
                  f"| {c['ceiling']['latent_mse']:.2f} | {c['chance']['latent_mse']:.2f} "
                  f"| {lost:.2f} | {'PASS' if d['detector_gate']['pass'] else 'FAIL'} |")
    (HERE / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    charts = []
    cs = [(lab, d["color_dRGB_by_occ"], col, False) for lab, d, col in loaded]
    cs += [(lab + " drift", d["matched_horizon_drift"]["color_dRGB"], col, True)
           for lab, d, col in loaded]
    charts.append(svg_chart(cs, grid, ceil_c, chance_c, "color",
                            "Hidden-color recall: dRGB at reveal (lower = better memory)",
                            0, max(chance_c * 1.2, 130)))
    ms = [(lab, d["latent_mse_by_occ"], col, False) for lab, d, col in loaded]
    ms += [(lab + " drift", d["matched_horizon_drift"]["latent_mse"], col, True)
           for lab, d, col in loaded]
    charts.append(svg_chart(ms, grid, ceil_m, chance_m, "mse",
                            "latent-token MSE at reveal (secondary)", 0, max(chance_m * 1.3, 1.1)))
    html = ("<!doctype html><meta charset='utf-8'><title>EXP-010 comparison</title>"
            "<body style='margin:24px'>" + "<br><br>".join(charts) + "</body>")
    (HERE / "comparison.html").write_text(html, encoding="utf-8")
    print(f"wrote {HERE/'comparison.md'} and {HERE/'comparison.html'}"
          + (f"  (missing: {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
