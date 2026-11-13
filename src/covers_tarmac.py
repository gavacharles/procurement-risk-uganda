"""Covers for the paper 3 Substack post ("Tender to Tarmac"), in the style of src/covers.py:
dark page, orange kicker, serif headline, method chips, a small data graphic with a big number, a Uganda map.

Out (posts/media/): cover_substack_tarmac.png (1456 x 816 @2x), cover_linkedin_tarmac.png (1200 x 627 @2x)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from covers import BASE_CSS, shoot, OUT  # noqa: E402
import road_progress as rp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
GRAVEL, ASPH, VEG, NOD = "#d9873a", "#8b929c", "#5f8f5e", "#2a2f36"
STATUS = {"completed": ("#3987e5", "completed"), "substantial": ("#3987e5", "completed"),
          "sealed": ("#3987e5", "completed"), "incomplete": ("#f08a5d", "overdue, being paved"),
          "nopaving": ("#e34948", "no paving past due"), "progress": ("#b9bcc2", "not yet due"),
          "control": ("#6b7280", "rebuilding a paved road"), "insufficient": ("#4a515b", "no verdict")}


def skey(status):
    for pre, k in [("completed", "completed"), ("substantially", "substantial"), ("incomplete", "incomplete"),
                   ("no paving detected past", "nopaving"), ("control", "control"), ("rehabilitation", "control"),
                   ("sealed", "sealed"), ("insufficient", "insufficient")]:
        if str(status).startswith(pre):
            return k
    return "progress"


def map_svg(width, height, R):
    g = json.loads((ROOT / "docs" / "data" / "districts.geojson").read_text())
    lon0, lon1, lat0, lat1 = 29.5, 35.1, -1.55, 4.3
    s = min(width / (lon1 - lon0), height / (lat1 - lat0))

    def pt(c):
        return f"{(c[0] - lon0) * s:.1f},{(lat1 - c[1]) * s:.1f}"
    paths = []
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        d = "".join("M" + "L".join(pt(c) for c in ring) + "Z" for poly in polys for ring in poly)
        paths.append(f'<path d="{d}" fill="#23282f" stroke="#14171b" stroke-width="0.7"/>')
    # roads: thinner first, so long roads draw on top
    for r in R.sort_values("contract_km").itertuples():
        road = next((x for x in rp.ROADS if x[0] == r.id), None)
        geo = rp.geometry(road) if road else None
        if not geo or str(r.status).startswith("excluded"):
            continue
        col = STATUS[skey(r.status)][0]
        d = "M" + "L".join(pt(c) for c in geo["coords"][::2])
        w = 2.2 + min(r.contract_km, 140) / 32
        paths.append(f'<path d="{d}" fill="none" stroke="#14171b" stroke-width="{w + 2.5:.1f}" stroke-linecap="round" stroke-linejoin="round"/>')
        paths.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w:.1f}" stroke-linecap="round" stroke-linejoin="round"/>')
    return f'<svg viewBox="0 0 {(lon1 - lon0) * s:.0f} {(lat1 - lat0) * s:.0f}" width="{width}" height="{height}">{"".join(paths)}</svg>'


def strip_svg(width, height, G, rid="hoima"):
    order = [c for c in G.columns if len(c) == 5 and c[:4].isdigit()]
    g = G[G.id == rid].sort_values("seg")
    code = {"unpaved": GRAVEL, "asphalt": ASPH, "vegetated": VEG}
    nx, ny = len(order), len(g)
    cw, ch = width / nx, height / ny
    rects = []
    for i, c in enumerate(order):
        for j, v in enumerate(g[c].tolist()):
            rects.append(f'<rect x="{i * cw:.1f}" y="{height - (j + 1) * ch:.1f}" width="{cw - 0.6:.1f}" height="{ch - 0.6:.1f}" fill="{code.get(v, NOD)}"/>')
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(rects)}</svg>'


def census_svg(width, height):
    """National paving census: all major roads faint; newly paved points, traced (blue) or not (orange)."""
    import paving_census as pc
    g = json.loads((ROOT / "docs" / "data" / "districts.geojson").read_text())
    lon0, lon1, lat0, lat1 = 29.5, 35.1, -1.55, 4.3
    s = min(width / (lon1 - lon0), height / (lat1 - lat0))

    def pt(c):
        return f"{(c[0] - lon0) * s:.1f},{(lat1 - c[1]) * s:.1f}"
    out = []
    for ft in g["features"]:
        geom = ft["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        d = "".join("M" + "L".join(pt(c) for c in ring) + "Z" for poly in polys for ring in poly)
        out.append(f'<path d="{d}" fill="#23282f" stroke="#14171b" stroke-width="0.7"/>')
    keep = set(pd.read_parquet(P / "paving_census_points.parquet", columns=["way"]).way)
    for w in pc.osm_major():
        if w["id"] in keep and len(w["geom"]) > 1:
            gg = w["geom"][::3] + [w["geom"][-1]]
            out.append(f'<path d="M{"L".join(pt(c) for c in gg)}" fill="none" stroke="#3a414b" stroke-width="1"/>')
    st = pd.read_parquet(P / "paving_census_stretches.parquet")
    new = pd.read_parquet(P / "paving_census_newly_paved.parquet")
    new = new[new.stretch >= 0]
    traced = set(st.index[st.matched_any])
    for r in new.itertuples():
        x, y = (r.lon - lon0) * s, (lat1 - r.lat) * s
        col = "#3987e5" if r.stretch in traced else "#f08a5d"
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(3.2, width / 210):.1f}" fill="{col}"/>')
    return f'<svg viewBox="0 0 {(lon1 - lon0) * s:.0f} {(lat1 - lat0) * s:.0f}" width="{width}" height="{height}">{"".join(out)}</svg>'


def cover2(w, h, f, km_median, launches, cen):
    """Second edition: adds launch timing and the national census."""
    chips = ["Sentinel-2", "Machine learning", "OpenStreetMap", "Survival analysis", "News checks"]
    e = launches
    wp = cen["with_places"]
    stat = lambda big, col, lines: (f'<div style="font-size:{14 * f}px;line-height:1.35"><b style="font-size:{34 * f}px;color:{col}">{big}</b>'
                                    f'<br>{lines}</div>')
    leg = (f'<div class="muted" style="font-size:{12.5 * f}px;display:flex;gap:{16 * f}px;margin-top:{6 * f}px">'
           f'<span><i style="display:inline-block;width:{10 * f}px;height:{10 * f}px;border-radius:50%;background:#3987e5;margin-right:6px"></i>new tarmac, contract published</span>'
           f'<span><i style="display:inline-block;width:{10 * f}px;height:{10 * f}px;border-radius:50%;background:#f08a5d;margin-right:6px"></i>no contract in the open data</span></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">{BASE_CSS}</head><body style="width:{w}px;height:{h}px">
<div style="position:absolute;left:{56 * f}px;top:{50 * f}px;width:{w * 0.55}px">
  <div class="kicker" style="font-size:{15 * f}px">Uganda · public works · satellite audit</div>
  <h1 style="font-size:{66 * f}px;margin-top:{14 * f}px">Tender to Tarmac</h1>
  <p class="muted" style="font-size:{19.5 * f}px;margin-top:{14 * f}px;line-height:1.4">I watched 22 contracted roads, then every major road in Uganda, from space, and checked every verdict against the news</p>
  <div style="margin-top:{20 * f}px;font-size:{13 * f}px">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div>
</div>
<div style="position:absolute;left:{56 * f}px;bottom:{44 * f}px;display:flex;align-items:flex-end;gap:{34 * f}px">
  {stat(f"{km_median:.1f} years", "#86b6ef", "median from tender<br>to a finished road")}
  {stat(f"{e['within_12m']} of {e['n']}", "#f08a5d", "road launches in the<br>year before an election")}
  {stat(f"{wp['unmatched_km']:,.0f} km", "#f08a5d", "of new tarmac with no<br>published contract")}
</div>
<div style="position:absolute;right:{40 * f}px;top:{38 * f}px">{census_svg(int(w * 0.36), int(h * 0.70))}{leg}</div>
<div style="position:absolute;right:{44 * f}px;bottom:{30 * f}px;text-align:right;font-size:{14 * f}px">
  <div class="by">By <b>Gava Charles</b></div><div class="url">gavacharles.github.io/procurement-risk-uganda</div></div>
</body></html>"""


def legend(f):
    seen, items = set(), []
    for k in ["completed", "incomplete", "nopaving", "progress", "control"]:
        col, lab = STATUS[k]
        if lab in seen:
            continue
        seen.add(lab)
        items.append(f'<span style="display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0">'
                     f'<i style="display:inline-block;width:{22 * f}px;height:{5 * f}px;border-radius:3px;background:{col}"></i>{lab}</span>')
    return f'<div class="muted" style="font-size:{12.5 * f}px;display:flex;flex-wrap:wrap;max-width:{430 * f}px">{"".join(items)}</div>'


def cover(w, h, f, R, G, km_median, pre_median):
    chips = ["Sentinel-2", "Machine learning", "OpenStreetMap", "Survival analysis", "News checks"]
    return f"""<!doctype html><html><head><meta charset="utf-8">{BASE_CSS}</head><body style="width:{w}px;height:{h}px">
<div style="position:absolute;left:{56 * f}px;top:{50 * f}px;width:{w * 0.55}px">
  <div class="kicker" style="font-size:{15 * f}px">Uganda · public works · satellite audit</div>
  <h1 style="font-size:{66 * f}px;margin-top:{14 * f}px">Tender to Tarmac</h1>
  <p class="muted" style="font-size:{19.5 * f}px;margin-top:{14 * f}px;line-height:1.4">I watched 22 contracted roads from space to see whether, and when, they were built, then checked every verdict against the news</p>
  <div style="margin-top:{20 * f}px;font-size:{13 * f}px">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div>
</div>
<div style="position:absolute;left:{56 * f}px;bottom:{40 * f}px;display:flex;align-items:flex-end;gap:{22 * f}px">
  <div>
    <div class="muted" style="font-size:{11 * f}px;margin-bottom:{5 * f}px">Hoima–Wanseko, every km, 2017 → 2026</div>
    {strip_svg(int(210 * f), int(104 * f), G)}
  </div>
  <div style="font-size:{14.5 * f}px;line-height:1.35"><b style="font-size:{36 * f}px;color:#f08a5d">{km_median:.1f} years</b><br>median from tender<br>to a finished road</div>
  <div style="font-size:{14.5 * f}px;line-height:1.35;margin-left:{6 * f}px"><b style="font-size:{36 * f}px;color:#86b6ef">{pre_median:.1f} years</b><br>median wait before<br>works even start</div>
</div>
<div style="position:absolute;right:{40 * f}px;top:{38 * f}px">{map_svg(int(w * 0.36), int(h * 0.70), R)}
  <div style="margin-top:{4 * f}px">{legend(f)}</div></div>
<div style="position:absolute;right:{44 * f}px;bottom:{30 * f}px;text-align:right;font-size:{14 * f}px">
  <div class="by">By <b>Gava Charles</b></div><div class="url">gavacharles.github.io/procurement-risk-uganda</div></div>
</body></html>"""


def main():
    R = pd.read_parquet(P / "road_progress_roads.parquet")
    G = pd.read_parquet(P / "road_progress_segments.parquet")
    js = json.loads((ROOT / "paper" / "road_progress.json").read_text())
    km_median = js["kaplan_meier"]["median_years"]
    L = R[R.t_launch.notna() & ~R.kind.str.startswith("control")]
    pre_median = float((L.t_launch - L.t_tender).median())
    OUT.mkdir(parents=True, exist_ok=True)
    if "--v1" in sys.argv:
        shoot(cover(1456, 816, 1.2, R, G, km_median, pre_median), "cover_substack_tarmac_v1.png", 1456, 816)
        shoot(cover(1200, 627, 1.0, R, G, km_median, pre_median), "cover_linkedin_tarmac_v1.png", 1200, 627)
        return
    e = json.loads((ROOT / "paper" / "launch_timing.json").read_text())["election_tests"][0]
    cen = json.loads((ROOT / "paper" / "paving_census.json").read_text())
    shoot(cover2(1456, 816, 1.2, km_median, e, cen), "cover_substack_tarmac.png", 1456, 816)
    shoot(cover2(1200, 627, 1.0, km_median, e, cen), "cover_linkedin_tarmac.png", 1200, 627)


if __name__ == "__main__":
    main()
