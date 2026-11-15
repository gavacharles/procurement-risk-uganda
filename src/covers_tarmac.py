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


TITLE3 = "Seven Years to Tarmac"
TAG3 = ("Uganda's road construction contracts are public. Their start dates, delays and disputes are not. "
        "I tracked them from tender to tarmac by satellite, against the government's own reports and the courts")


def wait_svg(width, height, f):
    """Each contracted road a bar from tender to the last image: orange while gravel, turning grey as the
    satellite sees it sealed (new sealing since the first image); white tick at completion (>= 80% sealed)."""
    R = pd.read_parquet(P / "road_progress_roads.parquet")
    S = pd.read_parquet(P / "road_progress_series.parquet")
    R = R[(R.kind == "upgrade") & R.t_tender.notna()
          & ~R.status.str.startswith(("excluded", "insufficient", "sealed by the first"))].sort_values("t_tender")
    x0, x1 = 2014.5, 2026.9
    lab_w = 0.0
    pad_l, pad_b = 6 * f, 26 * f
    W, H = width - pad_l, height - pad_b
    X = lambda t: pad_l + (t - x0) / (x1 - x0) * W  # noqa: E731
    n = len(R)
    rh = H / n
    bh = rh * 0.62

    def mix(u):
        u = max(0.0, min(1.0, u))
        a, b = (0xd9, 0x87, 0x3a), (0x9a, 0xa1, 0xab)
        return "#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * u) for k in range(3))
    out = []
    for yr in range(2015, 2027):
        out.append(f'<line x1="{X(yr):.1f}" x2="{X(yr):.1f}" y1="0" y2="{H:.1f}" stroke="#262b32" stroke-width="1"/>')
        if yr % 2 == 1 or yr == 2026:
            out.append(f'<text x="{X(yr):.1f}" y="{H + 18 * f:.1f}" fill="#8a9099" font-size="{11.5 * f:.1f}" '
                       f'text-anchor="middle" font-family="Inter">{yr}</text>')
    out.append(f'<line x1="{X(2017.0):.1f}" x2="{X(2017.0):.1f}" y1="0" y2="{H:.1f}" stroke="#4a515b" stroke-dasharray="3 3"/>')
    for i, r in enumerate(R.itertuples()):
        y = i * rh + (rh - bh) / 2
        ser = S[S.id == r.id].sort_values("t")
        s0 = float(ser.sealed_share.iloc[0]) if len(ser) else 0.0
        t_start = r.t_tender
        # before the archive: dashed thin bar (no images)
        if t_start < 2017.0:
            out.append(f'<rect x="{X(t_start):.1f}" y="{y + bh * 0.3:.1f}" width="{X(2017.0) - X(t_start):.1f}" '
                       f'height="{bh * 0.4:.1f}" rx="{bh * 0.2:.1f}" fill="#d9873a" opacity="0.45"/>')
        pts = list(zip(ser.t, ser.sealed_share))
        for (ta, sa), (tb, _) in zip(pts, pts[1:] + [(2026.9, None)]):
            a, b = max(ta - 0.25, t_start, 2017.0), min(tb - 0.25, 2026.9)
            if b <= a:
                continue
            u = ((sa - s0) / (1 - s0) if s0 < 0.95 else 1) / 0.8  # 80% sealed = completion = full grey
            out.append(f'<rect x="{X(a):.1f}" y="{y:.1f}" width="{X(b) - X(a) + 0.6:.1f}" height="{bh:.1f}" fill="{mix(u)}"/>')
        out.append(f'<circle cx="{X(t_start):.1f}" cy="{y + bh / 2:.1f}" r="{bh * 0.38:.1f}" fill="#f0efec"/>')
        if pd.notna(r.t_done):
            out.append(f'<rect x="{X(r.t_done) - 1.5 * f:.1f}" y="{y - 2 * f:.1f}" width="{3 * f:.1f}" height="{bh + 4 * f:.1f}" fill="#ffffff"/>')
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(out)}</svg>'


def cover3(w, h, f, km_median, pre_median, rho):
    stat = lambda big, col, lines: (f'<div style="font-size:{13.5 * f}px;line-height:1.35"><b style="font-size:{32 * f}px;color:{col}">{big}</b>'  # noqa: E731
                                    f'<br>{lines}</div>')
    sw = lambda c, lab, shape="bar": (f'<span style="display:inline-flex;align-items:center;gap:{6 * f}px;margin-right:{14 * f}px">'  # noqa: E731
                                      + (f'<i style="display:inline-block;width:{9 * f}px;height:{9 * f}px;border-radius:50%;background:{c}"></i>' if shape == "dot"
                                         else f'<i style="display:inline-block;width:{3 * f}px;height:{12 * f}px;background:{c}"></i>' if shape == "tick"
                                         else f'<i style="display:inline-block;width:{22 * f}px;height:{9 * f}px;border-radius:2px;background:{c}"></i>')
                                      + f'{lab}</span>')
    leg = (f'<div class="muted" style="font-size:{11.5 * f}px;margin-top:{8 * f}px;display:flex;flex-wrap:wrap">'
           + sw("#f0efec", "tender", "dot") + sw("#d9873a", "still gravel") + sw("linear-gradient(90deg,#d9873a,#9aa1ab)", "being sealed")
           + sw("#9aa1ab", "sealed") + sw("#ffffff", "finished", "tick") + '</div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">{BASE_CSS}</head><body style="width:{w}px;height:{h}px">
<div style="position:absolute;left:{56 * f}px;top:{50 * f}px;width:{w * 0.44}px">
  <div class="kicker" style="font-size:{14 * f}px">Construction procurement · delays · disputes</div>
  <h1 style="font-size:{60 * f}px;margin-top:{14 * f}px">{TITLE3}</h1>
  <p class="muted" style="font-size:{18 * f}px;margin-top:{16 * f}px;line-height:1.42">{TAG3}</p>
</div>
<div style="position:absolute;left:{56 * f}px;bottom:{44 * f}px;display:flex;align-items:flex-end;gap:{30 * f}px">
  {stat(f"{km_median:.1f} years", "#f08a5d", "median from tender<br>to a finished road")}
  {stat(f"{pre_median:.1f} years", "#f08a5d", "before works<br>even start")}
  {stat(f"{rho:.2f}", "#86b6ef", "correlation with the<br>government's progress reports")}
</div>
<div style="position:absolute;right:{44 * f}px;top:{46 * f}px">
  <div class="muted" style="font-size:{12 * f}px;margin-bottom:{8 * f}px">19 contracted roads, tender → today, as seen by Sentinel-2</div>
  {wait_svg(int(w * 0.44), int(h * 0.66), f)}{leg}
</div>
<div style="position:absolute;right:{44 * f}px;bottom:{30 * f}px;text-align:right;font-size:{14 * f}px">
  <div class="by">By <b>Gava Charles</b></div><div class="url">gavacharles.github.io/procurement-risk-uganda</div></div>
</body></html>"""


LIGHT_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:wght@600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f6f2ea;color:#1b1f24;font-family:Inter,system-ui,sans-serif;overflow:hidden}
.kicker{color:#b4442c;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
h1{font-family:"Source Serif 4",Georgia,serif;font-weight:700;line-height:1.04;letter-spacing:-.01em}
.muted{color:#5b6570}
.by{color:#3a4049;font-weight:600}.by b{color:#1b1f24}
.url{color:#2c5f8a}
</style>"""
GRAVEL_L, TARMAC_L = (0xe0, 0x8a, 0x3c), (0x2b, 0x30, 0x37)
DISPUTED = {"tirinyi", "kamonkoli", "muyembe", "bukakata", "kyenjojo"}  # data/manual/road_disputes_checked.csv (relevant, yes)
SHORT = {"tirinyi": "Tirinyi–Pallisa–Kumi", "kamonkoli": "Pallisa–Kamonkoli", "bumbobi": "Bumbobi–Lwakhakha",
         "muyembe": "Muyembe–Nakapiripirit", "kigumba": "Kigumba–Kabwoya", "rwenkunye": "Rwenkunye–Puranga",
         "bukakata": "Masaka–Bukakata", "hoima": "Hoima–Wanseko", "kalule": "Kalule–Bamunanika", "koboko": "Koboko–Yumbe–Moyo",
         "luwero": "Luwero–Butalangu", "namagumba": "Namagumba–Budadiri", "kyenjojo": "Kyenjojo–Bwizi",
         "senge": "Kayunga–Senge", "namulanda": "Namulanda–Kajjansi", "kisozi": "Kisozi–Kitemu", "katooke": "Nansana–Katooke",
         "bududa": "Bubulo–Bududa", "kalangala": "Kalangala–Mulabana"}


def cover_roads():
    """Upgrade roads with a verdict and >= 8 readable 1 km segments (shorter roads are too noisy to colour)."""
    R = pd.read_parquet(P / "road_progress_roads.parquet")
    return R[(R.kind == "upgrade") & R.t_tender.notna() & (R.n_segs >= 8)
             & ~R.status.str.startswith(("excluded", "insufficient", "sealed by the first"))].sort_values("t_tender")


def wait_svg_light(width, height, f):
    R = cover_roads()
    S = pd.read_parquet(P / "road_progress_series.parquet")
    x0, x1 = 2014.5, 2026.75
    lab = 150 * f
    W, H = width - lab - 22 * f, height - 30 * f
    X = lambda t: lab + (t - x0) / (x1 - x0) * W  # noqa: E731
    rh = H / len(R)
    bh = rh * 0.6
    fs = min(11.5 * f, rh * 0.62)

    def mix(u):
        u = max(0.0, min(1.0, u))
        return "#%02x%02x%02x" % tuple(round(GRAVEL_L[k] + (TARMAC_L[k] - GRAVEL_L[k]) * u) for k in range(3))
    o = []
    for yr in range(2015, 2027):
        o.append(f'<line x1="{X(yr):.1f}" x2="{X(yr):.1f}" y1="0" y2="{H:.1f}" stroke="#e4ded2" stroke-width="1"/>')
        if yr % 2 == 1:
            o.append(f'<text x="{X(yr):.1f}" y="{H + 19 * f:.1f}" fill="#7a8089" font-size="{11.5 * f:.1f}" text-anchor="middle" font-family="Inter">{yr}</text>')
    o.append(f'<text x="{X(2026.6):.1f}" y="{H + 19 * f:.1f}" fill="#b4442c" font-size="{11.5 * f:.1f}" font-weight="700" text-anchor="middle" font-family="Inter">today</text>')
    o.append(f'<text x="{X(2015.75):.1f}" y="{H + 19 * f:.1f}" fill="#a3a8af" font-size="{10 * f:.1f}" text-anchor="middle" font-family="Inter">before images</text>' if False else "")
    for i, r in enumerate(R.itertuples()):
        y = i * rh + (rh - bh) / 2
        cy = y + bh / 2
        o.append(f'<text x="{lab - 10 * f:.1f}" y="{cy + fs * 0.36:.1f}" fill="#3a4049" font-size="{fs:.1f}" text-anchor="end" font-family="Inter">{SHORT.get(r.id, r.road)}</text>')
        ser = S[S.id == r.id].sort_values("t")
        s0 = float(ser.sealed_share.iloc[0]) if len(ser) else 0.0
        t0 = r.t_tender
        if t0 < 2017.0:
            o.append(f'<line x1="{X(t0):.1f}" x2="{X(2017.0):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="#e08a3c" stroke-width="{bh * 0.35:.1f}" stroke-dasharray="{3 * f:.0f} {3 * f:.0f}" opacity="0.7"/>')
        pts = list(zip(ser.t, ser.sealed_share))
        for (ta, sa), (tb, _) in zip(pts, pts[1:] + [(x1, None)]):
            a, b = max(ta - 0.25, t0, 2017.0), min(tb - 0.25, x1)
            if b > a:
                u = ((sa - s0) / (1 - s0) if s0 < 0.95 else 1) / 0.8
                o.append(f'<rect x="{X(a):.1f}" y="{y:.1f}" width="{X(b) - X(a) + 0.6:.1f}" height="{bh:.1f}" fill="{mix(u)}"/>')
        o.append(f'<circle cx="{X(t0):.1f}" cy="{cy:.1f}" r="{bh * 0.42:.1f}" fill="#ffffff" stroke="#1b1f24" stroke-width="{1.6 * f:.1f}"/>')
        if pd.notna(r.t_done):
            o.append(f'<rect x="{X(r.t_done) - 1.6 * f:.1f}" y="{y - 2.5 * f:.1f}" width="{3.2 * f:.1f}" height="{bh + 5 * f:.1f}" fill="#f6f2ea"/>')
        if r.id in DISPUTED:
            o.append(f'<text x="{X(x1) + 6 * f:.1f}" y="{cy + fs * 0.55:.1f}" font-size="{fs * 1.55:.1f}" fill="#b4442c" font-weight="700">⚖</text>')
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(o)}</svg>'


def cover4(w, h, f, km_median, pre_median):
    n = len(cover_roads())
    nd = int(cover_roads().id.isin(DISPUTED).sum())
    stat = lambda big, lines: (f'<div style="display:flex;align-items:center;gap:{12 * f}px;margin-top:{12 * f}px">'  # noqa: E731
                               f'<b style="font-size:{30 * f}px;color:#b4442c;min-width:{118 * f}px;font-weight:800">{big}</b>'
                               f'<span style="font-size:{14 * f}px;line-height:1.3;color:#3a4049">{lines}</span></div>')
    key = lambda sw, lab: f'<span style="display:inline-flex;align-items:center;gap:{6 * f}px;margin-right:{16 * f}px">{sw}{lab}</span>'  # noqa: E731
    bar = lambda bg: f'<i style="display:inline-block;width:{22 * f}px;height:{9 * f}px;border-radius:2px;background:{bg}"></i>'  # noqa: E731
    legend = ('<div class="muted" style="font-size:{fs}px;margin-top:{m}px;display:flex;flex-wrap:wrap;padding-left:{pl}px">'.format(
        fs=11.5 * f, m=6 * f, pl=150 * f)
        + key(f'<i style="display:inline-block;width:{9 * f}px;height:{9 * f}px;border-radius:50%;background:#fff;border:{1.6 * f}px solid #1b1f24"></i>', "tender")
        + key(bar("#e08a3c"), "gravel") + key(bar("linear-gradient(90deg,#e08a3c,#2b3037)"), "being sealed")
        + key(bar("#2b3037"), "tarmac") + key(f'<span style="color:#b4442c;font-size:{15 * f}px;font-weight:700">⚖</span>', "formal dispute on record") + "</div>")
    return f"""<!doctype html><html><head><meta charset="utf-8">{LIGHT_CSS}</head><body style="width:{w}px;height:{h}px">
<div style="position:absolute;left:{56 * f}px;top:{52 * f}px;width:{w * 0.36}px">
  <div class="kicker" style="font-size:{(13.5 if w > 1300 else 12) * f}px;white-space:nowrap">Construction procurement · delays · disputes</div>
  <h1 style="font-size:{58 * f}px;margin-top:{16 * f}px">Seven Years to Tarmac</h1>
  <p style="font-size:{18.5 * f}px;margin-top:{16 * f}px;line-height:1.4;color:#3a4049">Uganda's road contracts are public. When the roads get built, and why they don't, is not.</p>
  <div style="margin-top:{26 * f}px;border-top:1px solid #ddd5c6;padding-top:{6 * f}px">
    {stat(f"{km_median:.1f} yrs", "median from tender<br>to a finished road")}
    {stat(f"{pre_median:.0f} yrs", "before works even start:<br>land, money, approvals")}
    {stat(f"{nd} of {n}", "roads with a formal dispute in<br>court, tribunal or arbitration")}
  </div>
</div>
<div style="position:absolute;right:{40 * f}px;top:{50 * f}px">
  <div style="font-size:{13 * f}px;color:#3a4049;margin-bottom:{10 * f}px;padding-left:{150 * f}px"><b>{n} road contracts, tender to today</b> <span class="muted">· seen from space, Sentinel-2</span></div>
  {wait_svg_light(int(w * 0.56), int(h * 0.70), f)}{legend}
</div>
<div style="position:absolute;left:{56 * f}px;bottom:{34 * f}px;font-size:{13.5 * f}px">
  <span class="by">By <b>Gava Charles</b></span> <span class="muted">·</span> <span class="url">gavacharles.github.io/procurement-risk-uganda</span></div>
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
    if "--v2" not in sys.argv:
        rho = json.loads((ROOT / "paper" / "progress_validation.json").read_text())["spearman"]["rho"]
        shoot(cover4(1456, 816, 1.2, km_median, pre_median), "cover_substack_tarmac.png", 1456, 816)
        shoot(cover4(1200, 627, 1.0, km_median, pre_median), "cover_linkedin_tarmac.png", 1200, 627)
        shoot(cover3(1456, 816, 1.2, km_median, pre_median, rho), "cover_substack_tarmac_dark.png", 1456, 816)
        shoot(cover3(1200, 627, 1.0, km_median, pre_median, rho), "cover_linkedin_tarmac_dark.png", 1200, 627)
        return
    e = json.loads((ROOT / "paper" / "launch_timing.json").read_text())["election_tests"][0]
    cen = json.loads((ROOT / "paper" / "paving_census.json").read_text())
    shoot(cover2(1456, 816, 1.2, km_median, e, cen), "cover_substack_tarmac_v2.png", 1456, 816)
    shoot(cover2(1200, 627, 1.0, km_median, e, cen), "cover_linkedin_tarmac_v2.png", 1200, 627)


if __name__ == "__main__":
    main()
