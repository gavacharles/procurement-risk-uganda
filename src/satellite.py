"""(4) Satellite verification pilot: was the road actually paved?

For road contracts whose titles name both endpoints ("Upgrading of A - B road ... to paved"):
  1. Geocode endpoints (OpenStreetMap Nominatim, 1 request/s, identified User-Agent).
  2. Route between them on the road network (OSRM public server) -> road geometry.
  3. Sample points every ~2 km along the route (cap 40) plus background points 60 m to each side.
  4. For each dry season (Dec-Feb) 2017/18 ... latest, take the least-cloudy Sentinel-2 L2A scene(s)
     (Element84 Earth Search, AWS open data), read B02 (blue), B04 (red) at each point, mask
     clouds/shadows with the SCL band.
  5. Signal: red laterite gravel is reddish (B04/B02 high); asphalt is grey (B04/B02 ~ 1).
     Road redness contrast = median over points of [ (B04/B02)_road - (B04/B02)_background ].
     Paving shows as a drop in redness contrast after the contract.
  6. Cross-check: OpenStreetMap surface tags along the route today (Overpass).

Controls: a 2025 tender (should not be paved yet) and rehabilitations of already-paved
roads (no gravel->asphalt change expected).

This is a pilot: 10 m pixels, mixed road/verge pixels and routing errors add noise. A
"no change" result is a prompt for a field check, not evidence of a ghost project.

Out: data/processed/satellite_roads.parquet, paper/satellite.json, posts/media/chart_satellite.png
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
CACHE = Path.home() / ".cache" / "procurement-risk" / "satellite"  # outside cloud-synced folder
UA = "procurement-risk-research (github.com/gavacharles/procurement-risk-uganda)"
STAC = "https://earth-search.aws.element84.com/v1"

# (label, endpoint A, endpoint B, kind, contract year, approx km)  -- from PPDA works titles
ROADS = [
    ("Hoima–Butiaba–Wanseko", "Hoima, Uganda", "Wanseko, Uganda", "upgrade", 2017, 111),
    ("Koboko–Yumbe–Moyo", "Koboko, Uganda", "Moyo, Uganda", "upgrade", 2021, 103),
    ("Muyembe–Nakapiripirit", "Muyembe, Uganda", "Nakapiripirit, Uganda", "upgrade", 2015, 92),
    ("Kigumba–Bulima–Kabwoya", "Kigumba, Uganda", "Kabwoya, Uganda", "upgrade", 2015, 135),
    ("Bumbobi–Lwakhakha", "Bumbobi, Uganda", "Lwakhakha, Uganda", "upgrade", 2015, 44.5),
    ("Namagumba–Budadiri–Nalugugu", "Namagumba, Uganda", "Budadiri, Uganda", "upgrade", 2022, 39),
    ("Luwero–Butalangu", "Luwero, Uganda", "Butalangu, Uganda", "upgrade", 2022, 30),
    ("Kyenjojo–Bwizi–Rwamwanja", "Kyenjojo, Uganda", "Rwamwanja, Uganda", "upgrade", 2023, 105),
    ("Rushere–Nshwerenkye", "Rushere, Uganda", "Nshwerenkye, Uganda", "upgrade", 2015, 10),
    ("Kashenyi–Mitooma", "Kashenyi, Uganda", "Mitooma, Uganda", "upgrade", 2015, 11.6),
    ("Kayunga–Senge–Kawanda (2025 tender)", "Senge, Wakiso, Uganda", "Kawanda, Uganda", "control_recent", 2025, 6.5),
    ("Ishaka–Katunguru (rehab, already paved)", "Ishaka, Uganda", "Katunguru, Uganda", "control_paved", 2016, 58),
    ("Fort Portal–Hima (rehab, already paved)", "Fort Portal, Uganda", "Hima, Uganda", "control_paved", 2016, 55),
]
SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]  # season Y = Dec(Y-1)..Feb(Y)


def http_json(url: str, data: bytes | None = None, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Content-Type": "application/json"} if data else {"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def cached(name: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{re.sub(r'[^a-z0-9]+', '_', name.lower())}.json"
    if f.exists():
        return json.loads(f.read_text())
    v = fn()
    f.write_text(json.dumps(v))
    return v


def geocode(q: str):
    def go():
        time.sleep(1.1)
        u = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": q, "format": "json", "limit": 1, "countrycodes": "ug"})
        r = http_json(u)
        return [float(r[0]["lon"]), float(r[0]["lat"])] if r else None
    return cached("geo " + q, go)


def route(a, b):
    def go():
        u = f"https://router.project-osrm.org/route/v1/driving/{a[0]},{a[1]};{b[0]},{b[1]}?overview=full&geometries=geojson"
        r = http_json(u)
        rt = r["routes"][0]
        return dict(coords=rt["geometry"]["coordinates"], km=rt["distance"] / 1000)
    return cached(f"route {a} {b}", go)


def sample_points(coords, step_km=2.0, cap=40):
    pts, acc = [], 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        seg = math.hypot((x1 - x0) * 111.32 * math.cos(math.radians(y0)), (y1 - y0) * 110.57)
        acc += seg
        if acc >= step_km:
            acc = 0
            dx, dy = x1 - x0, y1 - y0
            norm = math.hypot(dx, dy) or 1
            nx, ny = -dy / norm, dx / norm  # perpendicular unit (degrees)
            off = 60 / 111_000  # ~60 m
            pts.append(dict(road=(x1, y1), bg1=(x1 + nx * off, y1 + ny * off), bg2=(x1 - nx * off, y1 - ny * off)))
    if len(pts) > cap:
        idx = np.linspace(0, len(pts) - 1, cap).astype(int)
        pts = [pts[i] for i in idx]
    return pts


def scenes(bbox, season):
    def go():
        body = json.dumps({"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 50,
                           "datetime": f"{season - 1}-12-01T00:00:00Z/{season}-02-28T23:59:59Z",
                           "query": {"eo:cloud_cover": {"lt": 40}}}).encode()
        r = http_json(f"{STAC}/search", data=body)
        return [dict(id=f["id"], cloud=f["properties"]["eo:cloud_cover"], tile=f["properties"].get("grid:code"),
                     blue=f["assets"]["blue"]["href"], red=f["assets"]["red"]["href"], scl=f["assets"]["scl"]["href"])
                for f in r.get("features", [])]
    return cached(f"stac {bbox} {season}", go)


def read_points(scene, pts_lonlat):
    import rasterio
    from pyproj import Transformer
    out = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="3"):
        with rasterio.open(scene["blue"]) as src:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xy = [tr.transform(*p) for p in pts_lonlat]
            inside = [src.bounds.left < x < src.bounds.right and src.bounds.bottom < y < src.bounds.top for x, y in xy]
            if not any(inside):
                return None
            out["blue"] = np.array([v[0] for v in src.sample(xy)], dtype=float)
        with rasterio.open(scene["red"]) as src:
            out["red"] = np.array([v[0] for v in src.sample(xy)], dtype=float)
        with rasterio.open(scene["scl"]) as src:
            out["scl"] = np.array([v[0] for v in src.sample(xy)], dtype=float)
    out["inside"] = np.array(inside)
    return out


def season_contrast(pts, sc_list):
    """Median road-minus-background redness for one season, combining scenes (least cloudy first)."""
    flat = [p["road"] for p in pts] + [p["bg1"] for p in pts] + [p["bg2"] for p in pts]
    n = len(pts)
    vals = np.full(n, np.nan)
    for sc in sorted(sc_list, key=lambda s: s["cloud"])[:6]:
        todo = np.isnan(vals)
        if not todo.any():
            break
        try:
            r = read_points(sc, flat)
        except Exception:
            continue
        if r is None:
            continue
        ok = r["inside"] & np.isin(r["scl"], [4, 5, 6, 7]) & (r["blue"] > 0)  # veg, bare, water, unclassified
        ratio = np.where(ok, r["red"] / np.maximum(r["blue"], 1), np.nan)
        road, b1, b2 = ratio[:n], ratio[n:2 * n], ratio[2 * n:]
        c = road - np.nanmean(np.vstack([b1, b2]), axis=0)
        fill = todo & ~np.isnan(c)
        vals[fill] = c[fill]
    good = ~np.isnan(vals)
    return (float(np.nanmedian(vals)) if good.sum() >= max(5, n // 4) else None), int(good.sum())


def osm_surface(coords):
    def go():
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        bb = f"{min(ys)},{min(xs)},{max(ys)},{max(xs)}"
        # sample the route; ask Overpass for main roads within 30 m of ~15 route points
        idx = np.linspace(0, len(coords) - 1, 15).astype(int)
        around = "".join(f'way(around:30,{coords[i][1]},{coords[i][0]})["highway"~"primary|secondary|tertiary|trunk|unclassified"];' for i in idx)
        q = f"[out:json][timeout:60];({around});out tags;"
        r = http_json("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode())
        return [w.get("tags", {}).get("surface") for w in r.get("elements", [])]
    try:
        return cached(f"osm {coords[0]} {coords[-1]}", go)
    except Exception:
        return []


def main():
    rows, series = [], []
    for label, a, b, kind, year, km in ROADS:
        ga, gb = geocode(a), geocode(b)
        if not ga or not gb:
            print("geocode failed:", label)
            continue
        rt = route(ga, gb)
        pts = sample_points(rt["coords"])
        xs = [c[0] for c in rt["coords"]]
        ys = [c[1] for c in rt["coords"]]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        surf = [s for s in osm_surface(rt["coords"]) if s]
        paved_share = (sum(s in ("asphalt", "paved", "concrete") for s in surf) / len(surf)) if surf else None
        for s in SEASONS:
            sc = scenes(bbox, s)
            c, nvalid = season_contrast(pts, sc) if sc else (None, 0)
            series.append(dict(road=label, kind=kind, contract_year=year, season=s, redness_contrast=c, points=nvalid))
            print(f"{label:40s} {s} scenes={len(sc):2d} contrast={c} n={nvalid}", flush=True)
        rows.append(dict(road=label, kind=kind, contract_year=year, title_km=km, routed_km=round(rt["km"], 1),
                         points=len(pts), osm_paved_share=paved_share, osm_tags=len(surf)))
    S = pd.DataFrame(series)
    R = pd.DataFrame(rows)
    S.to_parquet(P / "satellite_series.parquet", index=False)
    # verdict per road: compare earliest valid season vs latest valid season
    verdicts = []
    for r in R.itertuples():
        s = S[(S.road == r.road) & S.redness_contrast.notna()].sort_values("season")
        if len(s) < 2:
            verdicts.append("insufficient imagery")
            continue
        first, last = s.iloc[0].redness_contrast, s.iloc[-1].redness_contrast
        drop = first - last
        verdicts.append("paving signal (redness fell)" if drop > 0.08 and last < 0.15 else
                        "low redness throughout (paved or no gravel signal)" if first < 0.08 and last < 0.08 else
                        "no clear change: field check")
    R["verdict"] = verdicts
    R.to_parquet(P / "satellite_roads.parquet", index=False)
    (ROOT / "paper" / "satellite.json").write_text(json.dumps(dict(roads=R.to_dict("records")), indent=2, default=float))
    print(R.to_string(index=False))
    chart(S)


def chart(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "src"))
    from visuals import BLUE, ORANGE, INK2, title
    roads = [r for r in S.road.unique()]
    n = len(roads)
    cols = 4
    rows_ = math.ceil(n / cols)
    fig = plt.figure(figsize=(10, 2.3 * rows_ + 1.3))
    title(fig, "Was the road paved? Sentinel-2 road redness over time",
          "Red laterite gravel -> grey asphalt. Line = road redness minus surroundings; dashed = contract year",
          source="Contracts: PPDA open contracting data")
    for i, r in enumerate(roads):
        s = S[S.road == r].sort_values("season")
        ax = fig.add_axes([0.06 + (i % cols) * 0.235, 0.80 - (i // cols + 1) * (0.70 / rows_), 0.20, 0.70 / rows_ - 0.08])
        k = s.kind.iloc[0]
        ax.plot(s.season, s.redness_contrast, color=ORANGE if k.startswith("control") else BLUE, lw=2, marker="o", ms=3)
        ax.axvline(s.contract_year.iloc[0], color=INK2, lw=1, ls="--")
        ax.axhline(0, color="#cfcfcf", lw=0.8)
        ax.set_title(r[:34], fontsize=8.5, loc="left", color=INK2)
        ax.tick_params(labelsize=7)
        ax.set_xlim(2017.5, 2026.5)
        ax.grid(axis="x", visible=False)
    fig.text(0.5, 0.025, "Imagery: Copernicus Sentinel-2 (ESA) via Earth Search · routes: OpenStreetMap / OSRM",
             fontsize=8.5, color=INK2, ha="center")
    out = ROOT / "posts" / "media" / "chart_satellite.png"
    fig.savefig(out, dpi=200)
    shutil.copy(out, ROOT / "docs" / "media" / out.name)


if __name__ == "__main__" and "--summarize" not in sys.argv:
    main()


def summarize():
    """Verdicts from saved readings (no re-download).
    Route check: routed length must be 0.45-1.6x the contract length, else the geocoder/router
    likely picked the wrong place -> excluded. Verdicts use the time pattern of road redness:
    red laterite gravel high (>~0.3), sealed asphalt low (<~0.15); earthworks can push it UP first."""
    S = pd.read_parquet(P / "satellite_series.parquet")
    R = pd.read_parquet(P / "satellite_roads.parquet")
    out = []
    for r in R.itertuples():
        ratio = r.routed_km / r.title_km
        s = S[(S.road == r.road) & S.redness_contrast.notna()].sort_values("season")
        v = s.redness_contrast.to_numpy()
        if not (0.45 <= ratio <= 1.6):
            verdict = "excluded: route does not match contract length (geocoding/routing error)"
        elif len(v) < 3:
            verdict = "insufficient cloud-free imagery"
        else:
            first, last2 = v[:2].mean(), v[-2:].mean()
            paved_season = next((int(x.season) for x in s.itertuples() if x.redness_contrast < 0.15 and x.season > r.contract_year), None)
            if first >= 0.3 and last2 < 0.15:
                verdict = f"paved during period (first low reading {paved_season})"
            elif np.median(v) < 0.15:
                verdict = "paved throughout (consistent with control)"
            elif last2 >= 0.3:
                verdict = "still gravel or under construction at latest image: field check"
            else:
                verdict = "mixed / partly paved: field check"
        out.append(dict(road=r.road, kind=r.kind, contract_year=r.contract_year, title_km=r.title_km,
                        routed_km=r.routed_km, route_ratio=round(ratio, 2), verdict=verdict,
                        osm_paved_share=r.osm_paved_share))
    V = pd.DataFrame(out)
    V.to_parquet(P / "satellite_roads.parquet", index=False)
    (ROOT / "paper" / "satellite.json").write_text(json.dumps(dict(roads=V.to_dict("records")), indent=2, default=float))
    print(V[["road", "contract_year", "route_ratio", "verdict"]].to_string(index=False))
    chart(S[S.road.isin(V[~V.verdict.str.startswith("excluded")].road)])


if __name__ == "__main__" and "--summarize" in sys.argv:
    summarize()
