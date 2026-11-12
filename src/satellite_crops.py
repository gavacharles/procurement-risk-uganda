"""Before / during / after true-colour Sentinel-2 crops along each validated contract road.

For each road that passed the route-length check in src/satellite.py:
  - candidate points at 20..80% along the routed line;
  - three dates (same as src/semantic_change.py): t0 = dry season before the contract
    (earliest 2018), t1 = contract year + 2, t2 = latest dry season;
  - for each date, the least-cloudy scene whose SCL band shows < 3% cloud/shadow in a
    2 km x 2 km window around the point; the true-colour TCI COG is read for that window only;
  - the first point where all three dates have a clean window is used.
One contrast stretch per road (pooled over its three crops) so dates are comparable.

Out: posts/media/chart_satellite_crops.png (+ docs/media copy, paper/figures copy),
     paper/satellite_crops.json (scene ids and point used, for reproducibility)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import satellite as sat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
HALF_M = 1000          # half window, metres -> 2 km square
LATEST = 2026
CLOUDY = [0, 1, 3, 8, 9, 10]  # no-data, saturated, shadow, cloud med/high, cirrus


def tci_href(scene):
    b = scene["blue"]
    return b.replace("/B02.tif", "/TCI.tif") if b.startswith("https://") and b.endswith("/B02.tif") else None


def read_window(href, lon, lat, out_px=None):
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="3"):
        with rasterio.open(href) as src:
            x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
            b = src.bounds
            if not (b.left + HALF_M < x < b.right - HALF_M and b.bottom + HALF_M < y < b.top - HALF_M):
                return None
            w = from_bounds(x - HALF_M, y - HALF_M, x + HALF_M, y + HALF_M, src.transform)
            kw = dict(out_shape=(src.count, out_px, out_px)) if out_px else {}
            return src.read(window=w, **kw)


def clean_crop(lon, lat, season):
    bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
    for sc in sorted(sat.scenes(bbox, season), key=lambda s: s["cloud"])[:8]:
        href = tci_href(sc)
        if not href:
            continue
        try:
            scl = read_window(sc["scl"], lon, lat)
            if scl is None or np.isin(scl[0], CLOUDY).mean() > 0.03:
                continue
            img = read_window(href, lon, lat)
        except Exception:
            continue
        if img is not None and img.shape[0] == 3 and (img.sum(axis=0) == 0).mean() < 0.01:
            return np.moveaxis(img, 0, -1).astype(float), sc["id"]
    return None, None


def dates(contract_year):
    # season Y = Dec(Y-1)..Feb(Y), so season == contract year is at or before the start of works
    t0 = 2018 if contract_year < 2018 else min(contract_year, LATEST - 2)
    t1 = min(max(contract_year + 2, t0 + 1), LATEST - 1)
    return [t0, t1, LATEST]


def main():
    V = pd.read_parquet(P / "satellite_roads.parquet")
    V = V[~V.verdict.str.startswith("excluded")]
    roads = {r[0]: r for r in sat.ROADS}
    out, meta = [], []
    for v in V.itertuples():
        _, a, b, kind, year, _ = roads[v.road]
        co = sat.route(sat.geocode(a), sat.geocode(b))["coords"]
        ds = dates(year)
        got = None
        for f in [0.5, 0.35, 0.65, 0.2, 0.8, 0.45, 0.55, 0.3, 0.7]:
            lon, lat = co[int(f * (len(co) - 1))]
            crops = [clean_crop(lon, lat, d) for d in ds]
            print(f"{v.road:40s} f={f:.2f} {[c[1] is not None for c in crops]}", flush=True)
            if all(c[0] is not None for c in crops):
                got = (lon, lat, f, crops)
                break
        if not got:
            print("  no clean triple:", v.road)
            continue
        lon, lat, f, crops = got
        # route segment inside the window, for a faint guide line
        seg = [(x, y) for x, y in co if abs(x - lon) < 0.012 and abs(y - lat) < 0.012]
        out.append(dict(road=v.road, kind=kind, year=year, verdict=v.verdict, lon=lon, lat=lat,
                        seasons=ds, imgs=[c[0] for c in crops], seg=seg))
        meta.append(dict(road=v.road, point=[round(lon, 5), round(lat, 5)], route_fraction=f,
                         seasons=ds, scenes=[c[1] for c in crops]))
    (ROOT / "paper" / "satellite_crops.json").write_text(json.dumps(meta, indent=2))
    chart(out)


def stretch(imgs):
    pool = np.concatenate([i.reshape(-1, 3) for i in imgs])
    pool = pool[pool.sum(axis=1) > 0]
    lo, hi = np.percentile(pool, 1, axis=0), np.percentile(pool, 99, axis=0)
    return [np.clip((i - lo) / (hi - lo), 0, 1) ** 0.85 for i in imgs]


SHORT = {
    "paved during period": "paving visible",
    "still gravel": "still unpaved: field check",
    "mixed": "mixed: field check",
    "paved throughout": "paved throughout",
}


def chart(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visuals import INK, INK2, title
    order = {"upgrade": 0}
    out = sorted(out, key=lambda r: (order.get(r["kind"], 1), r["road"]))
    n = len(out)
    row_h, top, bot = 1.95, 1.45, 0.45
    H = top + n * row_h + bot
    fig = plt.figure(figsize=(9, H))
    title(fig, "Was the road paved? Before, during, after",
          "Sentinel-2 true colour, 2 km × 2 km around one point on each contracted road.\nGravel shows as a pale or orange track; asphalt as a grey line.",
          source="Imagery: Copernicus Sentinel-2 (ESA) via Earth Search")
    # title() positions assume a landscape figure; re-anchor its text by inches
    for t in fig.texts:
        x, y = t.get_position()
        if y > 0.5:
            t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H)
        else:
            t.set_y(0.18 / H)
    img_w = 0.2
    for i, r in enumerate(out):
        y0 = 1 - (top + (i + 1) * row_h) / H
        h = (row_h - 0.25) / H
        lab = next((s for k, s in SHORT.items() if r["verdict"].startswith(k)), r["verdict"])
        ctl = r["kind"].startswith("control")
        name = r["road"].replace(" (rehab, already paved)", "")
        if len(name) > 22:  # wrap long names after the first endpoint
            name = name.replace("–", "–\n", 1)
        fig.text(0.04, y0 + h - 0.02 / H, name, fontsize=10.5,
                 fontweight="bold", color=INK, va="top")
        fig.text(0.04, y0 + h - 0.62 / H, ("Control: already paved before\nthe contract (rehabilitation)" if ctl else
                 f"Contract year {r['year']}\nReading: {lab}"), fontsize=8.5, color=INK2, va="top", linespacing=1.4)
        fig.text(0.04, y0 + h - 1.25 / H, f"{r['lat']:.3f}°, {r['lon']:.3f}°", fontsize=7.5, color=INK2, va="top")
        for j, (img, s) in enumerate(zip(stretch(r["imgs"]), r["seasons"])):
            ax = fig.add_axes([0.335 + j * (img_w + 0.018), y0, img_w, h])
            ax.imshow(img, interpolation="lanczos")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            lbl = ["before", "during", "latest"][j]
            ax.set_title(f"{lbl} · dry season {s - 1}/{str(s)[2:]}", fontsize=8, color=INK2, loc="left", pad=3)
    out_p = ROOT / "posts" / "media" / "chart_satellite_crops.png"
    fig.savefig(out_p, dpi=200)
    shutil.copy(out_p, ROOT / "docs" / "media" / out_p.name)
    shutil.copy(out_p, ROOT / "paper" / "figures" / "p2_f5b_satellite_crops.png")
    print("wrote", out_p)


if __name__ == "__main__":
    main()
