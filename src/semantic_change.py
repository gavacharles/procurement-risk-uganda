"""Experiment A: three-date semantic change detection for road contracts (Sentinel-2 + OSM labels).

1. Training labels (weak supervision): OpenStreetMap main roads in Uganda tagged
   surface=asphalt/paved  -> class "asphalt"
   surface=unpaved/gravel/dirt/ground/compacted -> class "gravel"
2. Features at each road point and its 30 m / 60 m surroundings from one Sentinel-2 L2A dry-season
   scene: B2,B3,B4,B8,B11,B12 + NDVI, BSI (bare-soil index), redness B4/B2, brightness, and
   road-minus-surroundings contrasts.
3. Gradient-boosted classifier asphalt vs gravel, evaluated with GROUPED cross-validation
   (points from the same OSM way never in train and test together).
4. Semantic state per point = asphalt | unpaved | vegetated, where
   vegetated  : road-pixel NDVI > 0.45 (no visible road)
   (an "earthworks" class was tried and dropped: dry-season fields are bare too, so a bare
    corridor does not identify construction)
5. For each validated contract road (src/satellite.py routes), three dates:
   t0 = dry season before the contract year (or earliest available, 2018)
   t1 = contract year + 2 (mid / expected completion)
   t2 = latest (2026)
   -> class shares per date and the dominant transition.

Out: data/processed/semantic_change.parquet, paper/semantic_change.json, posts/media/chart_semantic_change.png
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import satellite as sat  # noqa: E402  (reuses http, cache, routing, STAC search)

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
BANDS = ["blue", "green", "red", "nir", "swir16", "swir22", "scl"]
UGANDA = [29.5, -1.5, 35.1, 4.3]
OSM_PAVED = {"asphalt", "paved", "concrete", "chipseal"}
OSM_UNPAVED = {"unpaved", "gravel", "dirt", "ground", "compacted", "earth", "fine_gravel", "laterite"}


# ---------------------------------------------------------------- OSM training points
def osm_points(per_class=180):
    def go():
        q = ('[out:json][timeout:240];'
             f'way["highway"~"^(trunk|primary|secondary|tertiary)$"]["surface"]({UGANDA[1]},{UGANDA[0]},{UGANDA[3]},{UGANDA[2]});'
             'out tags geom 4000;')
        r = sat.http_json("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode())
        out = []
        for w in r.get("elements", []):
            s = (w.get("tags") or {}).get("surface", "")
            cls = "asphalt" if s in OSM_PAVED else "gravel" if s in OSM_UNPAVED else None
            g = w.get("geometry") or []
            if cls and len(g) >= 4:
                mid = g[len(g) // 2]
                nxt = g[min(len(g) // 2 + 1, len(g) - 1)]
                out.append(dict(way=w["id"], cls=cls, lon=mid["lon"], lat=mid["lat"], lon2=nxt["lon"], lat2=nxt["lat"]))
        return out
    pts = sat.cached("osm surface training ways uganda", go)
    df = pd.DataFrame(pts)
    rng = np.random.default_rng(0)
    if df.empty:
        return df
    n = int(min(per_class, df.cls.value_counts().min()))
    return df.groupby("cls").sample(n=n, random_state=1).reset_index(drop=True)


def offsets(lon, lat, lon2, lat2, metres):
    dx, dy = lon2 - lon, lat2 - lat
    n = math.hypot(dx, dy) or 1
    nx, ny = -dy / n, dx / n
    off = metres / 111_000
    return (lon + nx * off, lat + ny * off), (lon - nx * off, lat - ny * off)


# ---------------------------------------------------------------- Sentinel-2 sampling
def best_scenes(bbox, season, max_cloud=25):
    """Least-cloudy scene per MGRS tile for the dry season ending Feb of `season`."""
    def go():
        body = json.dumps({"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 100,
                           "datetime": f"{season - 1}-12-01T00:00:00Z/{season}-02-28T23:59:59Z",
                           "query": {"eo:cloud_cover": {"lt": max_cloud}}}).encode()
        r = sat.http_json(f"{sat.STAC}/search", data=body)
        best = {}
        for f in r.get("features", []):
            t = f["properties"].get("grid:code")
            if t not in best or f["properties"]["eo:cloud_cover"] < best[t]["cloud"]:
                best[t] = dict(cloud=f["properties"]["eo:cloud_cover"], bbox=f["bbox"],
                               **{b: f["assets"][b]["href"] for b in BANDS})
        return list(best.values())
    return sat.cached(f"best scenes {bbox} {season} {max_cloud}", go)


def sample_bands(scene, lonlats):
    import rasterio
    from pyproj import Transformer
    vals = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="3"):
        for b in BANDS:
            with rasterio.open(scene[b]) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xy = [tr.transform(*p) for p in lonlats]
                vals[b] = np.array([v[0] for v in src.sample(xy)], dtype=float)
    return vals


def features_for(points: pd.DataFrame, season: int) -> pd.DataFrame:
    """points: lon,lat,lon2,lat2 (+ any id cols). Returns per-point features (NaN where cloudy/outside)."""
    # search per 1-degree cell (a single country-wide search is rejected by the catalogue)
    scenes, seen = [], set()
    for (cx, cy), _ in points.groupby([np.floor(points.lon), np.floor(points.lat)]):
        cell = [float(cx), float(cy), float(cx) + 1, float(cy) + 1]
        try:
            for sc in best_scenes(cell, season):
                if sc["blue"] not in seen:
                    seen.add(sc["blue"])
                    scenes.append(sc)
        except Exception as e:
            print("  stac fail", cell, e, flush=True)
    feats = pd.DataFrame(index=points.index)
    for sc in sorted(scenes, key=lambda s: s["cloud"]):
        b = sc["bbox"]
        todo = points[(points.lon > b[0]) & (points.lon < b[2]) & (points.lat > b[1]) & (points.lat < b[3])]
        todo = todo[~todo.index.isin(feats.dropna().index)] if len(feats.columns) else todo
        if todo.empty:
            continue
        locs = []
        for r in todo.itertuples():
            n30 = offsets(r.lon, r.lat, r.lon2, r.lat2, 30)
            n60 = offsets(r.lon, r.lat, r.lon2, r.lat2, 60)
            locs += [(r.lon, r.lat), n30[0], n30[1], n60[0], n60[1]]
        try:
            v = sample_bands(sc, locs)
        except Exception as e:
            print("  read fail", e, flush=True)
            continue
        k = len(todo)
        arr = {bn: v[bn].reshape(k, 5) for bn in BANDS}
        ok = np.isin(arr["scl"][:, 0], [4, 5, 7]) & (arr["blue"][:, 0] > 0)

        def idx(a, i):
            bl, gr, rd, ni, s1, s2 = (arr[x][:, i] / 10000 for x in ["blue", "green", "red", "nir", "swir16", "swir22"])
            ndvi = (ni - rd) / (ni + rd + 1e-6)
            bsi = ((s1 + rd) - (ni + bl)) / ((s1 + rd) + (ni + bl) + 1e-6)
            return dict(bl=bl, gr=gr, rd=rd, ni=ni, s1=s1, s2=s2, ndvi=ndvi, bsi=bsi,
                        red_ratio=rd / (bl + 1e-6), bright=(bl + gr + rd) / 3)
        road = idx(arr, 0)
        bg30 = {key: (idx(arr, 1)[key] + idx(arr, 2)[key]) / 2 for key in road}
        bg60 = {key: (idx(arr, 3)[key] + idx(arr, 4)[key]) / 2 for key in road}
        f = {}
        for key in road:
            f[f"road_{key}"] = road[key]
            f[f"d30_{key}"] = road[key] - bg30[key]
            f[f"d60_{key}"] = road[key] - bg60[key]
        f["bg30_bsi"] = bg30["bsi"]
        F = pd.DataFrame(f, index=todo.index)
        F[~ok] = np.nan
        feats = F.combine_first(feats) if len(feats.columns) else F
    return feats


# ---------------------------------------------------------------- train
def train(season=2026):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, f1_score
    pts = osm_points()
    print(f"OSM training ways: {pts.cls.value_counts().to_dict()}", flush=True)
    fcache = sat.CACHE / f"train_features_{season}.parquet"
    if fcache.exists():
        X = pd.read_parquet(fcache)
    else:
        X = features_for(pts, season)
        X.to_parquet(fcache)
    d = pts.join(X).dropna(subset=["road_ndvi"])
    d = d[d.road_ndvi < 0.45]  # visible road only
    cols = [c for c in X.columns if c != "bg30_bsi"]
    y = (d.cls == "asphalt").astype(int).to_numpy()
    preds = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(d, y, groups=d.way):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d.iloc[tr][cols], y[tr])
        preds[te] = m.predict(d.iloc[te][cols])
    acc, f1 = accuracy_score(y, preds), f1_score(y, preds)
    base = max(y.mean(), 1 - y.mean())
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d[cols], y)
    print(f"grouped-CV accuracy {acc:.3f} (majority baseline {base:.3f}), F1(asphalt) {f1:.3f}, n={len(d)}", flush=True)
    return model, cols, dict(n=int(len(d)), cv_accuracy=float(acc), baseline=float(base), f1_asphalt=float(f1),
                             classes=d.cls.value_counts().to_dict())


# ---------------------------------------------------------------- apply to contract roads
def classify(model, cols, F: pd.DataFrame) -> pd.Series:
    out = pd.Series("no data", index=F.index, dtype=object)
    if "road_ndvi" not in F.columns:
        return out
    ok = F.road_ndvi.notna()
    veg = ok & (F.road_ndvi > 0.45)
    out[veg] = "vegetated"
    rest = ok & ~veg
    if rest.any():
        asph = model.predict(F.loc[rest, cols]) == 1
        st = np.where(asph, "asphalt", "unpaved")  # earthworks not separable: dry-season fields are bare too
        out[rest] = st
    return out


def road_points(label, a, b):
    ga, gb = sat.geocode(a), sat.geocode(b)
    rt = sat.route(ga, gb)
    pts = []
    co = rt["coords"]
    idx = np.linspace(1, len(co) - 2, 30).astype(int)
    for i in idx:
        pts.append(dict(road=label, lon=co[i][0], lat=co[i][1], lon2=co[i + 1][0], lat2=co[i + 1][1]))
    return pd.DataFrame(pts), rt["km"]


def main():
    model, cols, train_info = train()
    V = pd.read_parquet(P / "satellite_roads.parquet")
    valid = V[~V.verdict.str.startswith("excluded") & ~V.verdict.str.startswith("insufficient")]
    meta = {r[0]: r for r in sat.ROADS}
    rows = []
    for r in valid.itertuples():
        _, a, b, kind, year, km = meta[r.road]
        pts, rkm = road_points(r.road, a, b)
        dates = {"t0_before": max(2018, year), "t1_mid": min(2026, max(2019, year + 2)), "t2_latest": 2026}
        for tag, season in dates.items():
            F = features_for(pts, season)
            st = classify(model, cols, F)
            shares = st[st != "no data"].value_counts(normalize=True)
            rows.append(dict(road=r.road, kind=kind, contract_year=year, date=tag, season=season,
                             n=int((st != "no data").sum()),
                             **{f"share_{c}": float(shares.get(c, 0.0)) for c in ["asphalt", "unpaved", "vegetated"]}))
            print(f"{r.road:40s} {tag:9s} {season} n={rows[-1]['n']:2d} "
                  + " ".join(f"{c}={shares.get(c, 0):.2f}" for c in ["asphalt", "unpaved", "vegetated"]), flush=True)
    R = pd.DataFrame(rows)
    R.to_parquet(P / "semantic_change.parquet", index=False)

    def dom(g, tag):
        s = g[g.date == tag]
        if s.empty or s.n.iloc[0] < 5:
            return "?"
        c = s[["share_asphalt", "share_unpaved", "share_vegetated"]].iloc[0]
        return c.idxmax().replace("share_", "")
    trans = {road: " → ".join(dom(g, t) for t in ["t0_before", "t1_mid", "t2_latest"]) for road, g in R.groupby("road")}
    (ROOT / "paper" / "semantic_change.json").write_text(json.dumps(dict(training=train_info, transitions=trans,
                                                                          detail=R.to_dict("records")), indent=2, default=float))
    print(json.dumps(trans, indent=1))
    chart(R)


def chart(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shutil
    from visuals import title, INK2
    cols = {"asphalt": "#2a78d6", "unpaved": "#eb6834", "vegetated": "#1baf7a"}
    roads = list(R.road.unique())
    fig = plt.figure(figsize=(9, 0.62 * len(roads) * 1.6 + 1.8))
    title(fig, "Three-date semantic change: what is on the road?",
          "Sentinel-2 + classifier trained on OpenStreetMap surface tags; share of route points per class",
          source="Imagery: Copernicus Sentinel-2 (ESA); labels: OpenStreetMap")
    ax = fig.add_axes([0.30, 0.10, 0.66, 0.72])
    ylabels, y = [], 0
    for road in roads:
        g = R[R.road == road].set_index("date")
        for tag in ["t0_before", "t1_mid", "t2_latest"]:
            if tag not in g.index:
                continue
            left = 0
            for c, col in cols.items():
                v = g.loc[tag, f"share_{c}"]
                ax.barh(y, v, left=left, color=col, height=0.8, edgecolor="#fcfcfb", linewidth=1)
                left += v
            ylabels.append(f"{road[:28]} · {int(g.loc[tag, 'season'])}")
            y += 1
        y += 0.6
    ax.set_yticks([i for i in range(len(ylabels))] if False else [])
    ax.invert_yaxis()
    yy, k = [], 0
    for road in roads:
        n = (R.road == road).sum()
        yy += [k + i for i in range(n)]
        k += n + 0.6
    ax.set_yticks(yy)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()], labels=list(cols),
               loc="upper right", bbox_to_anchor=(0.97, 0.88), ncol=3, frameon=False, fontsize=9)
    out = ROOT / "posts" / "media" / "chart_semantic_change.png"
    fig.savefig(out, dpi=200)
    shutil.copy(out, ROOT / "docs" / "media" / out.name)


if __name__ == "__main__":
    main()
