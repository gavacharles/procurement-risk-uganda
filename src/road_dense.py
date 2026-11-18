"""Paper 3 extension: a denser and a longer satellite record of road sealing.

A. Monthly Sentinel-2 (Jan 2017 - Aug 2026). The main pipeline (road_progress.py) reads two dry-ish windows a
   year, which limits timing to about six months. Here every calendar month is read for every road, with the
   same points, classifier and sealing rule. Questions: how many months are readable; does the per-reading
   error rise in the wet season (checked on paved controls and on kilometres sealed before 2018); do monthly
   seal dates agree with half-yearly ones; and does the sharper timing improve agreement with news dates and
   with the physical progress in government monitoring reports.

B. Landsat 7/8/9 Collection 2 Level-2 (2013-2026, Microsoft Planetary Computer), half-yearly in the same
   windows. Landsat extends the record back four years, before the Sentinel-2 archive, at 30 m. A separate
   classifier is trained on the same OpenStreetMap labels (side pixels at 45 m and 90 m, since 30 m falls in
   the neighbouring pixel). ETM+ (Landsat 7) reflectance is converted to the OLI scale with the Roy et al.
   (2016) OLS coefficients. Questions: how well does it agree with Sentinel-2 where both exist (2017-2026); and
   can it date the roads tendered in 2014-2016, two of which were reported finished before 2017.

Neither part changes the main Paper 3 results; they are a validation and extension section.

Run:  python src/road_dense.py s2      (fetch monthly Sentinel-2 features; resumable)
      python src/road_dense.py landsat (train the Landsat classifier and fetch road features; resumable)
      python src/road_dense.py summary (paper/road_dense.json, figure p3_f18_dense_landsat.png)
Out:  data/processed/road_dense_{s2,landsat}_segments.parquet, paper/road_dense.json
"""
from __future__ import annotations

import fcntl
import io
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import road_progress as rp  # noqa: E402
import satellite as sat  # noqa: E402
import semantic_change as sc  # noqa: E402

ROOT = rp.ROOT
P = rp.P
DC = sat.CACHE / "dense_s2"
LC = sat.CACHE / "landsat"
PCSTAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
LCOLL = "landsat-c2-l2"

# ---------------------------------------------------------------- periods
MONTHS = []
for _y in range(2017, 2027):
    for _m in range(1, 13):
        if (_y, _m) > (2026, 8):
            break
        _end = (pd.Timestamp(_y, _m, 1) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        MONTHS.append((f"{_y}-{_m:02d}", f"{_y}-{_m:02d}-01", _end, _y + (_m - 0.5) / 12))
MT = {m[0]: m[3] for m in MONTHS}

LPERIODS = []
for _y in range(2013, 2027):
    LPERIODS.append((f"{_y}a", f"{_y - 1}-12-01", f"{_y}-02-28", _y + 0.08))
    LPERIODS.append((f"{_y}b", f"{_y}-06-01", f"{_y}-08-31", _y + 0.58))
LT = {p[0]: p[3] for p in LPERIODS}


def road_inputs():
    """Valid routes and their segment points, exactly as in the main pipeline."""
    out = {}
    for road in rp.ROADS:
        g = rp.geometry(road)
        if not g:
            continue
        lo = 0.35 if road[0] in rp.PARTIAL else 0.6
        if lo <= g["ratio"] <= 1.6:
            g["segs"] = rp.segments(g["coords"], road[3])
            out[road[0]] = g
    return out


def bbox_of(pts, pad=0.01):
    return [pts.lon.min() - pad, pts.lat.min() - pad, pts.lon.max() + pad, pts.lat.max() + pad]


# ---------------------------------------------------------------- A. monthly Sentinel-2
def s2_features(points: pd.DataFrame, scenes: list[dict]) -> pd.DataFrame:
    """semantic_change.features_from_scenes, but the scene classification layer is read first at the road
    points, and the six reflectance bands only for points that are clear in that scene. Same output."""
    import rasterio
    from pyproj import Transformer
    feats = pd.DataFrame(index=points.index)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="3")
    for s in sorted(scenes, key=lambda s: s["cloud"]):
        b = s["bbox"]
        todo = points[(points.lon > b[0]) & (points.lon < b[2]) & (points.lat > b[1]) & (points.lat < b[3])]
        if len(feats.columns):
            todo = todo[~todo.index.isin(feats.dropna(subset=["road_ndvi"]).index)]
        if todo.empty:
            continue
        try:
            with rasterio.Env(**env), rasterio.open(s["scl"]) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                scl = np.array([v[0] for v in src.sample([tr.transform(x, y) for x, y in zip(todo.lon, todo.lat)])])
        except Exception as e:
            print("  scl fail", str(e)[:100], flush=True)
            continue
        todo = todo[np.isin(scl, [4, 5, 7])]
        if todo.empty:
            continue
        G = sc.features_from_scenes(todo, [s])
        if len(G.columns):
            feats = G.combine_first(feats) if len(feats.columns) else G
    return feats


def s2_month_task(rid, month, pts_json):
    DC.mkdir(parents=True, exist_ok=True)
    f = DC / f"{rid}_{month}.parquet"
    if f.exists():
        return rid, month, "cached"
    pts = pd.read_json(io.StringIO(pts_json))
    _, start, end, _ = next(m for m in MONTHS if m[0] == month)
    try:
        scs = rp.scenes_window(bbox_of(pts), start, end, max_cloud=80, per_tile=4)
        F = s2_features(pts, scs) if scs else pd.DataFrame(index=pts.index)
    except Exception as e:
        return rid, month, f"fail {e}"
    F = F.reindex(pts.index)
    F["seg"] = pts.seg.values
    F.to_parquet(f)
    return rid, month, f"ok n={int(F.get('road_ndvi', pd.Series(dtype=float)).notna().sum())}"


SEGFILE = ROOT / "ci" / "road_segments.json"


def export_segments():
    """Segment points of every valid route, so the monthly fetch can run on a machine without the geocoding
    caches or the (git-ignored) procurement data, e.g. a GitHub Actions runner. Roads and points only."""
    geo = road_inputs()
    SEGFILE.parent.mkdir(exist_ok=True)
    SEGFILE.write_text(json.dumps({rid: g["segs"].round(6).to_dict("records") for rid, g in geo.items()}, indent=0))
    print(f"{len(geo)} roads -> {SEGFILE}")


def fetch_s2(workers=12, roads=None, segfile=None):
    if segfile:
        geo = {rid: dict(segs=pd.DataFrame(v)) for rid, v in json.loads(Path(segfile).read_text()).items()}
    else:
        geo = road_inputs()
    if roads:
        geo = {k: v for k, v in geo.items() if k in roads}
    tasks = [(rid, m[0], g["segs"].to_json()) for rid, g in geo.items() for m in MONTHS]
    run_pool(s2_month_task, tasks, workers)


def run_pool(fn, tasks, workers):
    done = fails = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, *t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            rid, per, msg = fu.result()
            done += 1
            fails += msg.startswith("fail")
            if msg != "cached":
                print(f"[{i + 1}/{len(futs)}] {rid} {per} {msg}", flush=True)
    print(f"finished {done} tasks, {fails} failed (rerun to retry)", flush=True)


# ---------------------------------------------------------------- B. Landsat
LBANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
# Roy et al. (2016), Remote Sensing of Environment 185, Table 2: OLS ETM+ -> OLI surface reflectance
ROY = {"blue": (0.0003, 0.8474), "green": (0.0088, 0.8483), "red": (0.0061, 0.9047),
       "nir08": (0.0412, 0.8462), "swir16": (0.0254, 0.8937), "swir22": (0.0172, 0.9071)}
LSIDE = (45, 90)


def pc_token():
    """One SAS token shared by all workers through a locked file (the token endpoint rate-limits)."""
    LC.mkdir(parents=True, exist_ok=True)
    tf, lk = LC / "token.json", LC / "token.lock"
    with open(lk, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            t = json.loads(tf.read_text())
            exp = datetime.fromisoformat(t["msft:expiry"].replace("Z", "+00:00"))
            if (exp - datetime.now(timezone.utc)).total_seconds() > 600:
                return t["token"]
        except (OSError, ValueError, KeyError):
            pass
        for i in range(6):
            try:
                with urllib.request.urlopen(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{LCOLL}", timeout=60) as r:
                    t = json.loads(r.read())
                tf.write_text(json.dumps(t))
                return t["token"]
            except Exception:
                time.sleep(15 * (i + 1))
        raise RuntimeError("no Planetary Computer token")


def ls_scenes(bbox, start, end, max_cloud=70, per_tile=4):
    def go():
        body = json.dumps({"collections": [LCOLL], "bbox": bbox, "limit": 200,
                           "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                           "query": {"eo:cloud_cover": {"lt": max_cloud},
                                     "platform": {"in": ["landsat-7", "landsat-8", "landsat-9"]}},
                           "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}]}).encode()
        r = sat.http_json(f"{PCSTAC}/search", data=body)
        tiles = {}
        for f in r.get("features", []):
            pr = f["properties"]
            t = (pr.get("landsat:wrs_path"), pr.get("landsat:wrs_row"))
            tiles.setdefault(t, [])
            if len(tiles[t]) < per_tile:
                tiles[t].append(dict(id=f["id"], cloud=pr["eo:cloud_cover"], bbox=f["bbox"], platform=pr["platform"],
                                     date=pr["datetime"][:10],
                                     **{b: f["assets"][b]["href"] for b in LBANDS + ["qa_pixel"]}))
        return sorted((s for v in tiles.values() for s in v), key=lambda s: s["cloud"])
    return sat.cached(f"ls {bbox} {start} {end} {max_cloud} {per_tile}", go)


def ls_sample(scene, lonlats):
    import rasterio
    from pyproj import Transformer
    tok = pc_token()
    vals = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="5",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff"):
        for b in LBANDS + ["qa_pixel"]:
            with rasterio.open(scene[b] + "?" + tok) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xy = [tr.transform(*p) for p in lonlats]
                vals[b] = np.array([v[0] for v in src.sample(xy)], dtype=float)
    return vals


def ls_clear(qa):
    """QA_PIXEL: fill (bit 0) off, clear (bit 6) on, dilated cloud (1), cloud (3), shadow (4), water (7) off."""
    q = qa.astype(np.int64)
    bit = lambda k: (q >> k) & 1  # noqa: E731
    return (bit(0) == 0) & (bit(6) == 1) & (bit(1) == 0) & (bit(3) == 0) & (bit(4) == 0) & (bit(7) == 0)


def ls_features(points: pd.DataFrame, scenes: list[dict]) -> pd.DataFrame:
    """Like semantic_change.features_from_scenes, for Landsat: least cloudy scene first fills each point."""
    feats = pd.DataFrame(index=points.index)
    for s in scenes:
        b = s["bbox"]
        todo = points[(points.lon > b[0]) & (points.lon < b[2]) & (points.lat > b[1]) & (points.lat < b[3])]
        if len(feats.columns):
            todo = todo[~todo.index.isin(feats.dropna(subset=["road_ndvi"]).index)]
        if todo.empty:
            continue
        locs = []
        for r in todo.itertuples():
            n1 = sc.offsets(r.lon, r.lat, r.lon2, r.lat2, LSIDE[0])
            n2 = sc.offsets(r.lon, r.lat, r.lon2, r.lat2, LSIDE[1])
            locs += [(r.lon, r.lat), n1[0], n1[1], n2[0], n2[1]]
        try:
            v = ls_sample(s, locs)
        except Exception as e:
            print("  read fail", s["id"], str(e)[:120], flush=True)
            continue
        k = len(todo)
        arr = {bn: v[bn].reshape(k, 5) for bn in LBANDS + ["qa_pixel"]}
        ok = ls_clear(arr["qa_pixel"]).all(axis=1) & (arr["blue"][:, 0] > 0)
        refl = {}
        for nm, bn in zip(sc.BKEYS, LBANDS):
            x = arr[bn] * 2.75e-5 - 0.2
            if s["platform"] == "landsat-7":
                a, slope = ROY[bn]
                x = a + slope * x
            refl[nm] = x
        road = {nm: refl[nm][:, 0] for nm in sc.BKEYS}
        bg1 = {nm: (refl[nm][:, 1] + refl[nm][:, 2]) / 2 for nm in sc.BKEYS}
        bg2 = {nm: (refl[nm][:, 3] + refl[nm][:, 4]) / 2 for nm in sc.BKEYS}
        F = pd.DataFrame(sc.features_from_bands(road, bg1, bg2), index=todo.index)
        F["l7"] = float(s["platform"] == "landsat-7")
        F[~ok] = np.nan
        feats = F.combine_first(feats) if len(feats.columns) else F
    return feats


LTRAIN = [("2026a", "2025-12-01", "2026-02-28"), ("2025b", "2025-06-01", "2025-08-31")]


def ls_train_features(window):
    tag, start, end = window
    f = LC / f"train_{tag}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    pts = sc.osm_points()
    scenes, seen = [], set()
    for (cx, cy), _ in pts.groupby([np.floor(pts.lon), np.floor(pts.lat)]):
        for s in ls_scenes([float(cx), float(cy), float(cx) + 1, float(cy) + 1], start, end, max_cloud=40, per_tile=3):
            if s["id"] not in seen:
                seen.add(s["id"])
                scenes.append(s)
    X = ls_features(pts, sorted(scenes, key=lambda s: s["cloud"]))
    LC.mkdir(parents=True, exist_ok=True)
    X.to_parquet(f)
    return X


def ls_cols(d):
    return [c for c in d.columns if c.startswith(("road_", "d30_", "d60_"))]


def ls_train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    pts = sc.osm_points()
    d = pd.concat([pts.join(ls_train_features(w)).dropna(subset=["road_ndvi"]).assign(window=w[0]) for w in LTRAIN],
                  ignore_index=True)
    d = d[d.road_ndvi < 0.45]
    cols = ls_cols(d)
    y = (d.cls == "asphalt").astype(int).to_numpy()
    preds = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(d, y, groups=d.way):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d.iloc[tr][cols], y[tr])
        preds[te] = m.predict(d.iloc[te][cols])
    info = dict(n=int(len(d)), n_ways=int(d.way.nunique()), cv_accuracy=round(float(accuracy_score(y, preds)), 3),
                f1_asphalt=round(float(f1_score(y, preds)), 3), baseline=round(float(max(y.mean(), 1 - y.mean())), 3))
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d[cols], y)
    print("landsat classifier:", info, flush=True)
    return model, cols, info


def ls_task(rid, period, pts_json):
    LC.mkdir(parents=True, exist_ok=True)
    f = LC / f"{rid}_{period}.parquet"
    if f.exists():
        return rid, period, "cached"
    pts = pd.read_json(io.StringIO(pts_json))
    _, start, end, _ = next(p for p in LPERIODS if p[0] == period)
    try:
        scs = ls_scenes(bbox_of(pts), start, end, max_cloud=70, per_tile=5)
        F = ls_features(pts, scs) if scs else pd.DataFrame(index=pts.index)
    except Exception as e:
        return rid, period, f"fail {e}"
    F = F.reindex(pts.index)
    F["seg"] = pts.seg.values
    F.to_parquet(f)
    return rid, period, f"ok n={int(F.get('road_ndvi', pd.Series(dtype=float)).notna().sum())}"


def fetch_landsat(workers=6):
    ls_train()
    geo = road_inputs()
    tasks = [(rid, p[0], g["segs"].to_json()) for rid, g in geo.items() for p in LPERIODS]
    run_pool(ls_task, tasks, workers)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="summary", choices=["s2", "landsat", "summary", "export-segments"])
    ap.add_argument("--roads", default="", help="comma-separated road ids (default: all)")
    ap.add_argument("--segments", default=None, help="segment file from export-segments (no geocoding needed)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    what = a.what
    if what == "export-segments":
        export_segments()
    elif what == "s2":
        fetch_s2(a.workers, [r for r in a.roads.split(",") if r], a.segments)
    elif what == "landsat":
        fetch_landsat()
    elif what == "summary":
        import road_dense_summary
        road_dense_summary.main()
