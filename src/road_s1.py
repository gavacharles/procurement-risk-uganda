"""Follow-on step 3a: monthly Sentinel-1 radar backscatter along each road, 2017-2026.

Why: Sentinel-2 is read twice a year in dry windows, so pauses are measured in half-years and only sealing is
seen. Sentinel-1 C-band SAR sees through cloud and images Uganda every 6-12 days; backscatter responds to
vegetation clearing (VH falls), surface roughness and moisture (earthworks, gravel base) and to smooth asphalt
(VV falls: specular reflection). Whether these changes identify construction activity at 10-20 m on roads
of 7-12 m width is an empirical question, tested in road_activity.py against reported progress.

Data: Microsoft Planetary Computer `sentinel-1-rtc` (radiometrically terrain-corrected gamma0, 10 m, VV+VH,
anonymous SAS token). For each road, the relative orbit covering most of it is used throughout (fixed viewing
geometry), and the first acquisition in each calendar month (fallback: next one if the first misses the road).
Points: the 1 km segment centres of road_progress.py, plus 30 m and 60 m to each side, read from the 20 m
overview (the native resolution of IW GRD-H is ~20 x 22 m; the 10 m grid is oversampled).

Out: ~/.cache/procurement-risk/s1/<road>_<yyyymm>.parquet, data/processed/road_s1.parquet (linear gamma0)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
import road_progress as rp  # noqa: E402
import satellite as sat  # noqa: E402
import semantic_change as sc  # noqa: E402

PCSTAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLL = "sentinel-1-rtc"
CACHE = sat.CACHE.parent / "s1"
START, END = "2017-01-01", "2026-10-31"
_TOKEN = {"t": None, "exp": 0}


def token():
    if time.time() > _TOKEN["exp"]:
        with urllib.request.urlopen(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{COLL}", timeout=60) as r:
            _TOKEN["t"] = json.load(r)["token"]
        _TOKEN["exp"] = time.time() + 3600
    return _TOKEN["t"]


def search(bbox):
    def go():
        out, body = [], {"collections": [COLL], "bbox": bbox, "datetime": f"{START}/{END}", "limit": 500}
        url = f"{PCSTAC}/search"
        while True:
            r = sat.http_json(url, data=json.dumps(body).encode())
            for f in r["features"]:
                if not {"vv", "vh"} <= set(f["assets"]):
                    continue  # single-polarisation acquisitions
                p = f["properties"]
                out.append(dict(id=f["id"], dt=p["datetime"][:10], orbit=p.get("sat:relative_orbit"),
                                direction=p.get("sat:orbit_state"), bbox=f["bbox"],
                                vv=f["assets"]["vv"]["href"], vh=f["assets"]["vh"]["href"]))
            nxt = next((lk for lk in r.get("links", []) if lk.get("rel") == "next"), None)
            if not nxt:
                return out
            body = nxt.get("body", body)
            url = nxt.get("href", url)
    return sat.cached(f"s1 search {bbox} {START} {END}", go)


def points(seg):
    locs = []
    for r in seg.itertuples():
        n30 = sc.offsets(r.lon, r.lat, r.lon2, r.lat2, 30)
        n60 = sc.offsets(r.lon, r.lat, r.lon2, r.lat2, 60)
        locs += [(r.lon, r.lat), n30[0], n30[1], n60[0], n60[1]]
    return locs


def read(scene, locs, tok):
    import rasterio
    from pyproj import Transformer
    out = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                       GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="20", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff", GDAL_INGESTED_BYTES_AT_OPEN="32768"):
        for b in ("vv", "vh"):
            # first overview = 20 m: Sentinel-1 IW GRD-H resolution is ~20 x 22 m (10 m posting is oversampled),
            # and reading it halves the tiles downloaded (bandwidth-bound)
            with rasterio.open(f"{scene[b]}?{tok}", OVERVIEW_LEVEL=0) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                v = np.array([x[0] for x in src.sample([tr.transform(*p) for p in locs])], dtype=float)
                v[(v <= 0) | ~np.isfinite(v) | (v == src.nodata)] = np.nan
                out[b] = v
    return out


def month_task(rid, ym, cands, seg_json, tok):
    f = CACHE / f"{rid}_{ym}.parquet"
    if f.exists():
        return rid, ym, "cached"
    seg = pd.read_json(StringIO(seg_json))
    locs = points(seg)
    k = len(seg)
    failed = False
    last = "no scene"
    for scn in cands:  # first acquisition of the month that actually covers the road
        try:
            v = read(scn, locs, tok)
        except Exception as e:  # noqa: BLE001
            last, failed = f"fail {str(e)[:120]}", True
            continue
        a = {b: v[b].reshape(k, 5) for b in v}
        if np.isfinite(a["vv"][:, 0]).mean() < 0.3:
            last = "no coverage"
            continue
        D = pd.DataFrame(dict(seg=seg.seg.values, ym=ym, date=scn["dt"], scene=scn["id"],
                              vv=a["vv"][:, 0], vh=a["vh"][:, 0],
                              vv30=np.nanmean(a["vv"][:, 1:3], 1), vh30=np.nanmean(a["vh"][:, 1:3], 1),
                              vv60=np.nanmean(a["vv"][:, 3:5], 1), vh60=np.nanmean(a["vh"][:, 3:5], 1)))
        D.to_parquet(f)
        return rid, ym, f"ok {scn['dt']}"
    if not failed:  # every candidate read but none covered the road: remember, so reruns skip it
        pd.DataFrame(columns=["seg", "ym"]).to_parquet(f)
    return rid, ym, last


def plan():
    tasks, meta = [], {}
    for road in rp.ROADS:
        g = rp.geometry(road)
        if not g or not (0.35 <= g["ratio"] <= 1.6):
            continue
        seg = rp.segments(g["coords"], road[3])
        pad = 0.01
        bbox = [seg.lon.min() - pad, seg.lat.min() - pad, seg.lon.max() + pad, seg.lat.max() + pad]
        S = pd.DataFrame(search(bbox))
        if S.empty:
            continue

        # orbit covering most segment centres (by scene bbox), ties -> most acquisitions
        def cover(o):
            bb = S[S.orbit == o].bbox.iloc[0]
            return ((seg.lon > bb[0]) & (seg.lon < bb[2]) & (seg.lat > bb[1]) & (seg.lat < bb[3])).mean()
        orbits = sorted(S.orbit.dropna().unique(), key=lambda o: (cover(o), (S.orbit == o).sum()), reverse=True)
        o = orbits[0]
        T = S[S.orbit == o].sort_values("dt")
        T["ym"] = T.dt.str[:7].str.replace("-", "")
        meta[road[0]] = dict(orbit=int(o), direction=T.direction.iloc[0], n_scenes=len(T), cover=round(cover(o), 2))
        sj = seg.to_json()
        for ym, grp in T.groupby("ym"):
            tasks.append((road[0], ym, grp.head(3).to_dict("records"), sj))
    return tasks, meta


PRIORITY = ["ftportal_hima", "muyembe", "rwenkunye", "kigumba", "luwero", "koboko", "tirinyi", "kamonkoli",
            "bumbobi", "hoima", "bukakata", "namagumba", "kyenjojo", "ishaka"]


def main(workers=24, only_priority=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    tasks, meta = plan()
    if only_priority:
        tasks = [t for t in tasks if t[0] in PRIORITY]
    tasks.sort(key=lambda t: PRIORITY.index(t[0]) if t[0] in PRIORITY else len(PRIORITY))  # validation roads first
    print(json.dumps(meta, indent=1), flush=True)
    print(len(tasks), "road-months", flush=True)
    tok = token()  # one SAS token (valid ~24 h) shared by all workers; the token endpoint rate-limits
    with ThreadPoolExecutor(max_workers=workers) as ex:  # I/O-bound; GDAL releases the GIL during reads
        futs = [ex.submit(month_task, *t, tok) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            rid, ym, msg = fu.result()
            if msg != "cached" and (i % 50 == 0 or not msg.startswith("ok")):
                print(f"[{i + 1}/{len(futs)}] {rid} {ym} {msg}", flush=True)
    frames = [pd.read_parquet(f).assign(id=f.stem.rsplit("_", 1)[0]) for f in sorted(CACHE.glob("*.parquet"))]
    D = pd.concat([f for f in frames if len(f)], ignore_index=True)
    D.to_parquet(P / "road_s1.parquet", index=False)
    (ROOT / "paper" / "road_s1_meta.json").write_text(json.dumps(meta, indent=2))
    print(D.groupby("id").ym.nunique().to_string())


if __name__ == "__main__":
    main()
