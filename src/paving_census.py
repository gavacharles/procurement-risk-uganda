"""Paper 3, extension (3): a national paving census reconciled with the open contracting record.

Instead of starting from contracts, start from the ground. Sample every OpenStreetMap trunk, primary and
secondary road in Uganda every STEP_KM, classify each point as asphalt or unpaved at the start of the archive
(dry seasons 2017-2018) and at the end (2025b, 2026a) with the road-surface classifier of road_progress.py,
and call a point "newly paved" if it was unpaved at the start and asphalt at the end. Newly paved points are
grouped into stretches along the same OSM way name/ref.

To keep downloads feasible at national scale (tens of thousands of points), the census reads Sentinel-2's
true-colour product (TCI: one 8-bit file with three channels per scene, about a tenth of the six-band data)
and uses an RGB classifier trained on the same OpenStreetMap labels; its accuracy is reported separately.

Reconciliation: a newly paved stretch is "matched" if it lies within MATCH_M of a route of a contract in the
paper 3 sample, or if its OSM name/ref, or the places at its ends, appear in the title of a works contract in
the PPDA open contracting feed (2014-2026). The unmatched share is an upper bound on paving that the open
record does not show (it also contains classification errors, paving financed outside PPDA's feed, and works
procured before 2014).

Out: data/processed/paving_census_points.parquet, paper/paving_census.json
"""
from __future__ import annotations

import json
import math
import re
import sys
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import road_progress as rp  # noqa: E402
import satellite as sat  # noqa: E402
import semantic_change as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
CC = sat.CACHE / "census"
STEP_KM = 4.0
MATCH_M = 1500
WINDOWS = {"start": [("2017a", "2016-12-01", "2017-02-28"), ("2018a", "2017-12-01", "2018-02-28")],
           "end": [("2026a", "2025-12-01", "2026-02-28"), ("2025b", "2025-06-01", "2025-08-31")]}
UGANDA = (-1.5, 29.5, 4.3, 35.1)
TRAIN_PER_CLASS = 900
MIN_PTS = 2


def osm_major():
    def go():
        s, w, n, e = UGANDA
        q = (f'[out:json][timeout:300];way["highway"~"^(trunk|primary|secondary)$"]({s},{w},{n},{e});out tags geom;')
        r = sat.http_json("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode())
        return [dict(id=x["id"], name=x.get("tags", {}).get("name"), ref=x.get("tags", {}).get("ref"),
                     highway=x.get("tags", {}).get("highway"), surface=x.get("tags", {}).get("surface"),
                     geom=[(g["lon"], g["lat"]) for g in x.get("geometry", [])]) for x in r.get("elements", [])]
    return sat.cached("osm major roads uganda", go)


def sample_points(ways, step=STEP_KM):
    """Points every `step` km along each way, carrying the remainder within a way (short ways get one point)."""
    rows = []
    for w in ways:
        g = w["geom"]
        if len(g) < 2:
            continue
        acc, nxt, total = 0.0, step / 2, 0.0
        segs = []
        for (x0, y0), (x1, y1) in zip(g[:-1], g[1:]):
            d = rp.km_between((x0, y0), (x1, y1))
            segs.append((x0, y0, x1, y1, d))
            total += d
        if total < step / 2:
            nxt = total / 2
        for x0, y0, x1, y1, d in segs:
            while d > 0 and acc + d >= nxt:
                f = (nxt - acc) / d
                rows.append(dict(way=w["id"], name=w["name"], ref=w["ref"], highway=w["highway"], osm_surface=w["surface"],
                                 lon=x0 + f * (x1 - x0), lat=y0 + f * (y1 - y0), lon2=x1, lat2=y1, km_on_way=nxt))
                nxt += step
            acc += d
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- true-colour (TCI) features
def tci_href(scene):
    b = scene["blue"]
    return b.replace("/B02.tif", "/TCI.tif") if b.startswith("https://") and b.endswith("/B02.tif") else None


def tci_sample(scene, lonlats):
    import rasterio
    from pyproj import Transformer
    out = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="3"):
        with rasterio.open(tci_href(scene)) as src:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xy = [tr.transform(*p) for p in lonlats]
            v = np.array([s_ for s_ in src.sample(xy)], dtype=float)
            out["r"], out["g"], out["b"] = v[:, 0] / 255, v[:, 1] / 255, v[:, 2] / 255
        with rasterio.open(scene["scl"]) as src:
            out["scl"] = np.array([s_[0] for s_ in src.sample(xy)], dtype=float)
    return out


def rgb_indices(r, g, b):
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    return dict(r=r, g=g, b=b, bright=(r + g + b) / 3, redness=r / (b + 1e-3), rg=r - g, sat=(mx - mn) / (mx + 1e-3))


def tci_features(points, scenes):
    feats = pd.DataFrame(index=points.index)
    for scn in sorted(scenes, key=lambda s_: s_["cloud"]):
        if not tci_href(scn):
            continue
        bb = scn["bbox"]
        todo = points[(points.lon > bb[0]) & (points.lon < bb[2]) & (points.lat > bb[1]) & (points.lat < bb[3])]
        if len(feats.columns):
            todo = todo[~todo.index.isin(feats.dropna().index)]
        if todo.empty:
            continue
        locs = []
        for r in todo.itertuples():
            n30 = sc.offsets(r.lon, r.lat, r.lon2, r.lat2, 30)
            n60 = sc.offsets(r.lon, r.lat, r.lon2, r.lat2, 60)
            locs += [(r.lon, r.lat), n30[0], n30[1], n60[0], n60[1]]
        try:
            v = tci_sample(scn, locs)
        except Exception as e:
            print("  read fail", str(e)[:80], flush=True)
            continue
        k = len(todo)
        R, Gc, B, S = (v[c].reshape(k, 5) for c in ("r", "g", "b", "scl"))
        ok = np.isin(S[:, 0], [4, 5, 7]) & ((R[:, 0] + Gc[:, 0] + B[:, 0]) > 0)
        road = rgb_indices(R[:, 0], Gc[:, 0], B[:, 0])
        b30 = rgb_indices((R[:, 1] + R[:, 2]) / 2, (Gc[:, 1] + Gc[:, 2]) / 2, (B[:, 1] + B[:, 2]) / 2)
        b60 = rgb_indices((R[:, 3] + R[:, 4]) / 2, (Gc[:, 3] + Gc[:, 4]) / 2, (B[:, 3] + B[:, 4]) / 2)
        f = {}
        for key in road:
            f[f"road_{key}"] = road[key]
            f[f"d30_{key}"] = road[key] - b30[key]
            f[f"d60_{key}"] = road[key] - b60[key]
        F = pd.DataFrame(f, index=todo.index)
        F[~ok] = np.nan
        feats = F.combine_first(feats) if len(feats.columns) else F
    return feats


def train_rgb():
    """RGB classifier on OSM-labelled roads (same labels as road_progress), Dec 2025 - Feb 2026."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, f1_score
    f = sat.CACHE / f"train_tci_2026a_{TRAIN_PER_CLASS}.parquet"
    pts = sc.osm_points(per_class=TRAIN_PER_CLASS)
    if f.exists():
        X = pd.read_parquet(f)
    else:
        scenes, seen = [], set()
        for (cx, cy), _ in pts.groupby([np.floor(pts.lon), np.floor(pts.lat)]):
            for s_ in rp.scenes_window([float(cx), float(cy), float(cx) + 1, float(cy) + 1], "2025-12-01", "2026-02-28", max_cloud=30, per_tile=2):
                if s_["blue"] not in seen:
                    seen.add(s_["blue"])
                    scenes.append(s_)
        X = tci_features(pts, scenes)
        X.to_parquet(f)
    d = pts.join(X).dropna(subset=["road_r"])
    cols = [c for c in X.columns if c.startswith(("road_", "d30_", "d60_"))]
    y = (d.cls == "asphalt").astype(int).to_numpy()
    pred = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(d, y, groups=d.way):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d.iloc[tr][cols], y[tr])
        pred[te] = m.predict(d.iloc[te][cols])
    info = dict(n=int(len(d)), cv_accuracy=float(accuracy_score(y, pred)), f1_asphalt=float(f1_score(y, pred)),
                baseline=float(max(y.mean(), 1 - y.mean())))
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d[cols], y)
    print("RGB classifier:", info, flush=True)
    return model, cols, info


def rgb_states(F, model, cols):
    st = pd.Series("nodata", index=F.index, dtype=object)
    ok = F.road_r.notna() if "road_r" in F.columns else pd.Series(False, index=F.index)
    # vegetation over the road: green dominant and low redness
    veg = ok & (F.road_g > F.road_r + 0.02) & (F.road_g > F.road_b + 0.02)
    st[veg] = "vegetated"
    rest = ok & ~veg
    if rest.any():
        st[rest] = np.where(model.predict(F.loc[rest, cols]) == 1, "asphalt", "unpaved")
    return st


def cell_task(cell, period, pts_json):
    CC.mkdir(parents=True, exist_ok=True)
    f = CC / f"{cell}_{period[0]}.parquet"
    if f.exists():
        return cell, period[0], "cached"
    pts = pd.read_json(pts_json)
    cx, cy = (float(v) for v in cell.split("_"))
    try:
        scenes = rp.scenes_window([cx, cy, cx + 1, cy + 1], period[1], period[2], max_cloud=40, per_tile=3)
        F = tci_features(pts, scenes) if scenes else pd.DataFrame(index=pts.index)
    except Exception as e:
        return cell, period[0], f"fail {e}"
    F = F.reindex(pts.index)
    F["pid"] = pts.pid.values
    F.to_parquet(f)
    return cell, period[0], f"ok {int(F.get('road_r', pd.Series(dtype=float)).notna().sum())}/{len(pts)}"


def extract(pts, workers=8):
    pts = pts.copy()
    pts["cell"] = np.floor(pts.lon).astype(int).astype(str) + "_" + np.floor(pts.lat).astype(int).astype(str)
    tasks = [(c, p, g.to_json()) for c, g in pts.groupby("cell") for k in WINDOWS for p in WINDOWS[k]]
    print(len(tasks), "cell-period tasks", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(cell_task, *t) for t in tasks])):
            c, per, msg = fu.result()
            if msg != "cached":
                print(f"[{i + 1}/{len(tasks)}] {c} {per} {msg}", flush=True)


def classify(pts):
    model, cols, info = train_rgb()
    pts = pts.copy()
    pts["cell"] = np.floor(pts.lon).astype(int).astype(str) + "_" + np.floor(pts.lat).astype(int).astype(str)
    out = {}
    for k, pers in WINDOWS.items():
        votes = []
        for p in pers:
            parts = [pd.read_parquet(CC / f"{c}_{p[0]}.parquet") for c in pts.cell.unique() if (CC / f"{c}_{p[0]}.parquet").exists()]
            if not parts:
                continue
            F = pd.concat(parts).set_index("pid")
            votes.append(rgb_states(F, model, cols).rename(p[0]))
        V = pd.concat(votes, axis=1).reindex(pts.pid)
        a = (V == "asphalt").sum(axis=1).to_numpy()
        u = (V == "unpaved").sum(axis=1).to_numpy()
        out[k] = np.where(a + u == 0, "nodata", np.where(a > u, "asphalt", np.where(u > a, "unpaved", "tie")))
    pts["state_start"], pts["state_end"] = out["start"], out["end"]
    return pts, info


# ---------------------------------------------------------------- reconciliation
def norm(s):
    return re.sub(r"[^a-z ]", " ", str(s).lower())


def contract_titles():
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    t = (w.title.fillna("") + " " + w.contract_title.fillna("") + " " + w.description.fillna("")).map(norm)
    return t[t.str.contains("road|upgrad|bitum|tarmac|paved|seal")].tolist()


def place_names(name, ref):
    out = set()
    for part in re.split(r"[-–/,]| to ", str(name or "")):
        part = re.sub(r"\b(road|rd|highway|street|avenue|bypass)\b", "", part.lower()).strip()
        if len(part) >= 4:
            out.add(part)
    return out


def reconcile(pts):
    new = pts[(pts.state_start == "unpaved") & (pts.state_end == "asphalt")].copy()
    # stretches: consecutive newly paved points on the same OSM way name (or way id if unnamed)
    new["key"] = new.name.fillna("way" + new.way.astype(str))
    # contract routes from the paper 3 sample
    routes = []
    for road in rp.ROADS:
        g = rp.geometry(road)
        if g:
            routes.append((road[0], np.array(g["coords"])))

    def near_route(lon, lat):
        kx, ky = 111320 * math.cos(math.radians(lat)), 110570
        for rid, L in routes:
            d = np.hypot((L[:, 0] - lon) * kx, (L[:, 1] - lat) * ky).min()
            if d < MATCH_M:
                return rid
        return None
    new["route_match"] = [near_route(r.lon, r.lat) for r in new.itertuples()]
    titles = contract_titles()

    def title_match(name, ref):
        names = place_names(name, ref)
        hits = [n for n in names if any(n in t for t in titles)]
        return ";".join(sorted(hits)) if len(hits) >= 1 and len(names) >= 1 else None
    new["title_match"] = [title_match(r.name, r.ref) for r in new.itertuples()]
    new["matched"] = new.route_match.notna() | new.title_match.notna()
    return new


def osm_uganda_ids():
    """Ids of the major-road ways that lie inside Uganda's national boundary (the bounding box also covers
    parts of Rwanda, DRC, South Sudan, Kenya and Tanzania)."""
    def go():
        q = ('[out:json][timeout:300];area["ISO3166-1"="UG"][admin_level=2]->.a;'
             'way["highway"~"^(trunk|primary|secondary)$"](area.a);out ids;')
        r = sat.http_json("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode())
        return [x["id"] for x in r.get("elements", [])]
    return set(sat.cached("osm major road ids inside uganda", go))


def way_lengths(ways):
    return {w["id"]: sum(rp.km_between(a, b) for a, b in zip(w["geom"][:-1], w["geom"][1:])) for w in ways}


def clusters(pts, frm, to, radius_km=STEP_KM * 1.5, min_pts=2):
    """Stretches: points that went from `frm` to `to`, linked when within radius_km of each other (OSM splits
    roads into many short ways, so linking by distance rather than by way id). A stretch needs min_pts points.
    Returns a cluster id per point (-1 = not in a qualifying stretch)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    flag = ((pts.state_start == frm) & (pts.state_end == to)).to_numpy()
    lab = np.full(len(pts), -1)
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return pd.Series(lab, index=pts.index)
    lat0 = float(pts.lat.mean())
    xy = np.c_[pts.lon.to_numpy()[idx] * 111.32 * math.cos(math.radians(lat0)), pts.lat.to_numpy()[idx] * 110.57]
    A = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), radius_km, output_type="coo_matrix")
    _, comp = connected_components(A, directed=False)
    sizes = np.bincount(comp)
    keep = sizes[comp] >= min_pts
    lab[idx[keep]] = comp[keep]
    return pd.Series(lab, index=pts.index)


def stretch_summary(valid):
    """Stretch rule and false-change benchmark. Real roads almost never go from asphalt back to unpaved over
    2017-2026, so asphalt->unpaved stretches are mostly classifier noise; with errors roughly symmetric, their
    length estimates how much of the unpaved->asphalt signal noise alone would produce."""
    res = {}
    for k in (1, 2, 3, 4):
        cn = clusters(valid, "unpaved", "asphalt", min_pts=k)
        cr = clusters(valid, "asphalt", "unpaved", min_pts=k)
        nk, rk = float(valid.w_km[cn >= 0].sum()), float(valid.w_km[cr >= 0].sum())
        res[f"min_pts_{k}"] = dict(new_km=nk, new_stretches=int(cn[cn >= 0].nunique()),
                                   reverted_km=rk, reverted_stretches=int(cr[cr >= 0].nunique()),
                                   noise_ratio=rk / nk if nk else None, net_new_km=nk - rk)
    return res


def reverse_places(lon, lat):
    """Place names around a point from Nominatim reverse geocoding at town (zoom 12) and village (zoom 14) level."""
    out = set()
    for z in (12, 14):
        def go(z=z):
            u = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
                dict(lat=f"{lat:.4f}", lon=f"{lon:.4f}", zoom=z, format="json", addressdetails=1))
            import time
            time.sleep(1.1)
            return sat.http_json(u)
        r = sat.cached(f"reverse {lon:.4f} {lat:.4f} z{z}", go)
        a = (r or {}).get("address", {})
        for k in ("village", "town", "city", "hamlet", "suburb", "municipality", "neighbourhood", "city_district"):
            if a.get(k):
                out.add(a[k])
    return out


def place_match(ns, st, titles):
    """Name each stretch by the places at its two ends and its middle; a stretch is matched by places if one
    works-contract title contains at least two of those names (strict), or at least one (loose)."""
    strict, loose, names = {}, {}, {}
    for sid, g in ns.groupby("stretch"):
        xy = g[["lon", "lat"]].to_numpy()
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        i, j = np.unravel_index(d.argmax(), d.shape)
        pl = set()
        for lon, lat in (xy[i], xy[j], xy.mean(axis=0)):
            pl |= reverse_places(float(lon), float(lat))
        for nm in g.name.dropna().unique():
            pl |= place_names(nm, None)
        toks = {norm(x).strip() for x in pl}
        toks = {t for t in toks if len(t) >= 4}
        hits = [sum(t in ti for t in toks) for ti in titles]
        strict[sid], loose[sid] = max(hits, default=0) >= 2, max(hits, default=0) >= 1
        names[sid] = "; ".join(sorted(pl))
    st["places"] = pd.Series(names)
    st["place_match_strict"] = pd.Series(strict).astype(bool)
    st["place_match_loose"] = pd.Series(loose).astype(bool)
    return st


def main():
    ways = osm_major()
    pts = sample_points(ways)
    pts["pid"] = np.arange(len(pts))
    print(f"{len(ways)} OSM ways, {len(pts)} points every {STEP_KM} km", flush=True)
    if "--classify" not in sys.argv:
        extract(pts)
    pts, rgb_info = classify(pts)
    # keep Uganda only; weight each point by its share of its way's length
    ug = osm_uganda_ids()
    L = way_lengths(ways)
    pts = pts[pts.way.isin(ug)].copy()
    pts["w_km"] = pts.way.map(L) / pts.groupby("way").pid.transform("size")
    pts.to_parquet(P / "paving_census_points.parquet", index=False)
    valid = pts[(pts.state_start.isin(["asphalt", "unpaved"])) & (pts.state_end.isin(["asphalt", "unpaved"]))].copy()
    valid["stretch"] = clusters(valid, "unpaved", "asphalt", min_pts=MIN_PTS)
    new = reconcile(pts)
    new = new.merge(valid[["pid", "stretch"]], on="pid", how="left")
    ns = new[new.stretch >= 0].copy()
    # a stretch is matched if any of its points matches; name a stretch by its commonest OSM name
    g = ns.groupby("stretch")
    st = pd.DataFrame(dict(km=g.w_km.sum(), points=g.size(), matched=g.matched.any(),
                           by_route=g.route_match.apply(lambda v: v.notna().any()),
                           name=g.name.agg(lambda v: v.dropna().mode().iloc[0] if v.notna().any() else None),
                           highway=g.highway.agg(lambda v: v.mode().iloc[0]),
                           lon=g.lon.mean(), lat=g.lat.mean()))
    st = place_match(ns, st, contract_titles())
    st["matched_any"] = st.matched | st.place_match_strict
    st.to_parquet(P / "paving_census_stretches.parquet")
    wkm = lambda d: float(d.w_km.sum())
    res = dict(
        rgb_classifier=rgb_info, step_km=STEP_KM, osm_ways_uganda=int(pts.way.nunique()), points=int(len(pts)),
        km_network=float(sum(L[w] for w in pts.way.unique())), km_valid=wkm(valid),
        points_valid_both=int(len(valid)),
        start_asphalt_share_km=float(valid.w_km[valid.state_start == "asphalt"].sum() / wkm(valid)),
        end_asphalt_share_km=float(valid.w_km[valid.state_end == "asphalt"].sum() / wkm(valid)),
        point_level=dict(newly_paved_km=wkm(new), reverted_km=wkm(valid[(valid.state_start == "asphalt") & (valid.state_end == "unpaved")]),
                         matched_share_km=float(new.w_km[new.matched].sum() / wkm(new)) if len(new) else None),
        stretches=stretch_summary(valid),
        stretch_rule=f"{MIN_PTS}+ newly paved points linked within {STEP_KM * 1.5:.0f} km",
        stretch_n=int(len(st)), stretch_km=float(st.km.sum()),
        stretch_matched_n=int(st.matched.sum()), stretch_matched_km=float(st.km[st.matched].sum()),
        stretch_matched_by_route_km=float(st.km[st.by_route].sum()),
        stretch_matched_share_km=float(st.km[st.matched].sum() / st.km.sum()) if len(st) else None,
        stretch_unmatched_km=float(st.km[~st.matched].sum()),
        with_places=dict(matched_km=float(st.km[st.matched_any].sum()), matched_share_km=float(st.km[st.matched_any].sum() / st.km.sum()),
                         unmatched_km=float(st.km[~st.matched_any].sum()), unmatched_n=int((~st.matched_any).sum()),
                         loose_matched_share_km=float(st.km[st.matched | st.place_match_loose].sum() / st.km.sum())),
        by_highway={h: dict(km_valid=wkm(d), new_km=float(d.w_km[d.stretch >= 0].sum())) for h, d in valid.groupby("highway")},
        stretches_list=st.sort_values("km", ascending=False).round(3).reset_index().to_dict("records"),
    )
    new.to_parquet(P / "paving_census_newly_paved.parquet", index=False)
    (ROOT / "paper" / "paving_census.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "stretches_list"}, indent=1, default=float))


if __name__ == "__main__":
    main()
