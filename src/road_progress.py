"""Satellite-observed construction progress and delay for contracted road upgrades (Sentinel-2, 2017-2026).

Question: given only open procurement data (what was tendered, when), can free satellite imagery
tell us whether, when and how fast a contracted road was paved, and whether it is late?

Pipeline
  1. Registry (ROADS): road-paving contracts from PPDA open contracting data whose titles name the
     endpoints, plus controls (already-paved roads being rehabilitated/reconstructed).
  2. Geometry: geocode each waypoint (Nominatim, nearest candidate to the neighbouring waypoint),
     route through all waypoints (OSRM). Routed length is checked against the contract length.
  3. Segments: the route is cut into equal segments (1 km, or longer so that there are at most
     MAX_SEG); one sample point per segment centre, plus 30 m / 60 m side points.
  4. Periods: two drier windows per year, Dec-Feb ("a") and Jun-Aug ("b"), 2016/17 .. 2026.
     For each road x period the least-cloudy Sentinel-2 L2A scenes are read at the points
     (SCL cloud mask; next scene fills gaps).
  5. Classifier: gradient boosting, asphalt vs unpaved, trained on OpenStreetMap surface tags
     in two recent periods (dry + mid-year) so it sees both seasons; grouped CV by OSM way.
  6. Segment state over time: a segment is "sealed" from the first period after which at least 70% of
     its valid readings are asphalt (change point; paving is not undone within the window).
     Road progress S(t) = share of segments sealed. Rehabilitation controls also get the raw share,
     since works on an existing road can strip the surface for a time.
  7. Milestones: onset (S - S0 >= 0.10), half (S >= 0.5), completion (S >= 0.80).
     Not completed by the last image -> right-censored.
  8. Benchmark: expected completion = tender invitation + signing lag + planned duration
     (by length). Contract dates from data/manual/road_contract_dates.csv override the benchmark.

A slow or incomplete road is a prompt for a site visit, not a finding of wrongdoing.

Out: data/processed/road_progress_{segments,series,roads}.parquet, paper/road_progress.json,
     posts/media/chart_road_progress*.png
"""
from __future__ import annotations

import json
import math
import sys
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import satellite as sat  # noqa: E402
import semantic_change as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
PC = sat.CACHE / "progress"
MAX_SEG = 40
YEARS = range(2017, 2027)
LATEST = "2026b"
PARTIAL = {"kalangala"}  # routed section is a known part of the contract (Kalangala town - Mulabana)

# id, label, waypoints (geocoder queries, in order), contract km, ocid, kind
#   kind: upgrade | control_paved (already paved; rehab/reconstruction)
ROADS = [
    ("tirinyi", "Tirinyi–Pallisa–Kumi", ["Tirinyi, Kibuku", "Pallisa Town", "Kumi"], 67, "ocds-rdvc92-1504903556", "upgrade"),
    ("kamonkoli", "Pallisa–Kamonkoli", ["Pallisa Town", "Kamonkoli, Budaka"], 44, "ocds-rdvc92-1504903556", "upgrade"),
    ("bumbobi", "Bumbobi–Bubulo–Lwakhakha", ["Bumbobi, Mbale", "Bubulo", "Lwakhakha"], 44.5, "ocds-rdvc92-1504905426", "upgrade"),
    ("muyembe", "Muyembe–Nakapiripirit", ["Muyembe, Bulambuli", "Nakapiripirit"], 92, "ocds-rdvc92-991582809276", "upgrade"),
    ("kigumba", "Kigumba–Bulima–Kabwoya", ["Kigumba", "Bulima, Masindi", "Kabwoya"], 135, "ocds-rdvc92-1504953965", "upgrade"),
    # ("rushere", "Rushere–Nshwerenkye", ["Rushere", "Nshwerenkye"], 11.1, "ocds-rdvc92-1504904235", "upgrade"),  # endpoint not found in OSM / unreliable
    # ("kashenyi", "Kashenyi–Mitooma", ["Kashenyi, Bushenyi", "Mitooma"], 11.6, "ocds-rdvc92-1504955378", "upgrade"),  # endpoint not found in OSM / unreliable
    ("seeta", "Seeta–Namugongo", ["Seeta, Mukono", "Namugongo"], 8, "ocds-rdvc92-1504903614", "upgrade"),
    ("rwenkunye", "Rwenkunye–Apac–Lira–Puranga", ["Kafu, Nakasongola", "Apac", "Lira", "Puranga"], 191, "ocds-rdvc92-461601456821", "upgrade"),
    ("fortkamwenge", "Fort Portal–Kamwenge", ["Fort Portal", "Kamwenge"], 66, "ocds-rdvc92-1504954998", "upgrade"),
    ("bukakata", "Masaka–Bukakata", ["Nyendo, Masaka", "Bukakata"], 41, "ocds-rdvc92-1506061762", "upgrade"),
    ("hoima", "Hoima–Butiaba–Wanseko", ["Hoima", "Butiaba", "Wanseko"], 111, "ocds-rdvc92-1511770302", "upgrade"),
    # ("lwenza", "Lwenza–Kigo", ["Lwenza, Wakiso", "Kigo, Wakiso"], 6, "ocds-rdvc92-621575965634", "upgrade"),  # endpoint not found in OSM / unreliable
    ("kalule", "Kalule–Bamunanika", ["Kalule, Luweero", "Bamunanika"], 11, "ocds-rdvc92-831581584380", "upgrade"),
    ("bunyonyi", "Kabale–Lake Bunyonyi", ["Kabale", "Lake Bunyonyi"], 10, "ocds-rdvc92-911721121980", "upgrade"),
    ("mgahinga", "Kisoro–Mgahinga", ["Kisoro", "Ntebeko, Kisoro"], 14, "ocds-rdvc92-911721121980", "upgrade"),
    ("koboko", "Koboko–Yumbe–Moyo", ["Koboko", "Yumbe", "Moyo"], 103, "ocds-rdvc92-501717667056", "upgrade"),
    # ("kafunta", "Kafunta–Buwampa", ["Kafunta, Buikwe", "Buwampa, Buikwe"], 9.5, "ocds-rdvc92-731650967747", "upgrade"),  # endpoint not found in OSM / unreliable
    ("luwero", "Luwero–Butalangu", ["Luwero", "Butalangu, Nakaseke"], 30, "ocds-rdvc92-151667464588", "upgrade"),
    ("namagumba", "Namagumba–Budadiri–Nalugugu", ["Namagumba, Mbale", "Budadiri", "Nalugugu, Sironko"], 39, "ocds-rdvc92-321704274286", "upgrade"),
    ("kyenjojo", "Kyenjojo–Bwizi–Rwamwanja", ["Kihura, Kyenjojo", "Bwizi, Kamwenge", "Rwamwanja, Kamwenge", "Kahunge"], 105, "ocds-rdvc92-521750666838", "upgrade"),
    # ("boroboro", "Lira–Boroboro (low-cost seal)", ["Lira", "Boroboro, Lira"], 5, "ocds-rdvc92-1504919820", "upgrade"),  # endpoint not found in OSM / unreliable
    ("senge", "Kayunga–Senge–Kawanda (Wakiso)", ["Kayunga, Wakiso", "Senge, Wakiso", "Kawanda"], 6.5, "ocds-rdvc92-711782991733", "upgrade"),
    ("namulanda", "Namulanda–Bweya–Kajjansi", ["Namulanda", "Bweya", "Kajjansi"], 9.5, "ocds-rdvc92-121782123071", "upgrade"),
    ("kisozi", "Kisozi–Kitemu", ["Kisozi, Wakiso", "Kitemu, Wakiso"], 4.5, "ocds-rdvc92-581724838918", "upgrade"),
    ("katooke", "Nansana–Wamala–Katooke", ["Nansana", "Katooke, Wakiso"], 4.87, "ocds-rdvc92-171744731168", "upgrade"),
    ("bududa", "Bubulo–Bududa", ["Bubulo", "Bududa"], 28, "ocds-rdvc92-251717665214", "upgrade"),
    # ("nkokonjeru", "Mbale–Nkokonjeru", ["Mbale", "Nkokonjeru, Bulambuli"], 21, "ocds-rdvc92-251717665214", "upgrade"),  # endpoint not found in OSM / unreliable
    ("kalangala", "Lunku–Kalangala–Mulabana", ["Kalangala", "Mulabana"], 65.3, "ocds-rdvc92-571695714749", "upgrade"),
    # controls: already paved before the contract
    ("ftportal_hima", "Fort Portal–Hima (control)", ["Fort Portal", "Hima, Kasese"], 55, None, "control_paved"),
    ("mityana", "Mityana–Mubende (control)", ["Mityana", "Mubende"], 86, "ocds-rdvc92-261612944200", "control_paved"),
    ("mutukula", "Masaka–Kyotera–Mutukula (control)", ["Masaka", "Kyotera", "Mutukula"], 89.5, "ocds-rdvc92-761715947925", "control_paved"),
    ("matugga", "Matugga–Semuto–Kapeeka (control)", ["Matugga", "Semuto", "Kapeeka"], 41, "ocds-rdvc92-251697104924", "control_paved"),
    ("ishaka", "Ishaka–Katunguru (control)", ["Ishaka", "Katunguru, Rubirizi"], 58, None, "control_paved"),
]


# ---------------------------------------------------------------- periods
def period_offset(p: str) -> float:
    """BOA offset for converting v1 cached features: 0 (see semantic_change.boa_offset)."""
    return 0.0


def periods():
    out = []
    for y in YEARS:
        out.append((f"{y}a", f"{y - 1}-12-01", f"{y}-02-28", y + 0.08))  # mid-Jan
        out.append((f"{y}b", f"{y}-06-01", f"{y}-08-31", y + 0.58))      # mid-Jul
    return out


PERIODS = periods()
PT = {p[0]: p[3] for p in PERIODS}


# ---------------------------------------------------------------- geometry
def geocode_all(q: str):
    def go():
        import time
        time.sleep(1.1)
        u = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
            {"q": q + ", Uganda", "format": "json", "limit": 6, "countrycodes": "ug"})
        return [[float(r["lon"]), float(r["lat"]), r.get("display_name", "")] for r in sat.http_json(u)]
    return sat.cached("geoall " + q, go)


def km_between(a, b):
    return math.hypot((a[0] - b[0]) * 111.32 * math.cos(math.radians(a[1])), (a[1] - b[1]) * 110.57)


def pick_waypoints(qs):
    cands = [geocode_all(q) or [] for q in qs]
    if any(not c for c in cands):
        return None
    # candidates whose name does not contain the place searched for (e.g. "Pallisa" -> a point labelled
    # "Kibuku") are penalised heavily, so a short wrong route cannot beat the right one
    def norm(x):  # spelling variants: Luwero / Luweero, Sseeta / Seeta
        import re
        return re.sub(r"([aeiou])\1", r"\1", re.sub(r"([bcdfghjklmnpqrstvwxyz])\1", r"\1", x.lower()))
    names = [norm(q.split(",")[0].replace(" Town", "")) for q in qs]
    pen = [[0 if names[i] in norm(c[2]) else 200 for c in cand] for i, cand in enumerate(cands)]
    # choose the combination (first candidate preferred) minimising total straight-line length
    import itertools
    best, best_cost = None, 1e18
    for combo in itertools.product(*[range(min(4, len(c))) for c in cands]):
        pts = [cands[i][j] for i, j in enumerate(combo)]
        cost = sum(km_between(a, b) for a, b in zip(pts[:-1], pts[1:])) + 3 * sum(combo) + sum(pen[i][j] for i, j in enumerate(combo))
        if cost < best_cost:
            best, best_cost = pts, cost
    return [p[:2] for p in best]


def route_chain(wps):
    def go():
        u = ("https://router.project-osrm.org/route/v1/driving/" + ";".join(f"{x},{y}" for x, y in wps)
             + "?overview=full&geometries=geojson")
        rt = sat.http_json(u)["routes"][0]
        return dict(coords=rt["geometry"]["coordinates"], km=rt["distance"] / 1000)
    return sat.cached("routechain " + json.dumps(wps), go)


def segments(coords, km_total):
    """Equal-length segments; returns centre point + a nearby point for direction."""
    cum = [0.0]
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        cum.append(cum[-1] + km_between((x0, y0), (x1, y1)))
    cum = np.array(cum)
    step = max(1.0, cum[-1] / MAX_SEG)
    centres = np.arange(step / 2, cum[-1] - 0.3, step)
    out = []
    for i, c in enumerate(centres):
        k = int(np.searchsorted(cum, c))
        k = min(max(k, 1), len(coords) - 1)
        x0, y0 = coords[k - 1]
        x1, y1 = coords[k]
        f = (c - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-9)
        out.append(dict(seg=i, km_at=round(float(c), 2), lon=x0 + f * (x1 - x0), lat=y0 + f * (y1 - y0), lon2=x1, lat2=y1))
    return pd.DataFrame(out)


def geometry(road):
    rid, label, qs, km, ocid, kind = road
    wps = pick_waypoints(qs)
    if not wps:
        return None
    rt = route_chain(wps)
    return dict(id=rid, waypoints=wps, coords=rt["coords"], routed_km=rt["km"], ratio=rt["km"] / km)


# ---------------------------------------------------------------- imagery
def scenes_window(bbox, start, end, max_cloud=70, per_tile=4):
    def go():
        body = json.dumps({"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 100,
                           "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                           "query": {"eo:cloud_cover": {"lt": max_cloud}},
                           "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]}).encode()
        r = sat.http_json(f"{sat.STAC}/search", data=body)
        tiles = {}
        for f in r.get("features", []):
            if not f["assets"]["blue"]["href"].startswith("https://"):
                continue
            t = f["properties"].get("grid:code")
            tiles.setdefault(t, [])
            if len(tiles[t]) < per_tile:
                tiles[t].append(dict(id=f["id"], cloud=f["properties"]["eo:cloud_cover"], bbox=f["bbox"],
                                     **{b: f["assets"][b]["href"] for b in sc.BANDS}))
        return sorted((s for v in tiles.values() for s in v), key=lambda s: s["cloud"])
    return sat.cached(f"win {bbox} {start} {end} {max_cloud} {per_tile}", go)


def features_task(rid, period, pts_json):
    """One road x period. Cached to parquet so reruns resume."""
    PC.mkdir(parents=True, exist_ok=True)
    f = PC / f"{rid}_{period}.parquet"
    if f.exists():
        return rid, period, "cached"
    pts = pd.read_json(pts_json)
    tag, start, end, _ = next(p for p in PERIODS if p[0] == period)
    pad = 0.01
    bbox = [pts.lon.min() - pad, pts.lat.min() - pad, pts.lon.max() + pad, pts.lat.max() + pad]
    try:
        scs = scenes_window(bbox, start, end)
        F = sc.features_from_scenes(pts, scs) if scs else pd.DataFrame(index=pts.index)
    except Exception as e:
        return rid, period, f"fail {e}"
    F = F.reindex(pts.index)
    F["seg"] = pts.seg.values
    F.to_parquet(f)
    return rid, period, f"ok n={int(F.get('road_ndvi', pd.Series(dtype=float)).notna().sum())}"


def fill_task(rid, period, pts_json):
    """Second pass for points still empty after the 4 least-cloudy scenes per tile (cloud, or the edge of
    the swath when those scenes share one orbit): try the next 6 scenes per tile for those points only."""
    f = PC / f"{rid}_{period}.parquet"
    if not f.exists():
        return rid, period, "missing"
    F = pd.read_parquet(f)
    if "filled" in F.columns:
        return rid, period, "done"
    pts = pd.read_json(pts_json)
    miss = F.road_ndvi.isna().to_numpy() if "road_ndvi" in F.columns else np.ones(len(pts), bool)
    if miss.mean() > 0.1:
        tag, start, end, _ = next(p for p in PERIODS if p[0] == period)
        pad = 0.01
        bbox = [pts.lon.min() - pad, pts.lat.min() - pad, pts.lon.max() + pad, pts.lat.max() + pad]
        try:
            tried = {s["id"] for s in scenes_window(bbox, start, end)}
            more = [s for s in scenes_window(bbox, start, end, per_tile=10) if s["id"] not in tried]
            sub = pts[miss]
            G = sc.features_from_scenes(sub, more) if more else pd.DataFrame(index=sub.index)
        except Exception as e:
            return rid, period, f"fail {e}"
        if "road_ndvi" in G.columns and G.road_ndvi.notna().any():
            F = sc.harmonise_cached(F.drop(columns=["seg"], errors="ignore"), 0.0)
            G = G.reindex(columns=F.columns)
            F.loc[G.index[G.road_ndvi.notna()]] = G[G.road_ndvi.notna()].to_numpy()
        F["seg"] = pts.seg.values
    F["filled"] = 1.0
    F.to_parquet(f)
    return rid, period, f"filled; empty {int(miss.sum())} -> {int(F.road_ndvi.isna().sum()) if 'road_ndvi' in F.columns else len(F)}"


# ---------------------------------------------------------------- OSM major-road check
MAJOR = "trunk|primary|secondary|tertiary|trunk_link|primary_link|secondary_link"


def on_major_road(segs: pd.DataFrame, metres=25) -> pd.Series:
    """True where the segment centre lies within `metres` of an OSM trunk/primary/secondary/tertiary way.
    Used for roads > 15 km, where a route between town names can stray onto minor roads."""
    bb = (segs.lat.min() - 0.01, segs.lon.min() - 0.01, segs.lat.max() + 0.01, segs.lon.max() + 0.01)

    def go():
        q = f'[out:json][timeout:120];way["highway"~"^({MAJOR})$"]({bb[0]},{bb[1]},{bb[2]},{bb[3]});out geom;'
        r = sat.http_json("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode())
        return [[(g["lon"], g["lat"]) for g in w.get("geometry", [])] for w in r.get("elements", [])]
    ways = sat.cached(f"osm major {bb}", go)
    lines = [np.array(w) for w in ways if len(w) >= 2]
    out = []
    for r in segs.itertuples():
        kx, ky = 111320 * math.cos(math.radians(r.lat)), 110570
        best = 1e9
        for L in lines:
            ax, ay = (L[:-1, 0] - r.lon) * kx, (L[:-1, 1] - r.lat) * ky
            bx, by = (L[1:, 0] - r.lon) * kx, (L[1:, 1] - r.lat) * ky
            dx, dy = bx - ax, by - ay
            t = np.clip(-(ax * dx + ay * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
            d = np.hypot(ax + t * dx, ay + t * dy).min()
            best = min(best, d)
            if best < metres:
                break
        out.append(best < metres)
    return pd.Series(out, index=segs.seg.values)


# ---------------------------------------------------------------- training (two seasons)
def train_features(window):
    tag, start, end = window
    f = sat.CACHE / f"train_features_{tag}.parquet"
    if f.exists():
        return sc.harmonise_cached(pd.read_parquet(f), period_offset(tag))
    legacy = sat.CACHE / "train_features_2026.parquet"  # same Dec-Feb 2026 window, from semantic_change.py (v1)
    if tag == "2026a" and legacy.exists():
        return sc.harmonise_cached(pd.read_parquet(legacy), 0.0)
    pts = sc.osm_points()
    scenes, seen = [], set()
    for (cx, cy), _ in pts.groupby([np.floor(pts.lon), np.floor(pts.lat)]):
        for s in scenes_window([float(cx), float(cy), float(cx) + 1, float(cy) + 1], start, end, max_cloud=30, per_tile=2):
            if s["blue"] not in seen:
                seen.add(s["blue"])
                scenes.append(s)
    X = sc.features_from_scenes(pts, sorted(scenes, key=lambda s: s["cloud"]))
    X.to_parquet(f)
    return X


TRAIN_WINDOWS = [("2026a", "2025-12-01", "2026-02-28")]
# Optional second season for training (adds ~4 GB of downloads country-wide):
TRAIN_WINDOW_MIDYEAR = ("2025b", "2025-06-01", "2025-08-31")
if "--train-midyear" in sys.argv:
    TRAIN_WINDOWS.append(TRAIN_WINDOW_MIDYEAR)


def train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, f1_score
    pts = sc.osm_points()
    parts = []
    for w in TRAIN_WINDOWS:
        X = train_features(w)
        d = pts.join(X).dropna(subset=["road_ndvi"])
        d = d[d.road_ndvi < 0.45].assign(window=w[0])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    cols = [c for c in d.columns if c.startswith(("road_", "d30_", "d60_"))]
    y = (d.cls == "asphalt").astype(int).to_numpy()
    preds = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(d, y, groups=d.way):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d.iloc[tr][cols], y[tr])
        preds[te] = m.predict(d.iloc[te][cols])
    info = dict(n=int(len(d)), n_ways=int(d.way.nunique()), cv_accuracy=float(accuracy_score(y, preds)),
                f1_asphalt=float(f1_score(y, preds)), baseline=float(max(y.mean(), 1 - y.mean())),
                by_window={w: float(accuracy_score(y[d.window == w], preds[d.window == w])) for w in d.window.unique()})
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d[cols], y)
    print("classifier:", info, flush=True)
    return model, cols, info


# ---------------------------------------------------------------- progress
def segment_states(F: pd.DataFrame, model, cols) -> pd.Series:
    st = pd.Series("nodata", index=F.index, dtype=object)
    if "road_ndvi" not in F.columns:
        return st
    ok = F.road_ndvi.notna()
    veg = ok & (F.road_ndvi > 0.45)
    st[veg] = "vegetated"
    rest = ok & ~veg
    if rest.any():
        st[rest] = np.where(model.predict(F.loc[rest, cols]) == 1, "asphalt", "unpaved")
    return st


SEAL_SHARE = 0.7


def sealed_period(states: list[str], order: list[str], share: float = None):
    """Change point: the first period from which >= SEAL_SHARE of the remaining valid observations
    (asphalt/unpaved) are asphalt, with at least two of them, the first being asphalt. Robust to the
    classifier's ~14% per-reading error; assumes paving is not undone within the window."""
    obs = [(p, s == "asphalt") for p, s in zip(order, states) if s in ("asphalt", "unpaved")]
    for k, (p, a) in enumerate(obs):
        rest = [x for _, x in obs[k:]]
        if a and len(rest) >= 2 and np.mean(rest) >= (SEAL_SHARE if share is None else share):
            return p
    return None


def activity_onset(bright: dict, segs_used, t_tender):
    """Construction activity: the corridor gets wider and brighter (cleared, graded, gravel base) before
    it is sealed. Per period, take the median over segments of road-minus-30 m brightness; baseline =
    readings before the tender (else the first two). Onset = first reading after the tender that is
    > 2 SD (min 0.01) above baseline in two consecutive readings."""
    if not bright or np.isnan(t_tender):
        return None
    ser = pd.Series({PT[p]: float(np.nanmedian(v.reindex(segs_used))) for p, v in bright.items()
                     if v.reindex(segs_used).notna().sum() >= max(3, len(segs_used) // 3)}).sort_index()
    base = ser[ser.index < t_tender]
    if len(base) < 2:
        return None
    thr = base.mean() + max(2 * base.std(), 0.01)
    after = ser[ser.index >= t_tender]
    hits = (after > thr).to_numpy()
    for k in range(len(hits) - 1):
        if hits[k] and hits[k + 1]:
            return float(after.index[k])
    return None


def planned_months(km):
    return 18 if km <= 15 else 24 if km <= 60 else 36


def main(workers=10, only_features=False):
    procs = pd.read_parquet(P / "works_tenders_scored.parquet")
    ocds_meta = tender_meta()
    geo, tasks = {}, []
    for road in ROADS:
        g = geometry(road)
        if not g:
            print("geocode failed", road[0])
            continue
        g["segs"] = segments(g["coords"], road[3])
        geo[road[0]] = g
        print(f"{road[0]:14s} routed {g['routed_km']:6.1f} km / contract {road[3]:6.1f}  ratio {g['ratio']:.2f}  segs {len(g['segs'])}", flush=True)
    valid = {k: v for k, v in geo.items() if (0.35 if k in PARTIAL else 0.6) <= v["ratio"] <= 1.6}
    print(f"{len(valid)}/{len(geo)} routes pass the length check", flush=True)
    for rid, g in valid.items():
        pj = g["segs"].to_json()
        for p in PERIODS:
            tasks.append((rid, p[0], pj))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(features_task, *t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            rid, per, msg = fu.result()
            if msg != "cached":
                print(f"[{i + 1}/{len(futs)}] {rid} {per} {msg}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fill_task, *t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            rid, per, msg = fu.result()
            if msg not in ("done",):
                print(f"[fill {i + 1}/{len(futs)}] {rid} {per} {msg}", flush=True)
    if only_features:
        return
    summarise(geo, valid, procs, ocds_meta)


def tender_meta():
    """tender start, value, buyer from the full OCDS file, by ocid."""
    import gzip
    want = {r[4] for r in ROADS if r[4]}
    out = {}
    with gzip.open(ROOT / "data" / "raw" / "uganda_ppda_ocds_full.jsonl.gz", "rt") as fh:
        for line in fh:
            if not any(w in line[:400] for w in want) and '"ocid"' in line[:400]:
                continue
            r = json.loads(line)
            r = r.get("compiledRelease", r)
            o = r.get("ocid")
            if o not in want:
                continue
            aw = r.get("awards", [])
            out[o] = dict(tender_start=((r.get("tender") or {}).get("tenderPeriod") or {}).get("startDate", "")[:10] or None,
                          release_date=(r.get("date") or "")[:10] or None,
                          award_value_bn=sum(((a.get("value") or {}).get("amount") or 0) for a in aw) / 1e9,
                          buyer=(r.get("buyer") or {}).get("name"))
    return out


_MODEL = {}


def summarise(geo, valid, procs, meta, seal_share=None, done_th=0.8, bench_shift=0.0, write=True):
    """seal_share: change-point share (default SEAL_SHARE); done_th: completion threshold on S;
    bench_shift: years added to the benchmark expected completion (sensitivity)."""
    if "m" not in _MODEL:
        _MODEL["m"] = train()
    model, cols, info = _MODEL["m"]
    seal_share = SEAL_SHARE if seal_share is None else seal_share
    order = [p[0] for p in PERIODS]
    manual = ROOT / "data" / "manual" / "road_contract_dates.csv"
    mdates = pd.read_csv(manual).set_index("id") if manual.exists() else pd.DataFrame()
    media = pd.DataFrame(columns=["id", "date", "type", "value"])
    for mf in (ROOT / "data" / "manual" / "media_reports.csv", ROOT / "data" / "manual" / "media_reports_v2.csv"):
        try:  # the first copy can be an unreadable cloud placeholder (OneDrive); the v2 copy is identical
            media = pd.read_csv(mf)
            break
        except (OSError, TimeoutError):
            continue
    _d = pd.to_datetime(media.date, errors="coerce")
    media["t"] = _d.dt.year + (_d.dt.dayofyear - 0.5) / 365.25
    seg_rows, ser_rows, road_rows = [], [], []
    for road in ROADS:
        rid, label, qs, km, ocid, kind = road
        if rid not in valid:
            g = geo.get(rid)
            road_rows.append(dict(id=rid, road=label, kind=kind, contract_km=km, ocid=ocid,
                                  routed_km=round(g["routed_km"], 1) if g else None, route_ratio=round(g["ratio"], 2) if g else None,
                                  status="excluded: route length mismatch" if g else "excluded: geocoding failed"))
            continue
        g = valid[rid]
        segs = g["segs"]
        states, bright = {}, {}
        for p in order:
            f = PC / f"{rid}_{p}.parquet"
            if not f.exists():
                continue
            F = pd.read_parquet(f).set_index("seg").drop(columns=["filled"], errors="ignore")
            if "fv" not in F.columns:
                F = sc.harmonise_cached(F, period_offset(p))
            states[p] = segment_states(F, model, cols)
            if "d30_bright" in F.columns:
                bright[p] = F.d30_bright
        if not states:
            continue
        S = pd.DataFrame(states).reindex(columns=[p for p in order if p in states])
        nvalid = S.isin(["asphalt", "unpaved"]).sum(axis=1)
        major = on_major_road(segs) if km > 15 else pd.Series(True, index=segs.seg.values)
        use = (nvalid >= 4) & major.reindex(S.index).fillna(False)
        sealed = {s: sealed_period(S.loc[s].tolist(), list(S.columns), seal_share) for s in S.index}
        for s in S.index:
            seg_rows.append(dict(id=rid, seg=int(s), km_at=float(segs.set_index("seg").km_at.get(s, np.nan)),
                                 sealed=sealed[s], used=bool(use[s]), on_major=bool(major.get(s, False)),
                                 **{p: S.loc[s, p] for p in S.columns}))
        segs_used = [s for s in S.index if use[s]]
        series = []
        for p in S.columns:
            share = np.mean([sealed[s] is not None and order.index(sealed[s]) <= order.index(p) for s in segs_used]) if segs_used else np.nan
            raw = S.loc[segs_used, p]
            raw_share = (raw == "asphalt").sum() / max(raw.isin(["asphalt", "unpaved"]).sum(), 1)
            nobs = int(raw.isin(["asphalt", "unpaved"]).sum())
            series.append((p, share))
            ser_rows.append(dict(id=rid, road=label, kind=kind, period=p, t=PT[p], sealed_share=share,
                                 raw_asphalt_share=raw_share if nobs else np.nan, n_obs=nobs, n_segs=len(segs_used)))
        m = meta.get(ocid, {}) if ocid else {}
        tender = pd.to_datetime(m.get("tender_start") or m.get("release_date"), errors="coerce")
        tdate_src = "tender invitation" if m.get("tender_start") else "portal entry (upper bound)" if m.get("release_date") else None
        t_tender = tender.year + (tender.dayofyear - 0.5) / 365.25 if pd.notna(tender) else np.nan
        sh = pd.Series(dict(series)).dropna()
        raw = pd.Series({PT[r["period"]]: r["raw_asphalt_share"] for r in ser_rows if r["id"] == rid}).dropna().sort_index()
        s0 = float(sh.iloc[0]) if len(sh) else np.nan
        t_last = PT[sh.index[-1]] if len(sh) else np.nan
        s_last = float(sh.iloc[-1]) if len(sh) else np.nan
        # state when the contract was tendered (first image if tendered before 2017)
        before = [PT[p] <= t_tender for p in sh.index] if not np.isnan(t_tender) else [False] * len(sh)
        s_tender = float(sh[np.array(before)].iloc[-1]) if any(before) else s0

        def first(th, rel=False):
            """First reading at/after the tender where S (or S - S_tender) reaches th."""
            for p, v in sh.items():
                if not np.isnan(t_tender) and PT[p] < t_tender - 0.25:
                    continue
                if (v - s_tender if rel else v) >= th:
                    return PT[p]
            return None
        t_onset, t_half, t_done = first(0.10, rel=True), first(0.5), first(done_th)
        t_activity = activity_onset(bright, segs_used, t_tender)
        # ---- media evidence (data/manual/media_reports.csv): launch, duration, deadline, completion
        md_ = media[media.id == rid] if len(media) else media

        def mdate(typ, how="min"):
            d = md_[(md_.type == typ) & md_.t.notna()].t
            return (d.min() if how == "min" else d.max()) if len(d) else None
        t_launch = mdate("launch")
        dur_m = md_[md_.type == "duration_months"].value.dropna()
        deadline = mdate("deadline", "max")
        t_media_done = min([x for x in (mdate("complete"), mdate("commissioned")) if x is not None], default=None)
        prog = md_[(md_.type == "progress_pct") & md_.t.notna()].sort_values("t")
        # expected completion: contract record > media deadline > media launch + duration > tender benchmark
        exp_done, exp_src = (t_tender + 0.5 + planned_months(km) / 12 + bench_shift, "benchmark: tender + 6 months + planned duration") \
            if not np.isnan(t_tender) else (np.nan, None)
        if t_launch is not None:
            exp_done, exp_src = t_launch + (float(dur_m.iloc[0]) if len(dur_m) else planned_months(km)) / 12, \
                "media: launch date + " + ("reported duration" if len(dur_m) else "planned duration")
        if deadline is not None:
            exp_done, exp_src = deadline, "media: reported contractual deadline"
        if rid in mdates.index:
            mdr = mdates.loc[rid]
            if pd.notna(mdr.get("planned_completion")):
                d = pd.to_datetime(mdr.planned_completion)
                exp_done, exp_src = d.year + (d.dayofyear - 0.5) / 365.25, "contract record"
        gain = s_last - s_tender
        past_due = not np.isnan(exp_done) and t_last > exp_done
        coverage = len(segs_used) / max(len(segs), 1)
        # plateau: S has not moved by > 0.03 over the last 4 readings (2 years) and rose >= 0.3 after tender
        t_plateau = None
        if len(sh) >= 4 and gain >= 0.30 and (sh.iloc[-4:].max() - sh.iloc[-4:].min()) <= 0.03:
            t_plateau = first(s_tender + 0.9 * gain)
        # tendered before the archive and essentially flat from the second reading: sealed by mid-2017
        early_sealed = ((np.isnan(t_tender) or t_tender < 2017.1) and len(sh) >= 4 and s_last >= 0.5
                        and (sh.max() - sh.iloc[1]) < 0.15)
        unsealed_km = [round(float(segs.set_index("seg").km_at.get(sg, np.nan)), 1) for sg in segs_used if sealed[sg] is None]
        # controls (already paved): raw asphalt share; reconstruction strips the surface for a time
        base_raw = float(raw.iloc[:4].mean()) if len(raw) else np.nan
        after = raw[raw.index >= (t_tender if not np.isnan(t_tender) else raw.index[0])]
        roll = after.rolling(2).max().dropna()  # two consecutive low readings, not one noisy one
        min_raw = float(roll.min()) if len(roll) else np.nan
        t_min_raw = float(roll.idxmin()) if len(roll) else np.nan
        last2 = float(raw.iloc[-2:].mean()) if len(raw) else np.nan
        if coverage < 0.5:
            status = (f"insufficient visible road ({len(segs_used)} of {len(segs)} segments readable: narrow or "
                      "tree-covered road): no satellite verdict")
        elif kind.startswith("control"):
            if base_raw < 0.6:
                status = "control: low asphalt share at start (check)"
            elif base_raw - min_raw >= 0.15:
                status = ("rehabilitation: surface stripped and restored" if last2 >= base_raw - 0.10 else
                          f"rehabilitation under way: surface stripped (asphalt share {base_raw:.0%} -> {min_raw:.0%}), not yet restored")
            else:
                status = "control: paved; no surface works visible"
        elif early_sealed:
            status = "sealed by the first images (mid-2017 or earlier)"
        elif s_tender >= 0.8:
            status = "sealed before tender: route may follow an existing paved road, or a later phase"
        elif t_done:
            status = "completed (satellite)"
        elif t_plateau:
            status = (f"substantially complete (works plateaued at {s_last:.0%} sealed); "
                      f"{len(unsealed_km)} segments never read as sealed: field check")
        elif past_due and gain < 0.10:
            status = "no paving detected past expected completion"
        elif past_due:
            status = f"incomplete past expected completion ({s_last:.0%} sealed)"
        elif gain < 0.10:
            status = "no paving detected yet (not yet due)"
        else:
            status = f"in progress / not yet due ({s_last:.0%} sealed)"
        t_complete = (2017.58 if early_sealed else (t_done or t_plateau)) if not status.startswith(("insufficient", "rehab", "control")) else None
        # works start: media launch, else the corridor-brightening signal; the first rise in sealed share is not
        # used because sealing comes late in construction
        t_start = t_launch if t_launch is not None else t_activity
        # satellite sealed share at each media progress report (nearest earlier reading)
        sat_at = []
        for r_ in prog.itertuples():
            ks = [k for k in sh.index if PT[k] <= r_.t + 0.05]
            sat_at.append(dict(date=str(r_.date)[:10], media_pct=float(r_.value), sat_sealed=float(sh[ks[-1]]) if ks else None))
        tw = procs[procs.ocid == ocid]
        pr = tw.iloc[0] if len(tw) else None
        road_rows.append(dict(
            id=rid, road=label, kind=kind, ocid=ocid, contract_km=km, routed_km=round(g["routed_km"], 1),
            route_ratio=round(g["ratio"], 2), n_segs=len(segs_used), n_segs_route=len(segs),
            n_segs_off_major=int((~major).sum()), buyer=m.get("buyer"),
            award_value_bn=m.get("award_value_bn"), tender_date=str(tender.date()) if pd.notna(tender) else None,
            tender_date_source=tdate_src, portal_entry=m.get("release_date"),
            t_tender=t_tender, s0=s0, s_tender=s_tender, s_last=s_last, t_last=t_last,
            base_raw=base_raw, min_raw_after_tender=min_raw, t_min_raw=t_min_raw, gain=gain,
            t_activity=t_activity, t_onset=t_onset, t_half=t_half, t_done=t_done, t_plateau=t_plateau,
            t_complete=t_complete, unsealed_km=json.dumps(unsealed_km), coverage=coverage, early_sealed=bool(early_sealed),
            t_launch=t_launch, t_start=t_start, t_media_done=t_media_done,
            media_progress=json.dumps(sat_at), n_media=int(len(md_)),
            years_tender_to_start=(t_launch - t_tender) if t_launch is not None and not np.isnan(t_tender) else np.nan,
            years_start_to_done=(t_complete - t_start) if t_complete and t_start and not early_sealed else np.nan,
            years_tender_to_done=(t_complete - t_tender) if t_complete and not early_sealed and not np.isnan(t_tender) and s_tender < 0.8 else np.nan,
            expected_done=exp_done, expected_source=exp_src,
            delay_years=np.nan if (kind.startswith("control") or s_tender >= 0.8 or early_sealed or coverage < 0.5) else
            ((t_complete - exp_done) if t_complete and not np.isnan(exp_done) else
             ((t_last - exp_done) if past_due else np.nan)),
            delay_censored=bool(not t_complete and s_tender < 0.8 and not kind.startswith("control") and coverage >= 0.5),
            status=status,
            risk_score=float(pr.risk_score) if pr is not None else np.nan,
            n_flags=float(pr.n_flags) if pr is not None else np.nan,
            award_to_est=float(pr.award_to_est) if pr is not None and pd.notna(pr.award_to_est) else np.nan,
            method=pr.method if pr is not None else None,
        ))
    R = pd.DataFrame(road_rows)
    if not write:
        return R
    Sr = pd.DataFrame(ser_rows)
    Sg = pd.DataFrame(seg_rows)
    R.to_parquet(P / "road_progress_roads.parquet", index=False)
    Sr.to_parquet(P / "road_progress_series.parquet", index=False)
    Sg.astype({c: str for c in Sg.columns if c in order or c == "sealed"}).to_parquet(P / "road_progress_segments.parquet", index=False)
    (ROOT / "paper" / "road_progress.json").write_text(json.dumps(dict(
        classifier=info, roads=R.to_dict("records")), indent=2, default=lambda x: None if pd.isna(x) else float(x)))
    pd.set_option("display.width", 250)
    print(R[["id", "route_ratio", "coverage", "tender_date", "s_last", "t_start", "t_complete", "t_media_done", "expected_done", "delay_years", "status"]].round(2).to_string(index=False, max_colwidth=70))


def sensitivity():
    """Re-derive statuses under alternative thresholds and benchmarks; writes paper/road_progress_sensitivity.json."""
    procs = pd.read_parquet(P / "works_tenders_scored.parquet")
    meta = tender_meta()
    geo = {}
    for road in ROADS:
        g = geometry(road)
        if g:
            g["segs"] = segments(g["coords"], road[3])
            geo[road[0]] = g
    valid = {k: v for k, v in geo.items() if (0.35 if k in PARTIAL else 0.6) <= v["ratio"] <= 1.6}
    rows = []
    for ss in (0.6, 0.7, 0.8):
        for dt in (0.7, 0.8, 0.9):
            for bs in (-1.0, 0.0, 1.0):
                R = summarise(geo, valid, procs, meta, seal_share=ss, done_th=dt, bench_shift=bs, write=False)
                R = R[~R.status.str.startswith("excluded")]
                up = R[~R.kind.str.startswith("control")]
                rows.append(dict(seal_share=ss, done_th=dt, bench_shift=bs,
                                 completed=int(up.status.str.startswith(("completed", "substantially")).sum()),
                                 no_paving_past_due=int(up.status.str.startswith("no paving detected past").sum()),
                                 overdue_in_progress=int(up.status.str.startswith("incomplete").sum()),
                                 median_tender_to_done=float(up.years_tender_to_done.median()),
                                 statuses={r.id: r.status.split(" (")[0].split(";")[0] for r in up.itertuples()}))
                print(rows[-1]["seal_share"], rows[-1]["done_th"], rows[-1]["bench_shift"], rows[-1]["completed"],
                      rows[-1]["no_paving_past_due"], rows[-1]["overdue_in_progress"], flush=True)
    (ROOT / "paper" / "road_progress_sensitivity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__" and "--sensitivity" in sys.argv:
    sensitivity()
    sys.exit()

if __name__ == "__main__":
    if "--geometry" in sys.argv:
        for road in ROADS:
            g = geometry(road)
            print(f"{road[0]:14s}", "FAILED" if not g else f"routed {g['routed_km']:6.1f} / {road[3]:6.1f} ratio {g['ratio']:.2f} wps {g['waypoints']}", flush=True)
    else:
        main(only_features="--features" in sys.argv)
