"""Label-free validation of the latest satellite state against OpenStreetMap surface tags (paper 3).

For every analysed road segment, find the nearest OSM highway within 25 m of the segment centre and its
`surface` tag (asphalt/paved/concrete/chipseal -> sealed; unpaved/gravel/dirt/ground/compacted/earth/laterite
-> unpaved). Compare with the satellite state at the latest reading (sealed by Jul 2026 or not).
OSM ways used to train the classifier are excluded, so the check is out-of-sample. OSM tags can be out
of date (a road paved last year may still be tagged gravel), which biases agreement down on newly paved roads.

Out: paper/road_progress_osmcheck.json
"""
from __future__ import annotations

import json
import math
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import road_progress as rp  # noqa: E402
import satellite as sat  # noqa: E402
import semantic_change as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEALED = {"asphalt", "paved", "concrete", "chipseal"}
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "compacted", "earth", "fine_gravel", "laterite", "murram"}


def osm_ways(bb):
    def go():
        q = (f'[out:json][timeout:180];way["highway"]["surface"]({bb[0]},{bb[1]},{bb[2]},{bb[3]});out tags geom;')
        r = sat.http_json("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode())
        return [dict(id=w["id"], surface=w.get("tags", {}).get("surface"), highway=w.get("tags", {}).get("highway"),
                     geom=[(g["lon"], g["lat"]) for g in w.get("geometry", [])]) for w in r.get("elements", [])]
    return sat.cached(f"osm surface ways {bb}", go)


def nearest(lon, lat, ways, metres=25):
    kx, ky = 111320 * math.cos(math.radians(lat)), 110570
    best, bw = 1e9, None
    for w in ways:
        L = np.array(w["geom"])
        if len(L) < 2:
            continue
        ax, ay = (L[:-1, 0] - lon) * kx, (L[:-1, 1] - lat) * ky
        bx, by = (L[1:, 0] - lon) * kx, (L[1:, 1] - lat) * ky
        dx, dy = bx - ax, by - ay
        t = np.clip(-(ax * dx + ay * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
        d = np.hypot(ax + t * dx, ay + t * dy).min()
        if d < best:
            best, bw = d, w
    return bw if best < metres else None


def main():
    G = pd.read_parquet(rp.P / "road_progress_segments.parquet")
    R = pd.read_parquet(rp.P / "road_progress_roads.parquet")
    keep = R[~R.status.str.startswith(("excluded", "insufficient"))].id
    G = G[G.id.isin(keep) & G.used.astype(bool)]
    train_ways = set(sc.osm_points().way)
    rows = []
    for rid, g in G.groupby("id"):
        road = next(x for x in rp.ROADS if x[0] == rid)
        segs = rp.segments(rp.geometry(road)["coords"], road[3]).set_index("seg")
        pts = segs.loc[g.seg.astype(int)]
        bb = (pts.lat.min() - 0.005, pts.lon.min() - 0.005, pts.lat.max() + 0.005, pts.lon.max() + 0.005)
        try:
            ways = osm_ways(bb)
        except Exception as e:
            print("overpass fail", rid, e)
            continue
        for r in g.itertuples():
            s = segs.loc[int(r.seg)]
            w = nearest(s.lon, s.lat, ways)
            if not w or w["id"] in train_ways:
                continue
            osm = "sealed" if w["surface"] in SEALED else "unpaved" if w["surface"] in UNPAVED else None
            if osm is None:
                continue
            sat_state = "sealed" if r.sealed not in ("None", None, "nan") else "unpaved"
            rows.append(dict(id=rid, seg=int(r.seg), osm=osm, sat=sat_state, surface=w["surface"],
                             kind=R.set_index("id").kind[rid]))
        print(rid, sum(1 for x in rows if x["id"] == rid), flush=True)
    D = pd.DataFrame(rows)
    ct = pd.crosstab(D.osm, D.sat)
    acc = float((D.osm == D.sat).mean())
    by_road = D.groupby("id").apply(lambda x: float((x.osm == x.sat).mean())).round(2).to_dict()
    out = dict(n_segments=int(len(D)), n_roads=int(D.id.nunique()), agreement=acc,
               confusion={f"osm_{a}_sat_{b}": int(ct.loc[a, b]) if a in ct.index and b in ct.columns else 0
                          for a in ("sealed", "unpaved") for b in ("sealed", "unpaved")},
               agreement_by_road=by_road,
               note="OSM ways used in classifier training excluded; OSM tags may lag recent paving")
    (ROOT / "paper" / "road_progress_osmcheck.json").write_text(json.dumps(out, indent=2))
    print(ct)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
