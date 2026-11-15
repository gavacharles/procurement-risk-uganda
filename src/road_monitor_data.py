"""Follow-on step 6: data for the public Road Monitor (docs/data/roads.json).

One record per contracted upgrade road with a satellite verdict: route and 1 km segments with the year each
was first read as sealed, the half-yearly sealed share, reported physical progress (government monitoring
reports), monthly radar activity (if road_activity.py has run), milestones and indicators, and formal
disputes described by type and public record only.

Public-output rules: no contractor, consultant or bidder is named; statuses are framed as prompts for
inspection, not findings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
import road_progress as rp  # noqa: E402

DISPUTE_PUBLIC = {  # type -> wording for the public site (no party names)
    "termination of main civil-works contract; guarantees called": "Main works contract terminated; guarantees called",
    "termination of main civil-works contract; arbitration": "Former contractor's challenge pending arbitration",
    "bid challenge (administrative review) of civil-works award": "Bid challenge against the works award (tribunal)",
    "bid challenge (appeal)": "Bid challenge on appeal (commercial court)",
    "bid challenge (remitted review)": "Bid challenge back before the tribunal",
    "subcontract payment dispute (haulage of materials)": "Subcontractor payment claim against the main contractor",
    "termination of supervision consultancy; arbitration claim": "Supervision consultant terminated and replaced; arbitration claim",
    "regulator investigation of the tender (cleared)": "Regulator investigated the tender and cleared it",
}


NOTES = {  # structure the segment indicators cannot see on their own (from the monitoring reports)
    "kigumba": "Built as two contracts (Kigumba–Bulima, Bulima–Kabwoya); the 'pause' is the gap between them.",
    "rwenkunye": "Built as two contracts (Rwenkunye–Apac, Apac–Lira–Puranga), both with works from about 2021.",
    "muyembe": "The kilometres sealed in 2017–18 at the Muyembe end probably belong to an adjacent rehabilitation contract; "
               "the main works contract was terminated and re-let, and reported progress was 4% in June 2021.",
}


def r2(x, k=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else round(float(x), k)


def main():
    R = pd.read_parquet(P / "road_progress_roads.parquet").set_index("id")
    S = pd.read_parquet(P / "road_progress_segments.parquet")
    SR = pd.read_parquet(P / "road_progress_series.parquet")
    I = pd.read_parquet(P / "road_indicators.parquet").set_index("id")
    rep = pd.read_csv(ROOT / "data" / "manual" / "reported_progress_checked.csv", parse_dates=["asof"])
    dis = pd.read_csv(ROOT / "data" / "manual" / "road_disputes_checked.csv")
    dis = dis[dis.relevant == "yes"]
    act = None  # radar activity failed validation (paper 5, §4.2): not shown publicly
    actj = None
    roads = []
    for road in rp.ROADS:
        rid, label, _, km, _, kind = road
        if kind != "upgrade" or rid not in R.index:
            continue
        r = R.loc[rid]
        if str(r.status).startswith(("excluded", "insufficient")):
            continue
        g = rp.geometry(road)
        coords = g["coords"]
        step = max(1, len(coords) // 300)
        route = [[round(y, 5), round(x, 5)] for x, y in coords[::step]]
        seg = rp.segments(coords, km)
        s = S[S.id == rid].set_index("seg")
        segs = []
        for q in seg.itertuples():
            if q.seg not in s.index:
                continue
            row = s.loc[q.seg]
            t = rp.PT.get(row.sealed) if isinstance(row.sealed, str) else None
            segs.append(dict(lat=round(q.lat, 5), lon=round(q.lon, 5), km=r2(q.km_at, 1),
                             sealed=r2(t, 2), readable=bool(row.used)))
        ser = SR[SR.id == rid].sort_values("t")
        rr = rep[rep.road_id == rid].sort_values("asof")
        ind = I.loc[rid] if rid in I.index else None
        rec = dict(
            id=rid, name=label, km=km, status=str(r.status), note=NOTES.get(rid), tender=r2(r.t_tender), launch=r2(r.t_launch),
            first_works=r2(r.t_onset), completed=r2(r.t_done), expected=r2(r.expected_done),
            route=route, segments=segs,
            sealed_series=[[r2(a), r2(b, 3)] for a, b in zip(ser.t, ser.sealed_share)],
            reported=[dict(t=r2(d.year + d.dayofyear / 365.25), pct=r2(p, 1), lot=l)
                      for d, p, l in zip(rr["asof"], rr.physical, rr.lot)],
            indicators=None if ind is None or pd.isna(ind.get("velocity_kmy", np.nan)) else dict(
                lag_tender_to_seal=r2(ind.lag_tender_to_seal, 1), velocity_kmy=r2(ind.velocity_kmy, 1),
                longest_pause_y=r2(ind.longest_stall_y, 1), share_never_sealed=r2(ind.share_never, 2)),
            disputes=[dict(year=str(d.date)[:4], what=DISPUTE_PUBLIC.get(d.type, d.type), stage=d.stage)
                      for d in dis[dis.road_id == rid].itertuples()],
        )
        if act is not None and rid in set(act.id):
            a = act[act.id == rid].sort_values("t")
            rec["activity"] = [[r2(t, 3), r2(v, 3)] for t, v in zip(a.t, a.A)]
            if actj and rid in actj["roads"]:
                sp = actj["roads"][rid]
                rec["radar"] = dict(longest_pause_months=sp["longest_spell_months"], pause=sp["longest_spell"])
        roads.append(rec)
    out = dict(updated=pd.Timestamp.today().strftime("%Y-%m-%d"),
               activity_threshold=actj["threshold"]["A_star"] if actj else None,
               roads=roads)
    f = ROOT / "docs" / "data" / "roads.json"
    f.write_text(json.dumps(out, separators=(",", ":")))
    print(len(roads), "roads", round(f.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
