"""District-level aggregates for the public interactive map.

Boundaries: Uganda districts (135, 2020), Uganda Bureau of Statistics via geoBoundaries
(gbHumanitarian UGA ADM2, CC BY 3.0 IGO).

Only *local-government* buyers (district/municipal/city councils) can be placed on the map;
central-government works (ministries, UNRA, NWSC, ...) are national and shown separately.
Aggregates only -- no firm names.

Out: docs/data/districts.geojson, docs/data/summary.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
GEO = ROOT / "data" / "raw" / "geo" / "uga_districts_ubos2020.geojson"
OUT = ROOT / "docs" / "data"

# towns/cities whose council sits inside a district of a different name, and spelling fixes
ALIAS = {
    "kampala capital": "Kampala", "kampala": "Kampala", "fort portal": "Kabarole", "entebbe": "Wakiso",
    "lugazi": "Buikwe", "njeru": "Buikwe", "ntugamo": "Ntungamo", "nansana": "Wakiso", "kira": "Wakiso",
    "makindye ssabagabo": "Wakiso", "mubende": "Mubende", "kasese": "Kasese", "sembabule": "Ssembabule",
    "ssembabule": "Ssembabule", "kapchorwa": "Kapchorwa", "luwero": "Luwero", "luweero": "Luwero",
    "nakaseke": "Nakaseke", "mityana": "Mityana", "bushenyi-ishaka": "Bushenyi", "ishaka": "Bushenyi",
    "kamwenge": "Kamwenge", "kiira": "Wakiso", "terego": "Arua", "kotido": "Kotido", "moroto": "Moroto", "koboko": "Koboko",
}
LG = re.compile(r"^(.*?)\s+(district|municipal|municipality|city|town)\b", re.I)


def district_of(buyer: str, names: dict[str, str]) -> str | None:
    if not isinstance(buyer, str):
        return None
    m = LG.match(buyer.strip())
    if not m:
        return None
    key = m.group(1).strip().lower()
    if key in ALIAS:
        return ALIAS[key]
    return names.get(key)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = json.loads(GEO.read_text())
    names = {f["properties"]["shapeName"].lower(): f["properties"]["shapeName"] for f in g["features"]}

    w = pd.read_parquet(P / "works_tenders_scored.parquet").assign(src="tender")
    bw = pd.read_parquet(P / "works_bids_scored.parquet").assign(src="bids")
    cols = ["buyer_name", "award_value_ugx", "risk_score", "n_flags", "src"]
    a = pd.concat([w[cols + ["f_direct_procurement", "f_restricted_method", "f_award_just_below_est"]],
                   bw[cols + ["f_single_bidder"]]], ignore_index=True)
    a["district"] = a.buyer_name.map(lambda b: district_of(b, names))
    lg = a[a.district.notna()]

    def frac(s):
        s = s.dropna()
        return float(s.astype(bool).mean()) if len(s) else None

    agg = lg.groupby("district").agg(
        works_records=("risk_score", "size"),
        works_value_ugx=("award_value_ugx", "sum"),
        mean_risk=("risk_score", "mean"),
        high_flag_records=("n_flags", lambda s: int((s >= 3).sum())),
        entities=("buyer_name", "nunique"),
    )
    agg["restricted_or_direct_share"] = lg.groupby("district").apply(
        lambda d: frac(d.f_direct_procurement.fillna(False) | d.f_restricted_method.fillna(False))
        if d.f_direct_procurement.notna().any() else None)
    agg["just_below_estimate_share"] = lg.groupby("district").f_award_just_below_est.apply(frac)

    for f in g["features"]:
        n = f["properties"]["shapeName"]
        props = {"district": n}
        if n in agg.index:
            r = agg.loc[n]
            props.update(
                works_records=int(r.works_records),
                works_value_ugx=float(r.works_value_ugx),
                mean_risk=round(float(r.mean_risk), 1),
                high_flag_records=int(r.high_flag_records),
                entities=int(r.entities),
                restricted_or_direct_share=None if pd.isna(r.restricted_or_direct_share) else round(float(r.restricted_or_direct_share), 3),
                just_below_estimate_share=None if pd.isna(r.just_below_estimate_share) else round(float(r.just_below_estimate_share), 3),
            )
        f["properties"] = props
        # trim coordinate precision (~100 m) to keep the file small
        def rnd(c):
            return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 3), round(c[1], 3)]
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
    (OUT / "districts.geojson").write_text(json.dumps(g, separators=(",", ":")))

    central = a[a.district.isna()]
    summary = dict(
        mapped_records=int(len(lg)), mapped_value_ugx=float(lg.award_value_ugx.sum()),
        districts_with_data=int(agg.shape[0]), districts_total=len(g["features"]),
        central_records=int(len(central)), central_value_ugx=float(central.award_value_ugx.sum()),
        unmatched_lg_buyers=sorted(set(a.buyer_name[a.buyer_name.str.contains(r"\b(?:District|Municipal|City)\b", case=False, na=False)
                                                    & a.district.isna()]))[:50],
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items()}, indent=1)[:1500])


if __name__ == "__main__":
    main()
