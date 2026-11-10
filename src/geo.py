"""District-level data for the public interactive map (aggregates only, no firm names).

Location of each works record
  * council  - the buyer is a district / municipal / city council -> that district
  * title    - central-government buyer (ministry, UNRA, NWSC...) and the title names one or
               more districts or towns (e.g. "Upgrading of Moroto–Lokitanyala road") -> those
               districts; value is split equally when several are named
  * unlocated - neither: national/HQ works, not mapped

Metrics are computed for every (district, financial year, sub-sector) cell, plus "all".

Sources: district boundaries 2020 and population 2022 (UBOS, via geoBoundaries and HDX COD-PS).
Out: docs/data/districts.geojson   geometry + name/region/population
     docs/data/district_metrics.json
     docs/data/summary.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entities import canonical  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
GEO = ROOT / "data" / "raw" / "geo" / "uga_districts_ubos2020.geojson"
POP = ROOT / "data" / "raw" / "geo" / "uga_admpop_adm2_2023.csv"
OUT = ROOT / "docs" / "data"
MIN_RELIABLE = 10

# towns/cities -> host district; spelling variants
ALIAS = {
    "kampala capital": "Kampala", "kcca": "Kampala", "fort portal": "Kabarole", "entebbe": "Wakiso",
    "lugazi": "Buikwe", "njeru": "Buikwe", "ntugamo": "Ntungamo", "nansana": "Wakiso", "kira": "Wakiso",
    "makindye ssabagabo": "Wakiso", "kiira": "Wakiso", "terego": "Arua", "sembabule": "Ssembabule",
    "luweero": "Luwero", "bushenyi-ishaka": "Bushenyi", "ishaka": "Bushenyi", "mukono": "Mukono",
    "karuma": "Kiryandongo", "namanve": "Mukono", "bugolobi": "Kampala", "nakawa": "Kampala",
    "ggaba": "Kampala", "gaba": "Kampala", "katosi": "Mukono", "kalangala": "Kalangala",
    "nebbi": "Nebbi", "pakwach": "Pakwach", "lokitanyala": "Moroto", "bududa": "Bududa",
    "mutukula": "Kyotera", "malaba": "Tororo", "elegu": "Amuru", "busunju": "Mityana",
    "kyenjojo": "Kyenjojo", "kagadi": "Kagadi", "kakumiro": "Kakumiro", "rwampara": "Rwampara",
}
LG = re.compile(r"^(.*?)\s+(district|municipal|municipality|city|town)\b", re.I)
WORD_EXCLUDE = {"central", "kole road"}  # placeholders for future exclusions


def fy(d: pd.Timestamp) -> str | None:
    if pd.isna(d) or d.year < 2014:
        return None
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y}/{str(y + 1)[2:]}"


def build_locator(names: list[str]):
    keys = {n.lower(): n for n in names}
    keys.update({k: v for k, v in ALIAS.items()})
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, keys), key=len, reverse=True)) + r")\b", re.I)

    def council(buyer):
        if not isinstance(buyer, str):
            return None
        m = LG.match(buyer.strip())
        if not m:
            return None
        k = m.group(1).strip().lower()
        return keys.get(k)

    def from_title(title):
        if not isinstance(title, str):
            return []
        return sorted({keys[m.group(1).lower()] for m in pat.finditer(title)})

    return council, from_title


def wshare(flag: pd.Series, w: pd.Series) -> float | None:
    ok = flag.notna()
    if not ok.any() or w[ok].sum() == 0:
        return None
    return float((flag[ok].astype(float) * w[ok]).sum() / w[ok].sum())


def cell_metrics(d: pd.DataFrame, pop: float | None) -> dict:
    w = d.weight
    val = float((d.award_value_ugx.fillna(0) * w).sum())
    sup = (d.award_value_ugx.fillna(0) * w).groupby(d.supplier_canon).sum()
    top_share = float(sup.max() / sup.sum()) if sup.sum() > 0 else None
    pc = d.post_close_days.dropna()
    out = dict(
        n=round(float(w.sum()), 1),
        records=int(len(d)),
        value=round(val),
        value_per_capita=round(val / pop) if pop else None,
        mean_risk=round(float(np.average(d.risk_score, weights=w)), 1),
        high_flags=int((d.n_flags >= 3).sum()),
        non_open_share=wshare(d.non_open, w),
        single_bid_share=wshare(d.single_bid, w),
        just_below_share=wshare(d.just_below, w),
        top_supplier_share=top_share,
        n_suppliers=int(d.supplier_canon.nunique()),
        median_post_close_days=float(pc.median()) if len(pc) >= 3 else None,
        council_value_share=float((d.award_value_ugx.fillna(0) * w)[d.loc_source == "council"].sum() / val) if val else None,
    )
    return {k: (round(v, 3) if isinstance(v, float) and k.endswith("share") else v) for k, v in out.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = json.loads(GEO.read_text())
    pop = pd.read_csv(POP).set_index("ADM2_EN")
    names = [f["properties"]["shapeName"] for f in g["features"]]
    council, from_title = build_locator(names)

    w = pd.read_parquet(P / "works_tenders_scored.parquet").assign(src="tender")
    w["post_close_days"] = (w.award_date - w.tender_end).dt.days.where(lambda s: s.between(0, 1500))
    w["non_open"] = w.method.map(lambda m: None if pd.isna(m) else m != "open")
    w["just_below"] = w.award_to_est.map(lambda r: None if pd.isna(r) else 0.99 <= r < 0.999999)
    w["single_bid"] = None
    bw = pd.read_parquet(P / "works_bids_scored.parquet").assign(src="bids")
    bw["post_close_days"] = np.nan
    bw["non_open"] = None
    bw["just_below"] = None
    bw["single_bid"] = bw.f_single_bidder
    cols = ["ocid", "buyer_name", "title", "subsector", "supplier_name", "award_value_ugx", "award_date",
            "risk_score", "n_flags", "post_close_days", "non_open", "just_below", "single_bid", "src"]
    a = pd.concat([w[cols], bw[cols]], ignore_index=True)
    a["supplier_canon"] = a.supplier_name.map(canonical).fillna("unknown")
    a["fy"] = a.award_date.map(fy)
    a["subsector"] = a.subsector.fillna("other_works")

    rows = []
    for r in a.itertuples(index=False):
        c = council(r.buyer_name)
        if c:
            rows.append((r, c, "council", 1.0))
            continue
        ds = from_title(r.title)
        if ds:
            for d in ds:
                rows.append((r, d, "title", 1.0 / len(ds)))
        else:
            rows.append((r, None, "unlocated", 1.0))
    loc = pd.DataFrame([dict(r._asdict(), district=d, loc_source=s, weight=wt) for r, d, s, wt in rows])

    mapped = loc[loc.district.notna()]
    years = sorted(mapped.fy.dropna().unique())
    subs = ["all", "buildings", "roads", "water", "other_works"]
    data: dict = {}
    national: dict = {}
    for (yr, sub) in [(y, s) for y in ["all"] + years for s in subs]:
        m = mapped if yr == "all" else mapped[mapped.fy == yr]
        m = m if sub == "all" else m[m.subsector == sub]
        cells = {}
        for dist, d in m.groupby("district"):
            p = float(pop.loc[dist, "T_TL"]) if dist in pop.index else None
            cells[dist] = cell_metrics(d, p)
        key = f"{yr}|{sub}"
        for dist, v in cells.items():
            data.setdefault(dist, {})[key] = v
        # national medians across reliable districts, for comparison
        rel = [v for v in cells.values() if v["records"] >= MIN_RELIABLE]
        national[key] = {k: (float(np.median([v[k] for v in rel if v[k] is not None]))
                             if any(v[k] is not None for v in rel) else None)
                         for k in ["value_per_capita", "mean_risk", "non_open_share", "single_bid_share",
                                   "just_below_share", "top_supplier_share", "median_post_close_days"]}

    # public procuring bodies per district (named: they are public institutions)
    bodies = (mapped.groupby(["district", "buyer_name", "loc_source"]).weight.sum().reset_index()
              .sort_values("weight", ascending=False))
    body_list = {d: [dict(name=b.buyer_name, kind=b.loc_source, records=round(float(b.weight), 1))
                     for b in grp.head(6).itertuples()] for d, grp in bodies.groupby("district")}

    for f in g["features"]:
        n = f["properties"]["shapeName"]
        f["properties"] = dict(district=n,
                               region=str(pop.loc[n, "ADM1_EN"]) if n in pop.index else None,
                               population=int(pop.loc[n, "T_TL"]) if n in pop.index else None)

        def rnd(c):
            return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 3), round(c[1], 3)]
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
    (OUT / "districts.geojson").write_text(json.dumps(g, separators=(",", ":")))
    (OUT / "district_metrics.json").write_text(json.dumps(
        dict(years=years, subsectors=subs, min_reliable=MIN_RELIABLE, national=national, data=data, bodies=body_list),
        separators=(",", ":"), default=float))

    loc["wvalue"] = loc.award_value_ugx.fillna(0) * loc.weight
    src_counts = loc.groupby("loc_source").agg(records=("ocid", "nunique"), value=("wvalue", "sum"))
    summary = dict(
        records_total=int(a.ocid.nunique()),
        by_location_source=src_counts.reset_index().to_dict("records"),
        districts_with_data=int(mapped.district.nunique()),
        districts_total=len(names),
        years=years,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
