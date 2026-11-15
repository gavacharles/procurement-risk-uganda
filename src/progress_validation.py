"""Follow-on step 2b: satellite progress against reported physical progress (government monitoring reports).

Panel: data/manual/reported_progress_checked.csv, hand-checked from the regex extraction in
progress_reports.py (only rows that clearly refer to the upgrade contract; lots kept separate).
Two roads in the satellite sample are two contracts each; their segments are split by chainage in proportion
to lot length (Kigumba–Bulima 69 km then Bulima–Kabwoya 66 km; Rwenkunye–Apac 90.9 km then Apac–Lira–Puranga
100.1 km).

For each report date, the Sentinel-2 reading closest in time is used (30 June -> mid-July "b"; 31 December ->
mid-January "a" of the next year). Satellite progress = share of readable segments sealed, measured as new
sealing since the first reading: (S - S0) / (1 - S0).

Reported progress counts earthworks, drainage and base layers, so it should lead sealing. We report the rank
correlation, the mean gap, and the reported progress level at which sealing becomes visible.

Out: paper/progress_validation.json, chart p3_f17_reported_vs_satellite
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
import road_progress as rp  # noqa: E402

LOTS = {  # road_id -> [(lot, start fraction, end fraction)] along the routed chainage
    "kigumba": [("Kigumba–Bulima", 0, 69 / 135), ("Bulima–Kabwoya", 69 / 135, 1)],
    "rwenkunye": [("Rwenkunye–Apac", 0, 90.9 / 191), ("Apac–Lira–Puranga", 90.9 / 191, 1)],
}
ORDER = [p[0] for p in rp.PERIODS]


def period_for(d: pd.Timestamp) -> str:
    t = d.year + (d.dayofyear - 0.5) / 365.25
    return min(ORDER, key=lambda p: abs(rp.PT[p] - t))


def lot_segments(S, rid, lot):
    s = S[(S.id == rid) & S.used].sort_values("km_at")
    if rid in LOTS:
        a, b = next((x[1], x[2]) for x in LOTS[rid] if x[0] == lot)
        f = s.km_at / s.km_at.max()
        s = s[(f > a) & (f <= b + 1e-9)]
    return s


def sealed_share(s, period):
    t = s.sealed.map(rp.PT)
    return float((t <= rp.PT[period] + 1e-6).mean())


def main():
    R = pd.read_csv(ROOT / "data" / "manual" / "reported_progress_checked.csv", parse_dates=["asof"])
    S = pd.read_parquet(P / "road_progress_segments.parquet")
    rows = []
    for r in R.itertuples():
        s = lot_segments(S, r.road_id, r.lot)
        if len(s) < 3:
            continue
        per = period_for(r.asof)
        s0 = sealed_share(s, ORDER[0])
        st = sealed_share(s, per)
        rows.append(dict(road_id=r.road_id, lot=r.lot, asof=r.asof, period=per, reported=r.physical,
                         planned=r.planned, time_progress=r.time_progress, s0=s0, sat_sealed=st,
                         sat_new=(st - s0) / (1 - s0) if s0 < 0.95 else np.nan, n_segs=len(s),
                         control="rehab" in r.lot))
    V = pd.DataFrame(rows)
    U = V[~V.control & V.sat_new.notna()]
    rho = spearmanr(U.reported, U.sat_new)
    gap = (U.reported / 100 - U.sat_new)
    # reported level at which the satellite first sees >= 10% new sealing, per lot
    first = []
    for lot, g in U.sort_values("asof").groupby("lot"):
        hit = g[g.sat_new >= 0.10]
        before = g[g.sat_new < 0.10]
        first.append(dict(lot=lot, last_reported_before=float(before.reported.max()) if len(before) else None,
                          first_reported_seen=float(hit.reported.iloc[0]) if len(hit) else None))
    out = dict(n_reports=int(len(U)), n_lots=int(U.lot.nunique()), n_roads=int(U.road_id.nunique()),
               spearman=dict(rho=round(float(rho.statistic), 2), p=float(f"{rho.pvalue:.2g}")),
               mean_gap_reported_minus_sealed=round(float(gap.mean()), 2),
               share_reported_ahead=round(float((gap > 0).mean()), 2),
               first_visible=first,
               behind_plan=dict(n_with_plan=int(V.planned.notna().sum()),
                                n_behind=int((V.planned - V.reported > 5).sum())),
               rows=V.assign(asof=V["asof"].dt.date.astype(str)).round(3).to_dict("records"))
    (ROOT / "paper" / "progress_validation.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(V[["lot", "asof", "period", "reported", "sat_new", "sat_sealed", "n_segs"]].round(2).to_string())
    chart(U, S, R)


def chart(U, S, R):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
    a = ax[0]
    lots = sorted(U.lot.unique())
    cm = plt.get_cmap("tab20")
    for i, lot in enumerate(lots):
        g = U[U.lot == lot].sort_values("asof")
        a.plot(g.reported, 100 * g.sat_new, "-o", ms=3.5, lw=1, color=cm(i % 20), label=lot)
    a.plot([0, 100], [0, 100], color="#8a8a85", lw=0.8, ls="--")
    a.set_xlabel("reported physical progress (%)")
    a.set_ylabel("satellite: new sealing since 2017 (% of km)")
    a.set_title("a  Reported progress leads visible sealing", loc="left")
    a.legend(fontsize=6.2, loc="upper left")
    b = ax[1]
    for i, lot in enumerate(lots):
        g = R[R.lot == lot].sort_values("asof")
        rid = g.road_id.iloc[0]
        s = lot_segments(S, rid, lot)
        s0 = sealed_share(s, ORDER[0])
        ser = [(rp.PT[p], 100 * (sealed_share(s, p) - s0) / (1 - s0)) for p in ORDER]
        b.plot([x for x, _ in ser], [y for _, y in ser], color=cm(i % 20), lw=1.2)
        b.plot(g["asof"].dt.year + g["asof"].dt.dayofyear / 365.25, g.physical, "o", ms=4, color=cm(i % 20), mfc="white")
    b.set_xlabel("year   (line: satellite new sealing; circles: reported physical progress)")
    b.set_ylabel("%")
    b.set_title("b  Over time, by contract", loc="left")
    fig.tight_layout()
    f = ROOT / "posts" / "media" / "chart_reported_vs_satellite.png"
    fig.savefig(f, dpi=200)
    shutil.copy(f, ROOT / "paper" / "figures" / "p3_f17_reported_vs_satellite.png")


if __name__ == "__main__":
    main()
