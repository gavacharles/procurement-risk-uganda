"""(3) Natural experiments: elections, fiscal year-end, COVID-19.

Timing uses the tender ADVERT date (OCDS tenderPeriod.startDate), not the award date
(which is the portal entry date; see memory note / Paper 2 correction).

Windows
  pre_election   0-6 months before a general election (18 Feb 2016, 14 Jan 2021, 15 Jan 2026 [verify])
  post_election  0-6 months after
  fy_q4          April-June (last quarter of Uganda's July-June fiscal year)
  covid          18 Mar 2020 (first lockdown) - 31 Dec 2020

Outcomes (tender records): direct procurement, non-open method, award at 99-100% of estimate,
award exactly equal to estimate, award >10% above estimate, short bidding window.

Model: linear probability, y ~ windows + C(year) + C(entity), SE clustered by entity.
Run on ALL sectors (power) and on WORKS only.

Out: paper/natural_experiments.json, posts/media/chart_election_event.png (+ docs/media)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import is_works  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
ELECTIONS = [pd.Timestamp("2016-02-18", tz="UTC"), pd.Timestamp("2021-01-14", tz="UTC"), pd.Timestamp("2026-01-15", tz="UTC")]
COVID = (pd.Timestamp("2020-03-18", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC"))
OUTCOMES = {"direct": "Direct procurement", "non_open": "Not openly bid", "bunched": "Award at 99–100% of estimate",
            "exact": "Award exactly equals estimate", "above_est": "Award >10% above estimate", "short_window": "Bidding window <10 days"}


def months_to_nearest_election(d: pd.Series) -> pd.Series:
    best = None
    for e in ELECTIONS:
        m = (d - e).dt.days / 30.44
        best = m if best is None else best.where(best.abs() <= m.abs(), m)
    return best


def prepare() -> pd.DataFrame:
    t = pd.read_parquet(P / "tenders.parquet")
    t = t[t.tender_start.between(pd.Timestamp("2015-07-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC"))].copy()
    t["works"] = t.title.map(is_works)
    t["ratio"] = t.award_value_ugx / t.estimate_ugx.where(t.estimate_ugx > 0)
    t["direct"] = (t.method == "direct").astype(float).where(t.method.notna())
    t["non_open"] = (t.method != "open").astype(float).where(t.method.notna())
    t["bunched"] = t.ratio.between(0.99, 0.999999).astype(float).where(t.ratio.between(0.3, 3))
    t["exact"] = np.isclose(t.ratio, 1.0).astype(float)
    t["exact"] = t.exact.where(t.ratio.between(0.3, 3))
    t["above_est"] = (t.ratio > 1.10).astype(float).where(t.ratio.between(0.3, 3))
    days = (t.tender_end - t.tender_start).dt.days
    t["short_window"] = (days < 10).astype(float).where(days.between(0, 400))
    m = months_to_nearest_election(t.tender_start)
    t["m_to_election"] = m
    t["pre_election"] = m.between(-6, 0, inclusive="left").astype(int)
    t["post_election"] = m.between(0, 6).astype(int)
    t["fy_q4"] = t.tender_start.dt.month.isin([4, 5, 6]).astype(int)
    t["covid"] = t.tender_start.between(*COVID).astype(int)
    t["year"] = t.tender_start.dt.year
    t["entity"] = t.buyer_id.astype(str)
    return t


def fit(d: pd.DataFrame, y: str) -> dict:
    s = d.dropna(subset=[y])
    s = s[s.groupby("entity")[y].transform("size") >= 5]
    if len(s) < 500 or s[y].std() == 0:
        return {}
    f = f"{y} ~ pre_election + post_election + fy_q4 + covid + C(year) + C(entity)"
    r = smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": s["entity"]})
    out = dict(n=int(r.nobs), base_rate=float(s[y].mean()))
    for k in ["pre_election", "post_election", "fy_q4", "covid"]:
        out[k] = dict(coef=float(r.params[k]), se=float(r.bse[k]), p=float(r.pvalues[k]),
                      ci95=[float(x) for x in r.conf_int().loc[k]])
    return out


def event_chart(d: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visuals import BLUE, ORANGE, INK2, title
    e = d[d.m_to_election.between(-12, 12)].copy()
    e["mbin"] = np.floor(e.m_to_election).astype(int)
    fig = plt.figure(figsize=(8, 4.6))
    title(fig, "Do procurement practices shift around elections?",
          "Raw monthly shares, all tenders, pooled over the 2016, 2021 and 2026 elections")
    for i, (y, col) in enumerate([("direct", BLUE), ("bunched", ORANGE)]):
        ax = fig.add_axes([0.08 + i * 0.47, 0.17, 0.40, 0.58])
        g = e.dropna(subset=[y]).groupby("mbin")[y].agg(["mean", "size"])
        g = g[g["size"] >= 30]
        ax.plot(g.index, g["mean"], color=col, lw=2, marker="o", ms=4)
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.axvspan(-6, 0, color=col, alpha=0.07, lw=0)
        ax.text(-3, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] else 0, "6 months\nbefore", ha="center", va="top", fontsize=9, color=INK2)
        ax.set_xlabel("Months relative to election day")
        ax.set_title(OUTCOMES[y], fontsize=11, loc="left", color=INK2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.grid(axis="x", visible=False)
    out = ROOT / "posts" / "media" / "chart_election_event.png"
    fig.savefig(out, dpi=200)
    shutil.copy(out, ROOT / "docs" / "media" / out.name)


def main():
    d = prepare()
    res = {"elections": [str(e.date()) for e in ELECTIONS], "note": "2026 election date to verify",
           "samples": {"all": int(len(d)), "works": int(d.works.sum())}}
    for name, s in [("all_sectors", d), ("works", d[d.works])]:
        res[name] = {y: fit(s, y) for y in OUTCOMES}
    # monthly FY profile (descriptive)
    res["fy_month_profile_all"] = d.groupby(d.tender_start.dt.month).agg(
        tenders=("ocid", "size"), direct=("direct", "mean"), bunched=("bunched", "mean")).round(3).reset_index().to_dict("records")
    (ROOT / "paper" / "natural_experiments.json").write_text(json.dumps(res, indent=2, default=float))
    event_chart(d)
    for name in ["all_sectors", "works"]:
        print(f"\n== {name}")
        for y, r in res[name].items():
            if not r:
                continue
            line = f"{OUTCOMES[y]:32s} n={r['n']:6d} base={r['base_rate']:.3f}"
            for k in ["pre_election", "post_election", "fy_q4", "covid"]:
                c = r[k]
                star = "***" if c["p"] < .01 else "**" if c["p"] < .05 else "*" if c["p"] < .1 else ""
                line += f" | {k} {c['coef']:+.3f}{star}"
            print(line)


if __name__ == "__main__":
    main()
