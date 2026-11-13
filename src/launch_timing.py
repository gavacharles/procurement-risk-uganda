"""Paper 3, extensions (1)-(2): ceremonial launches vs elections, and launch vs works visible from space.

(a) Election timing. Dated public launch ceremonies of road works (groundbreakings, flag-offs; collected from
news and government pages, data/manual/road_launches.csv) are placed in the electoral cycle (general elections
18 Feb 2016, 14 Jan 2021, 15 Jan 2026). Under timing unrelated to elections, the share of launches in the
12 months before an election equals 12 / cycle length. Exact binomial test; robustness without inferred dates
and for presidential ceremonies only; placebo distribution from shifting the election dates by -24..+24 months.
Caveat: launches are collected from news, whose coverage may itself rise before elections.

(b) Launch vs visible works. For roads in the satellite sample with a launch date, the first reading after the
launch at which works are visible: either sealing starts (sealed share +0.10) or the corridor is disturbed
(at least 25% of kilometres change brightness by more than 2 SD of their own pre-launch readings, in two
consecutive readings). Roads not yet showing works are censored at the latest reading.

Out: paper/launch_timing.json, charts p3_f13_launch_elections, p3_f14_launch_to_works
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import road_progress as rp  # noqa: E402
import semantic_change as sc  # noqa: E402

ELECTIONS = pd.to_datetime(["2016-02-18", "2021-01-14", "2026-01-15", "2031-01-15"])
WIN = 12


def to_year(d):
    d = pd.Timestamp(d)
    return d.year + (d.dayofyear - 0.5) / 365.25


def months_to_next(d, elections=ELECTIONS):
    nxt = elections[elections > d]
    prev = elections[elections <= d]
    return ((nxt[0] - d).days / 30.44 if len(nxt) else np.nan,
            ((nxt[0] - prev[-1]).days / 30.44) if len(nxt) and len(prev) else np.nan)


def election_test(L, label):
    x = L.copy()
    m = [months_to_next(d) for d in x.date]
    x["m_to_election"], x["cycle_m"] = [a for a, _ in m], [b for _, b in m]
    k = int((x.m_to_election <= WIN).sum())
    n = int(len(x))
    p0 = float((WIN / x.cycle_m).mean())
    t = binomtest(k, n, p0, alternative="greater")
    return dict(sample=label, n=n, within_12m=k, share=k / n if n else None, expected_share=p0, p_value=float(t.pvalue))


def placebo(L):
    out = []
    for s in range(-24, 25):
        el = ELECTIONS + pd.DateOffset(months=s)
        m = [months_to_next(d, el)[0] for d in L.date]
        out.append(dict(shift_months=s, share=float(np.mean(np.array(m) <= WIN))))
    return out


def visible_works(rid, t_launch, order, model, cols):
    rows = []
    for p in order:
        f = rp.PC / f"{rid}_{p}.parquet"
        if not f.exists():
            continue
        F = pd.read_parquet(f).set_index("seg").drop(columns=["filled"], errors="ignore")
        if "fv" not in F.columns:
            F = sc.harmonise_cached(F, 0.0)
        rows.append((rp.PT[p], F["d30_bright"] if "d30_bright" in F else None))
    if not rows:
        return None
    B = pd.DataFrame({t: s for t, s in rows if s is not None}).sort_index(axis=1)
    pre = B.loc[:, B.columns < t_launch]
    post = B.loc[:, B.columns >= t_launch]
    if pre.shape[1] < 2 or post.shape[1] < 1:
        return dict(baseline_readings=int(pre.shape[1]))
    mu, sd = pre.mean(axis=1), pre.std(axis=1).clip(lower=0.01)
    z = post.sub(mu, axis=0).div(sd, axis=0).abs()
    share = (z > 2).mean(axis=0)
    hit = share.to_numpy() >= 0.25
    t_dist = next((float(share.index[i]) for i in range(len(hit) - 1) if hit[i] and hit[i + 1]), None)
    return dict(baseline_readings=int(pre.shape[1]), t_disturbance=t_dist,
                disturbed_share_latest=float(share.iloc[-1]))


def main():
    L = pd.read_csv(ROOT / "data" / "manual" / "road_launches.csv", parse_dates=["date"])
    L = L[L.date.notna()]
    res = dict(election_tests=[election_test(L, "all launches"),
                               election_test(L[L.date_quality == "exact"], "exact dates only"),
                               election_test(L[L.officiated_by.str.startswith("President")], "presidential ceremonies")],
               placebo=placebo(L))
    # expected share over the window actually searched (first to last launch found), not the whole cycle
    lo, hi = L.date.min(), L.date.max()
    pre = sum(max(0, (min(e, hi) - max(e - pd.DateOffset(months=WIN), lo)).days) for e in ELECTIONS)
    p_win = pre / (hi - lo).days
    k = int(sum(months_to_next(d)[0] <= WIN for d in L.date))
    res["election_tests"].append(dict(sample="all launches; expected share over the searched window", n=int(len(L)), within_12m=k,
                                      share=k / len(L), expected_share=float(p_win),
                                      p_value=float(binomtest(k, len(L), p_win, alternative="greater").pvalue)))
    m = [months_to_next(d)[0] for d in L.date]
    L["m_to_election"] = m
    # (b) launch vs visible works for roads in the satellite sample
    R = pd.read_parquet(ROOT / "data" / "processed" / "road_progress_roads.parquet").set_index("id")
    model, cols, _ = rp.train()
    order = [p[0] for p in rp.PERIODS]
    rows = []
    for r in L[L.id.notna()].itertuples():
        if r.id not in R.index:
            continue
        rr = R.loc[r.id]
        tl = to_year(r.date)
        vw = visible_works(r.id, tl, order, model, cols) or {}
        t_seal = rr.t_onset if pd.notna(rr.t_onset) and rr.t_onset >= tl - 0.1 else None
        cands = [v for v in (vw.get("t_disturbance"), t_seal) if v is not None and v >= tl - 0.1]
        t_vis = min(cands) if cands else None
        rows.append(dict(id=r.id, road=r.road, launch=str(r.date.date()), t_launch=tl, t_sealing_onset=t_seal,
                         t_disturbance=vw.get("t_disturbance"), t_visible_works=t_vis,
                         lag_years=(t_vis - tl) if t_vis else None, censored=t_vis is None,
                         years_observed_after_launch=float(rr.t_last - tl), status=rr.status,
                         baseline_readings=vw.get("baseline_readings")))
    W = pd.DataFrame(rows)
    res["launch_to_works"] = W.to_dict("records")
    (ROOT / "paper" / "launch_timing.json").write_text(json.dumps(res, indent=2, default=lambda v: None if pd.isna(v) else float(v)))
    print(json.dumps(res["election_tests"], indent=1))
    print(W[["road", "launch", "t_visible_works", "lag_years", "censored", "years_observed_after_launch", "status"]].round(2).to_string(index=False, max_colwidth=45))
    charts(L, res, W)


def charts(L, res, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visuals import BLUE, ORANGE, INK, INK2, title
    out = ROOT / "posts" / "media"
    # (a) launches in the electoral cycle
    H = 5.0
    fig = plt.figure(figsize=(10, H))
    e0 = res["election_tests"][0]
    title(fig, "Road launches cluster before elections",
          f"Dated launch ceremonies of road works, 2018–2025 (n = {e0['n']}). {e0['within_12m']} fell in the 12 months before a general\n"
          f"election ({e0['share']:.0%}) against {e0['expected_share']:.0%} expected if timing were unrelated (binomial p = {e0['p_value']:.3f}).\n"
          "Right: the same share with the election dates shifted (placebo).",
          source="News and government reports (data/manual/road_launches.csv)")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.06, 2.75 / H, 0.9, 1.0 / H])
    for e in ELECTIONS[:3]:
        ax.axvline(to_year(e), color=INK, lw=1.5)
        ax.axvspan(to_year(e) - 1, to_year(e), color=ORANGE, alpha=0.12, lw=0)
        ax.text(to_year(e) + 0.05, 1.35, f"election\n{e:%b %Y}", fontsize=7.5, color=INK2, va="top")
    yrs = [to_year(d) for d in L.date]
    ax.scatter(yrs, np.ones(len(yrs)), s=60, color=[ORANGE if m_ <= WIN else BLUE for m_ in L.m_to_election], zorder=3, edgecolor="white")
    ax.set_xlim(2015.8, 2026.4)
    ax.set_ylim(0.6, 1.4)
    ax.set_yticks([])
    ax.grid(False)
    ax.tick_params(labelsize=8)
    ax2 = fig.add_axes([0.06, 0.8 / H, 0.42, 1.45 / H])
    bins = np.arange(0, 61, 6)
    ax2.hist(L.m_to_election, bins=bins, color=BLUE, edgecolor="white")
    ax2.axhline(len(L) / (60 / 6), color=INK2, ls="--", lw=1)
    ax2.set_xlabel("Months to the next election", fontsize=8.5)
    ax2.set_ylabel("Launches", fontsize=8.5)
    ax2.invert_xaxis()
    ax2.tick_params(labelsize=8)
    ax3 = fig.add_axes([0.56, 0.8 / H, 0.4, 1.45 / H])
    pl = pd.DataFrame(res["placebo"])
    ax3.plot(pl.shift_months, pl.share, color="#9aa0a6", lw=1.5)
    ax3.scatter([0], [pl.loc[pl.shift_months == 0, "share"].iloc[0]], color=ORANGE, s=50, zorder=3)
    ax3.set_xlabel("Election dates shifted by (months)", fontsize=8.5)
    ax3.set_ylabel("Share in prior 12 months", fontsize=8.5)
    ax3.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax3.tick_params(labelsize=8)
    fig.savefig(out / "chart_launch_elections.png", dpi=200)
    shutil.copy(out / "chart_launch_elections.png", ROOT / "paper" / "figures" / "p3_f13_launch_elections.png")
    plt.close(fig)
    # (b) launch to visible works
    D = W.sort_values("t_launch")
    H = 0.45 * len(D) + 2.0
    fig = plt.figure(figsize=(9, H))
    title(fig, "From ceremony to works you can see from space",
          "Circle = launch ceremony; diamond = first works visible in Sentinel-2 (corridor disturbed or sealing starts);\n"
          "arrow = no works visible yet at the latest image (Jul 2026). Election dates marked.",
          source="News reports · Copernicus Sentinel-2")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.32, 0.7 / H, 0.63, 1 - 2.0 / H])
    for k, r in enumerate(D.itertuples()):
        y = len(D) - 1 - k
        end = r.t_visible_works if pd.notna(r.t_visible_works) else r.t_launch + r.years_observed_after_launch
        ax.plot([r.t_launch, end], [y, y], color=ORANGE if r.censored else BLUE, lw=3)
        ax.plot(r.t_launch, y, "o", color=INK, ms=6)
        if r.censored:
            ax.annotate("", xy=(end + 0.35, y), xytext=(end, y), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))
        else:
            ax.plot(end, y, "D", color=BLUE, ms=6)
        ax.text(2017.9, y, r.road, ha="right", va="center", fontsize=8)
        lab = f"{r.lag_years:.1f} y" if pd.notna(r.lag_years) else f"> {r.years_observed_after_launch:.1f} y"
        ax.text(end + (0.45 if r.censored else 0.12), y, lab, va="center", fontsize=7.5, color=INK2)
    for e in ELECTIONS[:3]:
        if to_year(e) > 2018:
            ax.axvline(to_year(e), color=INK2, lw=1, ls="--")
    ax.set_xlim(2018, 2027.3)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "chart_launch_to_works.png", dpi=200)
    shutil.copy(out / "chart_launch_to_works.png", ROOT / "paper" / "figures" / "p3_f14_launch_to_works.png")
    plt.close(fig)
    print("wrote charts")


if __name__ == "__main__":
    main()
