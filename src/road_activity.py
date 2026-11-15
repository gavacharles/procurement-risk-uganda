"""Follow-on step 3b: monthly construction activity and inactivity from Sentinel-1, and whether it is real.

Input: data/processed/road_s1.parquet (road_s1.py): monthly gamma0 VV/VH at each 1 km segment centre and
30 m / 60 m to each side, one relative orbit per road.

Signal. Road works change the corridor: clearing removes vegetation (VH volume scattering falls), earthworks and
gravel base change roughness and moisture, and asphalt is smooth (VV falls). Rain and season change the
surroundings too, so each reading is referenced to the land 60 m away: c = dB(road) - dB(60 m), for VV and VH.
Per segment and month, a step score compares the median of c over the next 3 months with the previous 3,
divided by the segment's own noise (1.4826 x MAD of month-to-month changes); score = max over VV and VH.

Activity. A segment is "active" in a month when |step| >= Z. Road activity A(t) = share of readable segments
active, smoothed over 3 months. The threshold A* is the 95th percentile of A(t) on the paved control road with
no surface works in the window (Fort Portal–Hima), so roughly 1 month in 20 would be called active on a road
with no works. A road is active in month t if A(t) > A*; an inactivity spell is a run of >= 3 inactive months
between the first and last active months of the works window.

Change points. PELT (ruptures, RBF cost) on the road-mean contrast series, as a second view of regime changes.

Validation (does the radar see works?):
  V-a  Between consecutive government monitoring reports (lot level), the increase in reported physical
       progress per year against mean A(t) over the interval (Spearman).
  V-b  For segments sealed in Sentinel-2, the month of the largest radar step against the sealing period:
       if radar sees works, the step should fall near or before sealing.
  V-c  Controls: false activity rate on the no-works control.

Out: data/processed/road_activity.parquet (road x month), paper/road_activity.json, chart p3_f18_activity
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
import progress_validation as pv  # noqa: E402
import road_progress as rp  # noqa: E402

Z = 3.0
W = 3
CONTROL = "ftportal_hima"
MIN_SPELL = 3


def db(x):
    return 10 * np.log10(x)


def ym_to_t(ym):
    ym = str(ym)
    return int(ym[:4]) + (int(ym[4:]) - 0.5) / 12


def segment_steps(D: pd.DataFrame) -> pd.DataFrame:
    """D: one road, rows seg x ym. Returns seg x ym step scores (z units)."""
    D = D.copy()
    D["cvv"] = db(D.vv) - db(D.vv60)
    D["cvh"] = db(D.vh) - db(D.vh60)
    out = []
    months = sorted(D.ym.unique())
    for seg, g in D.groupby("seg"):
        g = g.set_index("ym").reindex(months)
        zs = []
        for c in ("cvv", "cvh"):
            x = g[c]
            noise = 1.4826 * np.nanmedian(np.abs(x.diff() - np.nanmedian(x.diff())))
            if not np.isfinite(noise) or noise == 0:
                zs.append(pd.Series(np.nan, index=x.index))
                continue
            after = x[::-1].rolling(W, min_periods=2).median()[::-1]  # t .. t+W-1
            before = x.shift(1).rolling(W, min_periods=2).median()  # t-W .. t-1
            zs.append((after - before) / noise)
        zv, zh = zs[0].to_numpy(), zs[1].to_numpy()
        # the larger-magnitude step of VV and VH (sign kept); NaN only where both are missing
        score = pd.Series(np.where(np.nan_to_num(np.abs(zv), nan=-1) >= np.nan_to_num(np.abs(zh), nan=-1), zv, zh),
                          index=g.index)
        out.append(pd.DataFrame(dict(seg=seg, ym=months, step=score.values, cvv=g.cvv.values, cvh=g.cvh.values)))
    return pd.concat(out, ignore_index=True)


def road_activity(St: pd.DataFrame) -> pd.DataFrame:
    g = St.groupby("ym")
    A = pd.DataFrame(dict(n=g.step.count(), active=g.step.apply(lambda s: float((s.abs() >= Z).mean()) if s.notna().any() else np.nan),
                          cvv=g.cvv.median(), cvh=g.cvh.median())).reset_index()
    A["A"] = A.active.rolling(3, center=True, min_periods=1).mean()
    A["t"] = A.ym.map(ym_to_t)
    return A


def spells(A: pd.DataFrame, thr: float, t_done=None):
    act = A[A.A > thr]
    if act.empty:
        return [], None, None
    t0, t1 = act.t.min(), act.t.max()
    if t_done is not None and np.isfinite(t_done):
        t1 = min(t1, t_done)
    w = A[(A.t >= t0) & (A.t <= t1)].sort_values("t")
    runs, cur = [], []
    for r in w.itertuples():
        if r.A <= thr:
            cur.append(r.t)
        else:
            if len(cur) >= MIN_SPELL:
                runs.append((cur[0], cur[-1], len(cur)))
            cur = []
    if len(cur) >= MIN_SPELL:
        runs.append((cur[0], cur[-1], len(cur)))
    return runs, t0, t1


def changepoints(A: pd.DataFrame):
    try:
        import ruptures as rpt
    except ImportError:
        return []
    x = A[["cvv", "cvh"]].interpolate(limit_direction="both").to_numpy()
    if len(x) < 24 or not np.isfinite(x).all():
        return []
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    bk = rpt.Pelt(model="rbf", min_size=6).fit(x).predict(pen=8)
    return [float(A.t.iloc[b]) for b in bk[:-1]]


def main():
    S1 = pd.read_parquet(P / "road_s1.parquet")
    R = pd.read_parquet(P / "road_progress_roads.parquet").set_index("id")
    Sg = pd.read_parquet(P / "road_progress_segments.parquet")
    Sg["sealed"] = Sg.sealed.where(Sg.sealed.isin(list(rp.PT)))  # never-sealed segments are stored as "None"
    steps, acts = {}, {}
    for rid, D in S1.groupby("id"):
        if D.ym.nunique() < 24:
            continue
        steps[rid] = segment_steps(D)
        acts[rid] = road_activity(steps[rid])
    ctrl = acts[CONTROL]
    thr = float(np.nanpercentile(ctrl.A, 95))
    rows, summ = [], {}
    for rid, A in acts.items():
        A = A.assign(id=rid, active_road=A.A > thr)
        rows.append(A)
        t_done = R.t_done.get(rid, np.nan) if rid in R.index else np.nan
        runs, t0, t1 = spells(A, thr, t_done)
        longest = max(runs, key=lambda x: x[2]) if runs else None
        summ[rid] = dict(kind=R.kind.get(rid) if rid in R.index else None, months=int(A.ym.nunique()),
                         share_active=round(float(A.active_road.mean()), 3),
                         first_active=t0, works_window_end=t1, n_spells=len(runs),
                         longest_spell_months=longest[2] if longest else 0,
                         longest_spell=[round(longest[0], 2), round(longest[1], 2)] if longest else None,
                         changepoints=[round(c, 2) for c in changepoints(A)],
                         t_tender=None if rid not in R.index or pd.isna(R.t_tender.get(rid)) else round(float(R.t_tender[rid]), 2))
    AA = pd.concat(rows, ignore_index=True)
    AA.to_parquet(P / "road_activity.parquet", index=False)

    # V-a: reported progress increments vs radar activity, lot level
    rep = pd.read_csv(ROOT / "data" / "manual" / "reported_progress_checked.csv", parse_dates=["asof"])
    va = []
    for lot, g in rep[~rep.lot.str.contains("rehab")].sort_values("asof").groupby("lot"):
        rid = g.road_id.iloc[0]
        if rid not in steps:
            continue
        segs = pv.lot_segments(Sg, rid, lot).seg
        St = steps[rid][steps[rid].seg.isin(segs)]
        Al = road_activity(St)
        for a, b in zip(g.itertuples(), g.iloc[1:].itertuples()):
            ta = a.asof.year + a.asof.dayofyear / 365.25
            tb = b.asof.year + b.asof.dayofyear / 365.25
            w = Al[(Al.t > ta) & (Al.t <= tb)]
            if len(w) >= 3 and tb > ta:
                va.append(dict(lot=lot, t0=round(ta, 2), t1=round(tb, 2), d_reported_per_y=(b.physical - a.physical) / (tb - ta),
                               mean_activity=float(w.active.mean()), share_active_months=float((w.A > thr).mean())))
    VA = pd.DataFrame(va)
    r_a = spearmanr(VA.d_reported_per_y, VA.mean_activity) if len(VA) >= 5 else None

    # V-b: radar step timing vs Sentinel-2 sealing, per segment
    vb = []
    for rid, St in steps.items():
        sg = Sg[(Sg.id == rid) & Sg.used & Sg.sealed.notna()]
        first = min(rp.PT.values())
        for s in sg.itertuples():
            ts = rp.PT[s.sealed]
            if ts <= first:
                continue
            x = St[St.seg == s.seg].assign(t=lambda d: d.ym.map(ym_to_t))
            x = x[(x.t > ts - 3) & (x.t < ts + 1.5) & x.step.notna()]
            if x.empty:
                continue
            k = x.step.abs().idxmax()
            vb.append(dict(id=rid, seg=s.seg, t_seal=ts, t_step=float(x.loc[k, "t"]), z=float(x.loc[k, "step"])))
    VB = pd.DataFrame(vb)
    VB["lag_y"] = VB.t_step - VB.t_seal
    # placebo: same for unsealed readable segments on upgrade roads, around a random period
    rng = np.random.default_rng(0)
    pl = []
    for rid, St in steps.items():
        sg = Sg[(Sg.id == rid) & Sg.used & Sg.sealed.isna()]
        for s in sg.itertuples():
            x = St[St.seg == s.seg].assign(t=lambda d: d.ym.map(ym_to_t))
            x = x[x.step.notna()]
            if len(x) > 30:
                ts = rng.uniform(x.t.min() + 3, x.t.max() - 1.5)
                y = x[(x.t > ts - 3) & (x.t < ts + 1.5)]
                if len(y):
                    pl.append(float(y.step.abs().max()))
    out = dict(
        threshold=dict(Z=Z, A_star=round(thr, 3), control=CONTROL,
                       control_share_active=round(float((ctrl.A > thr).mean()), 3)),
        roads=summ,
        va_reported_vs_activity=None if r_a is None else dict(rho=round(float(r_a.statistic), 2), p=float(f"{r_a.pvalue:.2g}"),
                                                              n_intervals=len(VA), n_lots=int(VA.lot.nunique())),
        vb_step_vs_sealing=dict(n_segments=len(VB), median_lag_y=round(float(VB.lag_y.median()), 2),
                                share_step_before_or_at_seal=round(float((VB.lag_y <= 0.25).mean()), 2),
                                median_abs_z_sealed=round(float(VB.z.abs().median()), 2),
                                median_abs_z_placebo_unsealed=round(float(np.median(pl)), 2) if pl else None,
                                n_placebo=len(pl)),
    )
    (ROOT / "paper" / "road_activity.json").write_text(json.dumps(out, indent=2, default=float))
    VA.to_csv(ROOT / "paper" / "tables" / "road_activity_va.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "roads"}, indent=2))
    for rid, s in summ.items():
        print(f"{rid:14s} {s['kind'] or '':14s} active {s['share_active']:.2f}  first {s['first_active']}  spells {s['n_spells']} "
              f"longest {s['longest_spell_months']} m {s['longest_spell']}  cps {s['changepoints']}")
    chart(AA, thr, rep, VB, pl)


def chart(AA, thr, rep, VB, pl):
    ids = ["tirinyi", "kigumba", "muyembe", "rwenkunye", "luwero", "koboko", "kyenjojo", CONTROL]
    ids = [i for i in ids if i in AA.id.unique()]
    fig = plt.figure(figsize=(10, 1.15 * len(ids) + 2.6))
    gs = fig.add_gridspec(len(ids) + 1, 2, height_ratios=[1] * len(ids) + [2.2], width_ratios=[2.2, 1])
    R = pd.read_parquet(P / "road_progress_series.parquet")
    for k, rid in enumerate(ids):
        a = fig.add_subplot(gs[k, :])
        A = AA[AA.id == rid]
        a.fill_between(A.t, 0, A.A, color="#2a78d6", alpha=0.35, lw=0)
        a.plot(A.t, A.A, color="#2a78d6", lw=0.8)
        a.axhline(thr, color="#8a8a85", lw=0.6, ls="--")
        a.fill_between(A.t, 0, 1, where=A.A <= thr, color="#eb6834", alpha=0.07, lw=0, transform=a.get_xaxis_transform())
        s = R[R.id == rid]
        a2 = a.twinx()
        a2.step(s.t, s.sealed_share, where="post", color="#0b0b0b", lw=1)
        g = rep[rep.road_id == rid]
        a2.plot(g["asof"].dt.year + g["asof"].dt.dayofyear / 365.25, g.physical / 100, "o", ms=3, mfc="white", color="#1baf7a")
        a2.set_ylim(0, 1.05)
        a2.set_yticks([])
        a.set_ylim(0, max(0.6, A.A.max() * 1.1))
        a.set_xlim(2017, 2026.9)
        a.set_yticks([])
        a.text(0.005, 0.8, rp.ROADS[[r[0] for r in rp.ROADS].index(rid)][1], transform=a.transAxes, fontsize=7.5)
        if k < len(ids) - 1:
            a.set_xticklabels([])
        a.grid(False)
    a = fig.add_subplot(gs[-1, 0])
    if len(VB):
        a.hist(12 * VB.lag_y, bins=np.arange(-36, 19, 3), color="#2a78d6", alpha=0.8)
        a.axvline(0, color="#0b0b0b", lw=0.8)
        a.set_xlabel("largest radar step minus Sentinel-2 sealing reading (months)")
        a.set_ylabel("segments")
        a.set_title("b  Timing of the largest radar step (no clear peak at sealing)", loc="left")
    b = fig.add_subplot(gs[-1, 1])
    if len(VB) and pl:
        b.boxplot([VB.z.abs(), pl], labels=["sealed\nsegments", "never-sealed\n(placebo)"], widths=0.5)
        b.set_ylabel("largest |step| (z)")
        b.set_title("c  Step size: sealed = placebo", loc="left")
    fig.text(0.01, 0.995, "a  Monthly radar activity (blue; dashed: threshold; shaded: inactive), sealed share (black), "
             "reported progress (green)", va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    f = ROOT / "posts" / "media" / "chart_road_activity.png"
    fig.savefig(f, dpi=200)
    shutil.copy(f, ROOT / "paper" / "figures" / "p5_f2_radar.png")


if __name__ == "__main__":
    main()
