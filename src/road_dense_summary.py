"""Summary for road_dense.py: monthly Sentinel-2 and Landsat 2013-2026 against the half-yearly Sentinel-2 record.

Reference sets for the per-reading error by calendar month (monthly Sentinel-2):
  paved   - every used segment of the two paved controls with no surface works (Fort Portal-Hima, Ishaka);
  unpaved - used segments never sealed (half-yearly record) on roads where no paving was detected, readings
            before 2024 (works on these roads had not reached sealing).
Seal dates use the main pipeline's change-point rule (road_progress.sealed_period) on each record.
Milestones (half / completion) use the same thresholds as the main pipeline, from the tender on.

Out: paper/road_dense.json, data/processed/road_dense_{s2,landsat}_segments.parquet,
     paper/figures/p3_f18_dense_landsat.png
"""
from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import progress_validation as pv  # noqa: E402
import road_dense as rd  # noqa: E402
import road_progress as rp  # noqa: E402
import semantic_change as sc  # noqa: E402

ROOT, P = rp.ROOT, rp.P
HALF = [p[0] for p in rp.PERIODS]
PAVED_REF = ["ftportal_hima", "ishaka"]
UNPAVED_REF = ["koboko", "kyenjojo", "kalule", "namagumba"]
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8a85", "#33332f"


def states_from(cache, rid, keys, model, cols, harmonise=True):
    out = {}
    for k in keys:
        f = cache / f"{rid}_{k}.parquet"
        if not f.exists():
            continue
        F = pd.read_parquet(f).set_index("seg").drop(columns=["filled"], errors="ignore")
        if harmonise and "road_ndvi" in F.columns and "fv" not in F.columns:
            F = sc.harmonise_cached(F, 0.0)
        out[k] = rp.segment_states(F, model, cols) if "road_ndvi" in F.columns else pd.Series(dtype=object)
    return pd.DataFrame(out)


def seal_times(S: pd.DataFrame, tmap: dict) -> pd.Series:
    return pd.Series({s: (tmap[p] if (p := rp.sealed_period(S.loc[s].tolist(), list(S.columns))) else np.nan)
                      for s in S.index})


def share_series(seal_t: pd.Series, times) -> pd.Series:
    return pd.Series({t: float((seal_t <= t + 1e-6).mean()) for t in times})


def first_after(sh: pd.Series, th, t_tender, rel_to=None):
    for t, v in sh.items():
        if not np.isnan(t_tender) and t < t_tender - 0.25:
            continue
        if (v - rel_to if rel_to is not None else v) >= th:
            return float(t)
    return None


def main():
    seg_main = pd.read_parquet(P / "road_progress_segments.parquet")
    seg_main["t_half"] = seg_main.sealed.map(lambda p: rp.PT.get(p, np.nan))
    roads = pd.read_parquet(P / "road_progress_roads.parquet").set_index("id")
    used = {rid: g[g.used].seg.tolist() for rid, g in seg_main.groupby("id")}
    model, cols, _ = rp.train()
    lmodel, lcols, linfo = rd.ls_train()
    months = [m[0] for m in rd.MONTHS]
    lpers = [p[0] for p in rd.LPERIODS]
    out = dict(landsat_classifier=linfo)

    # ------------------------------------------------ A. monthly Sentinel-2
    rows, mseal, mshare = [], {}, {}
    for rid in used:
        S = states_from(rd.DC, rid, months, model, cols)
        if S.empty:
            continue
        S = S.reindex(index=used[rid]).reindex(columns=[m for m in months if m in S.columns])
        for s in S.index:
            for m in S.columns:
                rows.append(dict(id=rid, seg=s, month=m, state=S.loc[s, m]))
        mseal[rid] = seal_times(S, rd.MT)
        mshare[rid] = share_series(mseal[rid], [rd.MT[m] for m in S.columns])
    M = pd.DataFrame(rows)
    M["cal"] = M.month.str[5:].astype(int)
    M["year"] = M.month.str[:4].astype(int)
    M["valid"] = M.state.isin(["asphalt", "unpaved"])
    M.to_parquet(P / "road_dense_s2_segments.parquet", index=False)
    per_year = M[M.valid].groupby(["id", "seg", "year"]).size()
    readable_by_cal = M.groupby("cal").valid.mean()
    # per-reading error by calendar month on reference sets
    pav = M[M.id.isin(PAVED_REF) & M.valid]
    unp_segs = seg_main[seg_main.id.isin(UNPAVED_REF) & seg_main.used & seg_main.sealed.isin(["None", None])]
    unp = M[M.valid & (M.year < 2024)].merge(unp_segs[["id", "seg"]], on=["id", "seg"])
    err_pav = (pav.state != "asphalt").groupby(pav.cal).mean()
    err_unp = (unp.state != "unpaved").groupby(unp.cal).mean()
    jan_jul = [1, 7]
    out["monthly_s2"] = dict(
        n_roads=int(M.id.nunique()), n_segment_months=int(len(M)), n_readings=int(M.valid.sum()),
        readings_per_segment_year_median=float(per_year.median()),
        readable_share_by_month={int(k): round(float(v), 3) for k, v in readable_by_cal.items()},
        error_paved_by_month={int(k): round(float(v), 3) for k, v in err_pav.items()},
        error_unpaved_by_month={int(k): round(float(v), 3) for k, v in err_unp.items()},
        error_paved=dict(all=round(float((pav.state != "asphalt").mean()), 3), n=int(len(pav)),
                         jan_jul=round(float((pav[pav.cal.isin(jan_jul)].state != "asphalt").mean()), 3)),
        error_unpaved=dict(all=round(float((unp.state != "unpaved").mean()), 3), n=int(len(unp)),
                           jan_jul=round(float((unp[unp.cal.isin(jan_jul)].state != "unpaved").mean()), 3)),
    )
    # segment seal dates: monthly vs half-yearly (segments sealed after the first half-yearly reading)
    cmp = []
    for rid, st in mseal.items():
        g = seg_main[(seg_main.id == rid) & seg_main.used].set_index("seg")
        for s, tm in st.items():
            th = g.t_half.get(s, np.nan)
            cmp.append(dict(id=rid, seg=s, t_month=tm, t_half=th))
    C = pd.DataFrame(cmp)
    both = C.dropna()
    trans = both[both.t_half > rp.PT[HALF[1]] + 0.01]  # a transition seen in the half-yearly record
    d = trans.t_month - trans.t_half
    out["monthly_s2"]["seal_dates"] = dict(
        n_sealed_half=int(C.t_half.notna().sum()), n_sealed_month=int(C.t_month.notna().sum()),
        n_sealed_both=int(len(both)), n_transitions=int(len(trans)),
        median_diff_years=round(float(d.median()), 2), within_6m=round(float((d.abs() <= 0.5).mean()), 2),
        within_1y=round(float((d.abs() <= 1.0).mean()), 2),
        spearman=round(float(spearmanr(trans.t_month, trans.t_half).statistic), 2),
        only_half=int((C.t_half.notna() & C.t_month.isna()).sum()),
        only_month=int((C.t_month.notna() & C.t_half.isna()).sum()))
    # road milestones and news completion dates
    mil = []
    for rid, sh in mshare.items():
        r = roads.loc[rid]
        if str(r.kind).startswith("control"):
            continue
        sh = sh.sort_index()
        s_t = float(sh[sh.index <= r.t_tender].iloc[-1]) if (sh.index <= r.t_tender).any() else float(sh.iloc[0])
        mil.append(dict(id=rid, t_tender=r.t_tender, half_m=first_after(sh, 0.5, r.t_tender),
                        done_m=first_after(sh, 0.8, r.t_tender), half_h=r.t_half, done_h=r.t_done,
                        complete_h=r.t_complete, media_done=r.t_media_done, early_sealed=bool(r.early_sealed),
                        s_last_m=float(sh.iloc[-1]), s_last_h=r.s_last, s_tender_m=s_t))
    Mi = pd.DataFrame(mil)
    md = Mi.dropna(subset=["media_done"])
    md = md[~md.early_sealed]
    out["monthly_s2"]["milestones"] = Mi.round(2).to_dict("records")
    out["monthly_s2"]["vs_news_completion"] = dict(
        n=int(len(md)), rows=md[["id", "media_done", "done_h", "done_m", "complete_h"]].round(2).to_dict("records"),
        mae_half=round(float((md.complete_h - md.media_done).abs().mean()), 2) if len(md) else None,
        mae_month=round(float((md.done_m - md.media_done).abs().mean()), 2) if md.done_m.notna().any() else None)
    # monitoring reports: monthly reading at the report date vs the half-yearly reading nearest to it
    R = pd.read_csv(ROOT / "data" / "manual" / "reported_progress_checked.csv", parse_dates=["asof"])
    vr = []
    for r in R.itertuples():
        if "rehab" in r.lot or r.road_id not in mseal:
            continue
        s = pv.lot_segments(seg_main, r.road_id, r.lot)
        if len(s) < 3:
            continue
        t = r.asof.year + (r.asof.dayofyear - 0.5) / 365.25
        tm = mseal[r.road_id].reindex(s.seg)
        th = s.set_index("seg").t_half
        def new(ts, t0):
            s0 = float((ts <= t0 + 1e-6).mean())
            st = float((ts <= t + 1e-6).mean())
            return (st - s0) / (1 - s0) if s0 < 0.95 else np.nan
        per = pv.period_for(r.asof)
        sh_h = float((th <= rp.PT[per] + 1e-6).mean())
        s0h = float((th <= rp.PT[HALF[0]] + 1e-6).mean())
        vr.append(dict(road_id=r.road_id, lot=r.lot, asof=str(r.asof.date()), reported=r.physical,
                       sat_month=new(tm, rd.MT[months[0]]),
                       sat_half=(sh_h - s0h) / (1 - s0h) if s0h < 0.95 else np.nan))
    V = pd.DataFrame(vr).dropna()
    rh, rm = spearmanr(V.reported, V.sat_half), spearmanr(V.reported, V.sat_month)
    out["monthly_s2"]["vs_monitoring"] = dict(
        n=int(len(V)), n_lots=int(V.lot.nunique()),
        rho_half=round(float(rh.statistic), 2), rho_month=round(float(rm.statistic), 2),
        p_month=float(f"{rm.pvalue:.2g}"),
        mean_gap_half=round(float((V.reported / 100 - V.sat_half).mean()), 2),
        mean_gap_month=round(float((V.reported / 100 - V.sat_month).mean()), 2),
        rows=V.round(3).to_dict("records"))

    # ------------------------------------------------ B. Landsat 2013-2026
    lrows, lseal, lshare = [], {}, {}
    for rid in used:
        S = states_from(rd.LC, rid, lpers, lmodel, lcols, harmonise=False)
        if S.empty:
            continue
        S = S.reindex(index=used[rid]).reindex(columns=[p for p in lpers if p in S.columns])
        for s in S.index:
            for p in S.columns:
                lrows.append(dict(id=rid, seg=s, period=p, state=S.loc[s, p]))
        lseal[rid] = seal_times(S, rd.LT)
        lshare[rid] = share_series(lseal[rid], [rd.LT[p] for p in S.columns])
    L = pd.DataFrame(lrows)
    L["valid"] = L.state.isin(["asphalt", "unpaved"])
    L.to_parquet(P / "road_dense_landsat_segments.parquet", index=False)
    # agreement with half-yearly Sentinel-2 in the overlap
    H = seg_main[seg_main.used].melt(id_vars=["id", "seg"], value_vars=HALF, var_name="period", value_name="s2")
    A = L[L.valid].merge(H[H.s2.isin(["asphalt", "unpaved"])], on=["id", "seg", "period"])
    agree = (A.state == A.s2)
    po = agree.mean()
    pa = (A.state == "asphalt").mean() * (A.s2 == "asphalt").mean() + (A.state == "unpaved").mean() * (A.s2 == "unpaved").mean()
    # sensor of each Landsat reading (Landsat 7 vs 8/9)
    l7 = []
    for (rid, p), g in L[L.valid].groupby(["id", "period"]):
        F = pd.read_parquet(rd.LC / f"{rid}_{p}.parquet").set_index("seg")
        l7 += [dict(id=rid, seg=s, period=p, l7=F.l7.get(s, np.nan)) for s in g.seg]
    A = A.merge(pd.DataFrame(l7), on=["id", "seg", "period"], how="left")
    by_sensor = {("Landsat 7" if k == 1 else "Landsat 8/9"): round(float((g.state == g.s2).mean()), 3)
                 for k, g in A.groupby(A.l7.fillna(0))}
    # seal dates
    lc = []
    for rid, st in lseal.items():
        g = seg_main[(seg_main.id == rid) & seg_main.used].set_index("seg")
        lc += [dict(id=rid, seg=s, t_ls=t, t_half=g.t_half.get(s, np.nan)) for s, t in st.items()]
    LCmp = pd.DataFrame(lc)
    lt = LCmp.dropna()
    lt = lt[lt.t_half > rp.PT[HALF[1]] + 0.01]
    dl = lt.t_ls - lt.t_half
    readable = L.groupby(L.period.str[:4].astype(int) < 2017).valid.mean()
    out["landsat"] = dict(
        n_roads=int(L.id.nunique()), n_readings=int(L.valid.sum()),
        readable_share=dict(pre2017=round(float(readable.get(True, np.nan)), 3), from2017=round(float(readable.get(False, np.nan)), 3)),
        agreement_with_s2=dict(n=int(len(A)), accuracy=round(float(po), 3), kappa=round(float((po - pa) / (1 - pa)), 3),
                               by_sensor=by_sensor),
        seal_dates=dict(n_transitions=int(len(lt)), median_diff_years=round(float(dl.median()), 2) if len(lt) else None,
                        within_1y=round(float((dl.abs() <= 1.0).mean()), 2) if len(lt) else None,
                        spearman=round(float(spearmanr(lt.t_ls, lt.t_half).statistic), 2) if len(lt) > 2 else None))
    # roads tendered before the Sentinel-2 archive
    pre = []
    for rid, sh in lshare.items():
        r = roads.loc[rid]
        if np.isnan(r.t_tender) or r.t_tender >= 2017.1 or str(r.kind).startswith("control"):
            continue
        sh = sh.sort_index()
        s2_first = rp.PT[HALF[0]]
        pre.append(dict(id=rid, t_tender=round(float(r.t_tender), 2), status_s2=r.status,
                        media_done=r.t_media_done, ls_2013=round(float(sh.iloc[:2].mean()), 2),
                        ls_at_s2_start=round(float(sh[sh.index <= s2_first + 0.01].iloc[-1]), 2),
                        s2_first=round(float(r.s0), 2), ls_half=first_after(sh, 0.5, r.t_tender),
                        ls_done=first_after(sh, 0.8, r.t_tender), s2_done=r.t_complete,
                        pre2017_sealed_km_share=round(float(((lseal[rid] < 2017.0)).mean()), 2)))
    out["landsat"]["pre_archive_roads"] = pd.DataFrame(pre).round(2).to_dict("records")
    (ROOT / "paper" / "road_dense.json").write_text(json.dumps(out, indent=2, default=lambda x: None if pd.isna(x) else float(x)))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("rows", "milestones")} if isinstance(v, dict) else v
                      for k, v in out.items()}, indent=1, default=str)[:6000])
    chart(err_pav, err_unp, mshare, lshare, roads, R, seg_main, mseal)


def chart(err_pav, err_unp, mshare, lshare, roads, R, seg_main, mseal):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw=dict(width_ratios=[1, 1.4, 1.4]))
    a = ax[0]
    a.plot(err_pav.index, 100 * err_pav.values, "-o", color=BLUE, lw=2, ms=5, label="paved controls read unpaved")
    a.plot(err_unp.index, 100 * err_unp.values, "-o", color=ORANGE, lw=2, ms=5, label="unpaved roads read asphalt")
    a.axvspan(0.5, 2.5, color="#eeece6", zorder=0)
    a.axvspan(5.5, 8.5, color="#eeece6", zorder=0)
    a.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    a.set_ylabel("per-reading error (%)")
    a.set_title("a  Monthly Sentinel-2: error by month", loc="left", fontsize=10)
    a.legend(fontsize=7, frameon=False, loc="upper left")
    b = ax[1]
    show = [r for r in ["hoima", "bukakata", "tirinyi", "muyembe"] if r in mshare]
    shades = ["#2a78d6", "#5b9be3", "#8cbcef", "#b9d6f6"]
    for k, rid in enumerate(show):
        sh = mshare[rid].sort_index()
        b.plot(sh.index, 100 * sh.values, color=shades[k], lw=2)
        b.text(sh.index[-1] + 0.1, 100 * sh.values[-1], rid.replace("_", " ").title(), fontsize=7, color=INK, va="center")
        hs = seg_main[(seg_main.id == rid) & seg_main.used]
        tt = sorted(rp.PT.values())
        b.step(tt, [100 * float((hs.t_half <= t + 1e-6).mean()) for t in tt], where="post", color=GREY, lw=1, ls="--")
    b.set_ylabel("km sealed (%)")
    b.set_xlim(2017, 2027.3)
    b.set_title("b  Monthly (blue) vs half-yearly (dashed) progress", loc="left", fontsize=10)
    c = ax[2]
    pre = [r for r in ["seeta", "fortkamwenge", "kamonkoli", "kigumba"] if r in lshare]
    for k, rid in enumerate(pre):
        sh = lshare[rid].sort_index()
        c.plot(sh.index, 100 * sh.values, color=shades[k], lw=2)
        c.text(sh.index[-1] + 0.1, 100 * sh.values[-1], rid.title(), fontsize=7, color=INK, va="center")
        md = roads.loc[rid].t_media_done
        if pd.notna(md):
            c.plot([md], [100 * float(np.interp(md, sh.index, sh.values))], "D", color=ORANGE, ms=6, mec="white")
    c.axvline(2017.08, color=GREY, lw=1, ls=":")
    c.text(2017.15, 3, "Sentinel-2 archive starts", fontsize=7, color=GREY)
    c.set_xlim(2013, 2027.6)
    c.set_ylabel("km sealed (%)")
    c.set_title("c  Landsat from 2013 (diamond: reported completion)", loc="left", fontsize=10)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
        x.grid(axis="y", color="#e5e3dc", lw=0.6)
    fig.tight_layout()
    f = ROOT / "posts" / "media" / "chart_dense_landsat.png"
    fig.savefig(f, dpi=200)
    shutil.copy(f, ROOT / "paper" / "figures" / "p3_f18_dense_landsat.png")


if __name__ == "__main__":
    main()
