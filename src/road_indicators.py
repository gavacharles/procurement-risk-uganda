"""Paper 3 / follow-on, step 1: segment-level progress indicators from the existing Sentinel-2 record.

Reads the per-segment sealing dates produced by src/road_progress.py (no new imagery) and derives, for each
contracted upgrade road with a satellite verdict:

  commencement lag   tender -> first segment sealed (years); tender -> activity onset is in road_progress
  sealing window     first -> last newly sealed segment (years)
  velocity           km newly sealed per year over the sealing window, and the peak over any 12 months
  spatial disparity  inter-quartile range of segment sealing dates (years); share of readable segments never
                     sealed; direction of works (Spearman rho of sealing date on chainage: |rho| near 1 =
                     one advancing front, near 0 = scattered fronts); number of contiguous work fronts in the
                     first sealing period
  inactivity         longest run of readings with no material new sealing (fewer than max(2, 5% of segments)
                     newly sealed, so a stray single-segment reading does not end a stall) between the first sealing and completion (>= 80%
                     sealed) or the last image; open spells at the last image are censored. Pre-sealing
                     inactivity (tender -> first works) is the commencement lag.

Limits: two readings a year (6-month resolution, so gaps are multiples of 0.5 y); "inactivity" means no
new *sealing*, and earthworks, drainage or base layers are invisible to this measure. Segments already sealed
in the first image (2017a) are pre-existing and excluded from the works-based indicators.

Out: data/processed/road_indicators.parquet, paper/road_indicators.json, chart p3_f16_indicators
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

DONE = 0.8
STEP = 0.5  # years between readings


def road_indicators(seg: pd.DataFrame, road: pd.Series) -> dict:
    s = seg[seg.used].sort_values("km_at")
    n = len(s)
    km_seg = (road.routed_km or road.contract_km or np.nan) / max(road.n_segs_route or n, 1)
    t = s.sealed.map(rp.PT)  # NaN = never sealed
    first_img = min(rp.PT.values())
    floor = max(first_img, road.t_tender - STEP) if pd.notna(road.t_tender) else first_img
    new = t[t > floor + 1e-6]  # sealed after the tender (6-month tolerance): this contract's works
    out = dict(id=road.id, n_segs=n, km_seg=km_seg, share_pre_sealed=float((t == first_img).mean()) if n else np.nan,
               share_never=float(t.isna().mean()) if n else np.nan, n_new=len(new))
    if len(new) < 3:
        return out
    t0 = new.min()
    # robust sealing window: 10th-90th percentile of sealing dates (stray single-segment seals are often noise)
    p10, p90 = np.percentile(new, [10, 90])
    p10, p90 = min(rp.PT.values(), key=lambda g: abs(g - p10)), min(rp.PT.values(), key=lambda g: abs(g - p90))
    core = new[(new >= p10 - 1e-6) & (new <= p90 + 1e-6)]
    t1 = p90
    out.update(t_first_seal=t0, t_last_seal=new.max(), window_y=p90 - p10,
               lag_tender_to_seal=t0 - road.t_tender if pd.notna(road.t_tender) else np.nan)
    # velocity: km newly sealed per year over the window (a single-period window counts as half a year)
    out["velocity_kmy"] = len(core) * km_seg / (p90 - p10 + STEP)  # inclusive: p10..p90 readings
    counts = new.round(2).value_counts().sort_index()
    grid = np.round(np.arange(t0, t1 + 1e-6, STEP), 2)
    per = counts.reindex(grid, fill_value=0)
    out["peak_kmy"] = float(per.rolling(2, min_periods=1).sum().max() * km_seg)  # 2 readings = 12 months
    # spatial disparity
    q1, q3 = np.percentile(new, [25, 75])
    out["iqr_seal_y"] = q3 - q1
    rho = spearmanr(s.loc[new.index, "km_at"], new).statistic
    out["front_rho"] = float(rho) if np.isfinite(rho) else np.nan
    first = (t.round(2) == round(t0, 2)).to_numpy()
    out["fronts_first"] = int(np.sum(first & ~np.r_[False, first[:-1]]))
    # inactivity after the first sealing: periods with no new sealed segment while < DONE sealed
    sealed_share = pd.Series({g: float((t <= g + 1e-6).mean()) for g in sorted(rp.PT.values()) if g >= t0})
    quiet = max(2, int(np.ceil(0.05 * n)))
    gaps, cur, start = [], 0, None
    for g, sh in sealed_share.items():
        if g == t0:
            continue
        if sh >= DONE:
            break
        if counts.get(round(g, 2), 0) < quiet:  # no material sealing (single stray segments are noise)
            cur += STEP
            start = start if start is not None else g
        else:
            if cur:
                gaps.append((cur, start, False))
            cur, start = 0, None
    if cur:
        # still stalled at the last image -> censored; a run ending at the p90 cutoff is closed
        gaps.append((cur, start, g >= max(rp.PT.values()) - 1e-6 and sh < DONE))
    longest = max(gaps, key=lambda x: x[0]) if gaps else (0.0, np.nan, False)
    out.update(longest_stall_y=longest[0], stall_start=longest[1], stall_censored=bool(longest[2]),
               n_stalls_ge1y=sum(g[0] >= 1 for g in gaps),
               reached_done=bool((sealed_share >= DONE).any()))
    return out


def main():
    S = pd.read_parquet(P / "road_progress_segments.parquet")
    R = pd.read_parquet(P / "road_progress_roads.parquet")
    R = R[(R.kind == "upgrade") & R.n_segs.notna() & ~R.status.str.startswith(("excluded", "insufficient", "sealed by the first"))]
    rows = [road_indicators(S[S.id == r.id], r) for r in R.itertuples(index=False)
            for r in [pd.Series(r._asdict())]]
    I = pd.DataFrame(rows).merge(R[["id", "road", "status", "t_tender", "t_onset", "t_done", "delay_years"]], on="id")
    I.to_parquet(P / "road_indicators.parquet", index=False)

    W = I[I.n_new >= 3]
    med = lambda c: None if W[c].dropna().empty else round(float(W[c].median()), 2)  # noqa: E731
    rng = lambda c: [round(float(W[c].min()), 2), round(float(W[c].max()), 2)]  # noqa: E731
    out = dict(
        n_roads=int(len(I)), n_with_works=int(len(W)),
        lag_tender_to_first_seal_y=dict(median=med("lag_tender_to_seal"), range=rng("lag_tender_to_seal")),
        window_y=dict(median=med("window_y"), range=rng("window_y")),
        velocity_kmy=dict(median=med("velocity_kmy"), range=rng("velocity_kmy")),
        peak_kmy=dict(median=med("peak_kmy"), range=rng("peak_kmy")),
        iqr_seal_y=dict(median=med("iqr_seal_y"), range=rng("iqr_seal_y")),
        front_rho_abs=dict(median=round(float(W.front_rho.abs().median()), 2),
                           n_single_front=int((W.front_rho.abs() >= 0.7).sum())),
        longest_stall_y=dict(median=med("longest_stall_y"), range=rng("longest_stall_y"),
                             n_ge_1y=int((W.longest_stall_y >= 1).sum()),
                             n_censored=int(W.stall_censored.sum())),
        corr_velocity_delay=None,
        roads=W[["road", "lag_tender_to_seal", "window_y", "velocity_kmy", "peak_kmy", "iqr_seal_y", "front_rho",
                 "longest_stall_y", "stall_censored", "share_never", "status"]].round(2).to_dict("records"),
        caveat="6-month resolution; inactivity = no new sealing (earthworks invisible); a stall is a prompt to "
               "check, not a finding of contractor delay.",
    )
    d = W.dropna(subset=["velocity_kmy", "delay_years"])
    if len(d) >= 5:
        r = spearmanr(d.velocity_kmy, d.delay_years)
        out["corr_velocity_delay"] = dict(rho=round(float(r.statistic), 2), p=round(float(r.pvalue), 3), n=len(d))
    (ROOT / "paper" / "road_indicators.json").write_text(json.dumps(out, indent=2, default=float))
    chart(S, W)
    print(json.dumps({k: v for k, v in out.items() if k != "roads"}, indent=2))
    print(W[["id", "lag_tender_to_seal", "window_y", "velocity_kmy", "peak_kmy", "iqr_seal_y", "front_rho",
             "fronts_first", "longest_stall_y", "stall_censored", "share_never"]].round(2).to_string())


def chart(S, W):
    """(a) segment sealing dates by chainage for each road (strip); (b) velocity vs longest stall."""
    W = W.sort_values("t_first_seal")
    fig, ax = plt.subplots(1, 2, figsize=(10, 0.35 * len(W) + 1.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    a = ax[0]
    for k, r in enumerate(W.itertuples()):
        s = S[(S.id == r.id) & S.used].sort_values("km_at")
        t = s.sealed.map(rp.PT)
        x = (s.km_at / s.km_at.max()).to_numpy()
        a.scatter(x[t.notna()], np.full(t.notna().sum(), k), c=t[t.notna()], cmap="viridis", vmin=2017, vmax=2026.6,
                  s=14, marker="s")
        a.scatter(x[t.isna()], np.full(t.isna().sum(), k), color="#d9d6cf", s=14, marker="s")
    a.set_yticks(range(len(W)), W.road.str.slice(0, 32))
    a.set_xlabel("position along road (0 = start, 1 = end); grey = never sealed")
    a.invert_yaxis()
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(2017, 2026.6))
    fig.colorbar(sm, ax=a, fraction=0.03, pad=0.01, label="year sealed")
    a.set_title("a  When each kilometre was sealed", loc="left")
    b = ax[1]
    b.scatter(W.velocity_kmy, W.longest_stall_y, color=np.where(W.stall_censored, "#eb6834", "#2a78d6"), s=28)
    for r in W.itertuples():
        b.annotate(r.road.split("–")[0][:14], (r.velocity_kmy, r.longest_stall_y), fontsize=6.5,
                   xytext=(3, 2), textcoords="offset points")
    b.set_xlabel("sealing velocity (km per year)\norange = still stalled at last image")
    b.set_ylabel("longest stall after sealing began (years)")
    b.set_title("b  Pace and pauses", loc="left")
    fig.tight_layout()
    f = ROOT / "posts" / "media" / "chart_road_indicators.png"
    fig.savefig(f, dpi=200)
    shutil.copy(f, ROOT / "paper" / "figures" / "p3_f16_indicators.png")


if __name__ == "__main__":
    main()
