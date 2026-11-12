"""Charts for src/road_progress.py (reads the saved parquet files; no downloads).

  chart_road_progress.png         sealed share over time, one panel per road
  chart_road_progress_strips.png  space-time strips: km along the road x period, coloured by state
  chart_road_progress_timeline.png  tender -> expected -> observed completion, per road
  chart_road_progress_km.png      Kaplan-Meier: years from tender to satellite-observed completion
  chart_road_progress_risk.png    delay vs procurement risk score (exploratory)
"""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from visuals import BLUE, ORANGE, INK, INK2, title  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "posts" / "media"
ASPH, UNP, VEG, NOD = "#4a4f57", "#d9873a", "#7fb77e", "#f1f1ef"
STATUS_COL = {"completed": BLUE, "substantial": BLUE, "incomplete": ORANGE, "nopaving": "#c0392b",
              "progress": "#8a8f98", "control": "#8a8f98", "sealed": "#8a8f98", "insufficient": "#c8c8c8"}


def save(fig, name, paper=None):
    f = OUT / name
    fig.savefig(f, dpi=200)
    shutil.copy(f, ROOT / "docs" / "media" / name)
    if paper:
        shutil.copy(f, ROOT / "paper" / "figures" / paper)
    plt.close(fig)
    print("wrote", f)


def status_key(s):
    for pre, k in [("completed", "completed"), ("substantially", "substantial"), ("incomplete", "incomplete"),
                   ("no paving detected past", "nopaving"), ("control", "control"), ("rehabilitation", "control"),
                   ("sealed", "sealed"), ("insufficient", "insufficient")]:
        if s.startswith(pre):
            return k
    return "progress"


def load():
    R = pd.read_parquet(P / "road_progress_roads.parquet")
    S = pd.read_parquet(P / "road_progress_series.parquet")
    G = pd.read_parquet(P / "road_progress_segments.parquet")
    R = R[~R.status.str.startswith("excluded")].copy()
    R["ctrl"] = R.kind.str.startswith("control")
    R = R.sort_values(["ctrl", "t_tender"], na_position="last")
    return R, S, G


def fy(t):
    return f"{int(t)}"


# ---------------------------------------------------------------- 1. curves
def chart_curves(R, S):
    n, cols = len(R), 6
    rows = math.ceil(n / cols)
    H = 1.75 * rows + 1.6
    fig = plt.figure(figsize=(12, H))
    title(fig, "How far along is each road? Share of the route sealed, 2017–2026",
          "Sentinel-2, two readings a year. Line = share of 1 km segments sealed (stays sealed once paved). Dots = raw asphalt share that period.\n"
          "Dashed = tender invitation. Yellow = expected completion from news reports or contract records; grey = benchmark window\n"
          "(signing 3–12 months after tender + planned build time) where no date is reported.",
          source="Imagery: Copernicus Sentinel-2 · contracts: PPDA open contracting data")
    for t in fig.texts:
        x, y = t.get_position()
        if y > 0.5:
            t.set_y(1 - (0.3 if y > 0.9 else 0.7) / H)
        else:
            t.set_y(0.2 / H)
    top, bot = 1 - 1.45 / H, 0.55 / H
    ph = (top - bot) / rows
    for i, r in enumerate(R.itertuples()):
        s = S[S.id == r.id].sort_values("t")
        ax = fig.add_axes([0.045 + (i % cols) * 0.159, top - (i // cols + 1) * ph + 0.32 / H, 0.13, ph - 0.62 / H])
        if not np.isnan(r.t_tender):
            pm = 18 if r.contract_km <= 15 else 24 if r.contract_km <= 60 else 36
            if not r.ctrl and pd.notna(r.expected_done):
                if str(r.expected_source).startswith("benchmark"):
                    ax.axvspan(r.t_tender + 0.25 + pm / 12, r.t_tender + 1.0 + pm / 12, color="#9aa0a6", alpha=0.22, lw=0)
                else:  # contract record or news report: a date, not a window
                    ax.axvspan(r.expected_done - 0.12, r.expected_done + 0.12, color="#f2c14e", alpha=0.55, lw=0)
            ax.axvline(r.t_tender, color=INK2, lw=0.9, ls="--")
        ax.scatter(s.t, s.raw_asphalt_share, s=5, color="#b9bcc2", zorder=2)
        col = STATUS_COL[status_key(r.status)]
        ax.plot(s.t, s.raw_asphalt_share if r.ctrl else s.sealed_share, color=col, lw=2, zorder=3)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(2016.8, 2027)
        ax.set_xticks([2018, 2022, 2026])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6.5)
        if i % cols:
            ax.set_yticklabels([])
        ax.grid(axis="x", visible=False)
        name = r.road.replace(" (control)", "")
        ax.set_title(name if len(name) <= 26 else name[:25] + "…", fontsize=7.5, loc="left", color=INK, pad=11)
        k = status_key(r.status)
        tag = (("rehab: stripped, not restored" if "under way" in r.status else "rehab: stripped & restored"
                if "restored" in r.status else "control: paved") if r.ctrl else
               "too few readable segments" if k == "insufficient" else
               "complete " + fy(r.t_done) if k == "completed" else
               f"done ~{fy(r.t_plateau)}, {r.s_last:.0%} sealed" if k == "substantial" else
               ("sealed by 2017" if "first images" in r.status else "sealed before tender") if k == "sealed" else
               "no paving detected" if k == "nopaving" else
               f"overdue, {r.s_last:.0%} sealed" if k == "incomplete" else
               f"not yet due, {r.s_last:.0%}")
        ax.text(0, 1.02, tag, transform=ax.transAxes, fontsize=6.5, color=col if not r.ctrl else INK2, va="bottom")
    save(fig, "chart_road_progress.png", "p3_f2_progress_curves.png")


# ---------------------------------------------------------------- 2. strips
def chart_strips(R, G, ids=None):
    order = [c for c in G.columns if len(c) == 5 and c[:4].isdigit()]
    ids = ids or [i for i in ["hoima", "koboko", "namagumba", "kyenjojo", "muyembe", "luwero"] if i in set(R.id)]
    n = len(ids)
    H = 1.9 * n + 1.5
    fig = plt.figure(figsize=(10, H))
    title(fig, "Where and when each kilometre was sealed",
          "Each row = one ~1 km segment along the road (start at bottom); each column = one Sentinel-2 reading (Jan and Jul).\n"
          "Grey = asphalt, orange = unpaved, green = vegetation over the road, blank = cloud. Dashed = tender invitation.",
          source="Imagery: Copernicus Sentinel-2 · contracts: PPDA")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    cmap = ListedColormap([NOD, UNP, ASPH, VEG])
    code = {"nodata": 0, "unpaved": 1, "asphalt": 2, "vegetated": 3}
    for i, rid in enumerate(ids):
        g = G[G.id == rid].sort_values("seg")
        r = R[R.id == rid].iloc[0]
        M = np.array([[code.get(v, 0) for v in g[c]] for c in order]).T
        ax = fig.add_axes([0.2, 1 - (1.35 + (i + 1) * 1.9) / H + 0.3 / H, 0.74, 1.45 / H])
        ax.imshow(M, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, origin="lower", interpolation="nearest",
                  extent=[-0.5, len(order) - 0.5, 0, g.km_at.max() + 0.5])
        xt = [k for k, c in enumerate(order) if c.endswith("a") and int(c[:4]) % 2 == 0]
        ax.set_xticks(xt)
        ax.set_xticklabels([order[k][:4] for k in xt], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("km", fontsize=7, color=INK2)
        ax.grid(False)
        if not np.isnan(r.t_tender):
            pos = np.interp(r.t_tender, [2017.08 + 0.5 * k for k in range(len(order))], range(len(order)))
            if r.t_tender >= 2017:
                ax.axvline(pos, color=INK, lw=1, ls="--")
        lab = r.road.replace(" (control)", "")
        fig.text(0.03, 1 - (1.35 + i * 1.9 + 0.3) / H, lab.replace("–", "–\n", 1) if len(lab) > 20 else lab,
                 fontsize=9, fontweight="bold", color=INK, va="top")
        short = {"completed": "completed", "substantial": "substantially complete", "incomplete": "overdue, in progress",
                 "nopaving": "no paving detected", "progress": "not yet due", "control": "control", "sealed": "sealed before tender"}
        fig.text(0.03, 1 - (1.35 + i * 1.9 + 0.95) / H, f"tender {r.tender_date[:7] if r.tender_date else 'n/a'}\n{short[status_key(r.status)]}",
                 fontsize=7.5, color=INK2, va="top")
    save(fig, "chart_road_progress_strips.png", "p3_f3_space_time_strips.png")


# ---------------------------------------------------------------- 3. timeline
def chart_timeline(R):
    D = R[~R.ctrl & R.t_tender.notna() & ~R.status.str.startswith("insufficient")].sort_values("t_tender")
    n = len(D)
    H = 0.32 * n + 2.3
    fig = plt.figure(figsize=(10, H))
    title(fig, "Tender to tarmac: expected vs satellite-observed completion",
          "● tender invitation   ▲ works start (media launch, else satellite activity)   ★ completion reported in the media\n"
          "◇ expected completion (media deadline or launch + duration where reported, else tender benchmark)\n"
          "Bar = to satellite completion (80% sealed, or levelled off); arrow = not complete by Jul 2026. * tender = portal entry date.",
          source="Imagery: Copernicus Sentinel-2 · contracts: PPDA")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.27, 0.75 / H, 0.68, 1 - 2.45 / H])
    for k, r in enumerate(D.itertuples()):
        y = n - 1 - k
        sk = status_key(r.status)
        col = STATUS_COL[sk]
        if sk == "sealed":
            ax.barh(y, 2017.08 - r.t_tender, left=r.t_tender, height=0.5, color="white", edgecolor="#9aa0a6", hatch="////", lw=0.6)
        elif pd.notna(r.t_complete):
            ax.barh(y, r.t_complete - r.t_tender, left=r.t_tender, height=0.5, color=col, alpha=0.85)
        else:
            ax.barh(y, r.t_last - r.t_tender, left=r.t_tender, height=0.5, color=col, alpha=0.85)
            ax.annotate("", xy=(r.t_last + 0.45, y), xytext=(r.t_last, y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
        ax.plot(r.t_tender, y, "o", color=INK, ms=4)
        if pd.notna(r.t_start) and r.t_start > r.t_tender:
            ax.plot(r.t_start, y, "^", color="white", mec=INK, ms=5.5, mew=1.2, zorder=4)
        if pd.notna(r.t_media_done):
            ax.plot(r.t_media_done, y, "*", color="#f2c14e", mec=INK, ms=8, mew=0.6, zorder=5)
        if not np.isnan(r.expected_done):
            ax.plot(r.expected_done, y, "D", color=INK, ms=4.5, mfc="white", mew=1.2)
        star = "*" if r.tender_date_source and r.tender_date_source.startswith("portal") else ""
        ax.text(2013.6, y, r.road + star, fontsize=7.8, color=INK, va="center", ha="right")
        lbl = (f"{r.delay_years:+.1f} y" if not np.isnan(r.delay_years) and not r.delay_censored else
               f"> {r.delay_years:.1f} y late" if not np.isnan(r.delay_years) else "")
        if lbl:
            ax.text(max(r.t_complete if pd.notna(r.t_complete) else r.t_last, r.expected_done) + 0.55, y, lbl, fontsize=7, color=col, va="center")
    ax.axvline(2017.08, color="#c8c8c8", lw=0.8)
    ax.text(2017.12, n - 0.3, "first image", fontsize=6.5, color=INK2)
    ax.set_xlim(2014, 2029.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", visible=False)
    save(fig, "chart_road_progress_timeline.png", "p3_f4_timeline.png")


# ---------------------------------------------------------------- 4. Kaplan-Meier
def km_curve(t, event):
    order = np.argsort(t)
    t, event = np.asarray(t)[order], np.asarray(event)[order]
    S, xs, ys = 1.0, [0], [1.0]
    at_risk = len(t)
    for ti in np.unique(t):
        d = ((t == ti) & event).sum()
        c = (t == ti).sum()
        if d:
            S *= 1 - d / at_risk
            xs.append(ti)
            ys.append(S)
        at_risk -= c
    return np.array(xs), np.array(ys)


def chart_km(R):
    D = R[~R.ctrl & R.t_tender.notna() & ~R.status.str.startswith(("sealed", "insufficient"))].copy()
    D["dur"] = np.where(D.t_complete.notna(), D.t_complete - D.t_tender, D.t_last - D.t_tender)
    D["event"] = D.t_complete.notna()
    D = D[D.dur > 0]
    xs, ys = km_curve(D.dur.to_numpy(), D.event.to_numpy())
    at_risk = np.array([(D.dur >= x).sum() for x in xs])
    keep = at_risk >= 3
    xs, ys = xs[keep], ys[keep]
    H = 5.0
    fig = plt.figure(figsize=(8, H))
    title(fig, "How long from tender to a sealed road?",
          f"Kaplan–Meier estimate, {len(D)} contracted roads tendered 2014–2025.\n"
          "Ticks = not complete by Jul 2026 (censored). Curve stops when < 3 roads remain under observation.",
          source="Imagery: Copernicus Sentinel-2 · contracts: PPDA")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.1, 0.95 / H, 0.85, 1 - 2.1 / H])
    ax.step(np.append(xs, D.dur[D.dur >= xs[-1]].min() if len(xs) else 0), np.append(1 - ys, (1 - ys)[-1]), where="post", color=BLUE, lw=2.2)
    cens = D[~D.event]
    for r in cens.itertuples():
        yv = 1 - ys[np.searchsorted(xs, r.dur, side="right") - 1]
        ax.plot(r.dur, yv, "|", color=ORANGE, ms=10, mew=1.6)
    for yrs, lab in [(2.5, "18-month build + 1 y procurement"), (4.0, "36-month build + 1 y procurement")]:
        ax.axvline(yrs, color=INK2, lw=0.8, ls="--")
        ax.text(yrs + 0.08, 0.95, lab, fontsize=7, color=INK2, va="top", rotation=90)
    ax.set_xlabel("Years since tender invitation", fontsize=9)
    ax.set_ylabel("Share of roads complete", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(10, D.dur.max() + 0.5))
    ax.tick_params(labelsize=8)
    save(fig, "chart_road_progress_km.png", "p3_f5_time_to_completion.png")
    med = xs[np.argmax(1 - ys >= 0.5)] if (1 - ys >= 0.5).any() else None
    return dict(n=int(len(D)), events=int(D.event.sum()), median_years=float(med) if med else None,
                share_done_by_4y=float(1 - ys[np.searchsorted(xs, 4.0, side="right") - 1]),
                share_done_by_6y=float(1 - ys[np.searchsorted(xs, 6.0, side="right") - 1]))


# ---------------------------------------------------------------- 5. delay vs risk
def chart_risk(R):
    D = R[~R.ctrl & R.risk_score.notna() & R.delay_years.notna()].copy()
    H = 5.0
    fig = plt.figure(figsize=(8, H))
    title(fig, "Are flagged contracts delivered later? (exploratory)",
          f"{len(D)} roads with a procurement risk score and a satellite delay estimate.\n"
          "Hollow = still incomplete (delay is a lower bound). Small sample: descriptive only.",
          source="Imagery: Copernicus Sentinel-2 · contracts: PPDA")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.1, 0.95 / H, 0.85, 1 - 2.1 / H])
    seen = {}
    for r in D.itertuples():
        ax.plot(r.risk_score, r.delay_years, "o", ms=7, color=ORANGE if r.delay_censored else BLUE,
                mfc="white" if r.delay_censored else BLUE, mew=1.5)
        k = (round(r.risk_score), round(r.delay_years * 2))
        seen[k] = seen.get(k, 0) + 1
        ax.annotate(r.road.split("–")[0].replace(" (control)", ""), (r.risk_score, r.delay_years), fontsize=6.5,
                    color=INK2, xytext=(6, 3 - 9 * (seen[k] - 1)), textcoords="offset points")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Procurement risk score (higher = more red flags)", fontsize=9)
    ax.set_ylabel("Years late vs expected completion", fontsize=9)
    ax.tick_params(labelsize=8)
    save(fig, "chart_road_progress_risk.png", "p3_f6_delay_vs_risk.png")
    from scipy.stats import spearmanr
    rho, p = spearmanr(D.risk_score, D.delay_years) if len(D) > 4 else (np.nan, np.nan)
    return dict(n=int(len(D)), spearman_rho=float(rho), p=float(p))


# ---------------------------------------------------------------- 6. media validation
def chart_media(R, S):
    import json
    rows = []
    for r in R.itertuples():
        if r.kind.startswith("control") or r.status.startswith("insufficient"):
            continue  # reconstruction (sealed share not comparable) / no satellite verdict
        for m in json.loads(r.media_progress or "[]"):
            if m.get("sat_sealed") is not None:
                rows.append(dict(road=r.road, kind=r.kind, **m))
    M = pd.DataFrame(rows)
    C = R[R.t_media_done.notna() & R.t_complete.notna()]
    H = 5.2
    fig = plt.figure(figsize=(10, H))
    title(fig, "Does the satellite agree with what was reported?",
          "Left: physical progress reported in the media vs share of the route sealed at the nearest earlier Sentinel-2 reading.\n"
          "Right: completion or commissioning date reported in the media vs satellite completion (hollow = upper bound).",
          source="Media reports listed in data/manual/media_reports.csv · Copernicus Sentinel-2")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.07, 0.95 / H, 0.4, 1 - 2.15 / H])
    if len(M):
        ax.plot([0, 100], [0, 1], color="#c8c8c8", lw=1, ls="--")
        for r in M.itertuples():
            c = "#8a8f98" if r.kind.startswith("control") else BLUE
            ax.plot(r.media_pct, r.sat_sealed, "o", color=c, ms=7)
            ax.annotate(f"{r.road.split('–')[0]} {r.date[:7]}", (r.media_pct, r.sat_sealed), fontsize=6.5, color=INK2,
                        xytext=(5, 2), textcoords="offset points")
    ax.set_xlabel("Reported physical progress (%)", fontsize=8.5)
    ax.set_ylabel("Satellite: share of route sealed", fontsize=8.5)
    ax.set_xlim(-3, 105)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=7.5)
    ax.text(60, 0.08, "physical progress counts earthworks,\ndrainage and base before any sealing", fontsize=6.5, color=INK2)
    ax2 = fig.add_axes([0.57, 0.95 / H, 0.4, 1 - 2.15 / H])
    if len(C):
        lo, hi = 2016.5, 2026.8
        ax2.plot([lo, hi], [lo, hi], color="#c8c8c8", lw=1, ls="--")
        for r in C.itertuples():
            ub = bool(r.early_sealed)
            ax2.plot(r.t_media_done, r.t_complete, "o", color=BLUE, ms=7, mfc="white" if ub else BLUE, mew=1.5)
            ax2.annotate(r.road.split("–")[0], (r.t_media_done, r.t_complete), fontsize=7, color=INK2,
                         xytext=(5, -3), textcoords="offset points")
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
    ax2.set_xlabel("Completion reported in the media", fontsize=8.5)
    ax2.set_ylabel("Satellite completion", fontsize=8.5)
    ax2.tick_params(labelsize=7.5)
    save(fig, "chart_road_progress_media.png", "p3_f7_media_validation.png")
    diff = (C.t_complete - C.t_media_done)[~C.early_sealed.astype(bool)]
    return dict(n_progress_reports=int(len(M)), n_completion_pairs=int(len(diff)),
                completion_lag_years_median=float(diff.median()) if len(C) else None,
                completion_abs_error_years_mean=float(diff.abs().mean()) if len(C) else None)



# ---------------------------------------------------------------- 7. map
def chart_map(R):
    import json
    import road_progress as rp
    geo = json.loads((ROOT / "data" / "raw" / "geo" / "uga_districts_ubos2020.geojson").read_text())
    H = 8.2
    fig = plt.figure(figsize=(9, H))
    title(fig, "Where the contracted roads are, and what the satellite sees",
          "Status at July 2026 from Sentinel-2, checked against news reports. Line width ~ contract length.",
          source="UBOS · OpenStreetMap · Copernicus Sentinel-2")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.03, 0.5 / H, 0.62, 1 - 1.5 / H])
    for f in geo["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            xy = np.array(poly[0])
            ax.fill(xy[:, 0], xy[:, 1], color="#efeeea", ec="#d6d4cf", lw=0.4, zorder=1)
    cols = {"completed": BLUE, "substantial": "#8fb8ec", "sealed": "#1f3d73", "incomplete": ORANGE,
            "nopaving": "#c0392b", "progress": "#b9bcc2", "control": "#5f646c", "insufficient": "#dcdcdc"}
    styles = {"control": (0, (2, 1.2))}
    near_kla = []
    labels = {"completed": "completed", "substantial": "substantially complete", "sealed": "sealed by 2017",
              "incomplete": "overdue, still being sealed", "nopaving": "no paving past expected completion",
              "progress": "not yet due", "control": "rehabilitation of paved road", "insufficient": "no verdict"}
    seen = set()
    for r in R.itertuples():
        road = next((x for x in rp.ROADS if x[0] == r.id), None)
        g = rp.geometry(road) if road else None
        if not g:
            continue
        xy = np.array(g["coords"])
        k = status_key(r.status)
        ax.plot(xy[:, 0], xy[:, 1], color=cols[k], lw=1.4 + min(r.contract_km, 140) / 40, solid_capstyle="round",
                ls=styles.get(k, "-"), zorder=3)
        seen.add(k)
        nm = r.road.split(" (")[0]
        parts = nm.split("–")
        short = parts[0] if len(parts) < 2 else f"{parts[0]}–{parts[-1]}"
        if r.contract_km < 12 and abs(xy[:, 0].mean() - 32.58) < 0.3 and abs(xy[:, 1].mean() - 0.32) < 0.3:
            near_kla.append(short)
        else:
            dx, dy, ha = {"tirinyi": (-0.08, 0.25, "right"), "kamonkoli": (0.0, -0.13, "left"),
                          "namagumba": (0.1, 0.06, "left"), "bumbobi": (0.08, -0.08, "left"),
                          "fortkamwenge": (0.06, -0.08, "left")}.get(r.id, (0.06, 0.0, "left"))
            ax.text(xy[len(xy) // 2, 0] + dx, xy[len(xy) // 2, 1] + dy, short, fontsize=6.3, color=INK2, zorder=4, ha=ha)
    ax.set_aspect(1 / np.cos(np.radians(1.3)))
    ax.set_xlim(29.5, 35.1)
    ax.set_ylim(-1.5, 4.3)
    ax.axis("off")
    order = ["completed", "substantial", "sealed", "incomplete", "nopaving", "progress", "control", "insufficient"]
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], color=cols[k], lw=3.5, ls=styles.get(k, "-")) for k in order if k in seen]
    if near_kla:
        fig.text(0.70, 0.30, "Short roads around Kampala:\n" + "\n".join(sorted(near_kla)), fontsize=7.5, color=INK2, va="top")
    fig.legend(hs, [labels[k] for k in order if k in seen],
               loc="upper left", bbox_to_anchor=(0.66, 1 - 1.4 / H), frameon=False, fontsize=8.5)
    save(fig, "chart_road_progress_map.png", "p3_f1_map.png")


# ---------------------------------------------------------------- 8. delay decomposition
def chart_decomp(R):
    D = R[~R.ctrl & R.t_start.notna() & R.t_tender.notna() & ~R.status.str.startswith(("insufficient", "sealed"))].copy()
    D = D[D.t_start > D.t_tender - 0.3].sort_values("t_tender")
    n = len(D)
    H = 0.42 * n + 2.0
    fig = plt.figure(figsize=(9, H))
    title(fig, "Where does the time go? Before works vs construction",
          "Grey = tender invitation to works start (media launch date; else the satellite earthworks signal, marked °).\n"
          "Blue = works start to satellite completion; orange = works start to Jul 2026 for roads not yet complete.",
          source="PPDA tender dates · news reports · Copernicus Sentinel-2")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.3, 0.7 / H, 0.62, 1 - 2.05 / H])
    for k, r in enumerate(D.itertuples()):
        y = n - 1 - k
        pre = r.t_start - r.t_tender
        ax.barh(y, pre, left=0, color="#b9bcc2", height=0.55)
        if pd.notna(r.t_complete):
            ax.barh(y, r.t_complete - r.t_start, left=pre, color=BLUE, height=0.55)
            end = pre + r.t_complete - r.t_start
        else:
            ax.barh(y, r.t_last - r.t_start, left=pre, color=ORANGE, height=0.55)
            end = pre + r.t_last - r.t_start
        src = "" if pd.notna(r.t_launch) else "°"
        ax.text(-0.15, y, r.road + src, ha="right", va="center", fontsize=8, color=INK)
        ax.text(end + 0.1, y, f"{pre:.1f} + {end - pre:.1f} y", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks([])
    ax.set_xlim(0, max(10, D.t_last.max() - D.t_tender.min()) * 0.0 + 11)
    ax.set_xlabel("Years since tender invitation", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", visible=False)
    save(fig, "chart_road_progress_decomposition.png", "p3_f8_decomposition.png")


# ---------------------------------------------------------------- 9. sensitivity heatmap
def chart_sensitivity():
    import json
    f = ROOT / "paper" / "road_progress_sensitivity.json"
    if not f.exists():
        return
    S = pd.DataFrame(json.loads(f.read_text()))
    S = S[S.bench_shift == 0]
    H = 4.3
    fig = plt.figure(figsize=(9, H))
    title(fig, "How sensitive are the verdicts to the thresholds?",
          "Rows: share of later readings that must be asphalt for a kilometre to count as sealed.\n"
          "Columns: share of kilometres sealed for a road to count as complete. Central setting: 70% / 80%.",
          source="paper/road_progress_sensitivity.json")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    for j, (col, lab) in enumerate([("completed", "Roads completed"), ("no_paving_past_due", "No paving past due"),
                                    ("overdue_in_progress", "Overdue, in progress")]):
        ax = fig.add_axes([0.08 + j * 0.31, 0.65 / H, 0.25, 1 - 2.05 / H])
        M = S.pivot(index="seal_share", columns="done_th", values=col)
        ax.imshow(M.values, cmap="Blues", vmin=0, vmax=max(10, M.values.max()))
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, int(M.values[a, b]), ha="center", va="center", fontsize=11,
                        color="white" if M.values[a, b] > 6 else INK)
        ax.set_xticks(range(M.shape[1]), [f"{c:.0%}" for c in M.columns], fontsize=7.5)
        ax.set_yticks(range(M.shape[0]), [f"{c:.0%}" for c in M.index], fontsize=7.5)
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
        ax.grid(False)
    save(fig, "chart_road_progress_sensitivity.png", "p3_f9_sensitivity.png")


# ---------------------------------------------------------------- 10. crops (replaces the pilot crops)
CROP_ROADS = ["hoima", "bumbobi", "muyembe", "luwero", "koboko", "namagumba", "kalule", "mutukula", "matugga"]


def chart_crops(R, G):
    import road_progress as rp
    import road_progress_labelkit as lk
    import satellite_crops as cr
    lk.HALF = 600
    order = [c for c in G.columns if len(c) == 5 and c[:4].isdigit()]
    rows = []
    for rid in CROP_ROADS:
        if rid not in set(R.id):
            continue
        r = R[R.id == rid].iloc[0]
        g = G[(G.id == rid) & G.used.astype(bool)].copy()
        road = next(x for x in rp.ROADS if x[0] == rid)
        segs = rp.segments(rp.geometry(road)["coords"], road[3]).set_index("seg")
        sealed = g[g.sealed.isin(order)]
        if pd.notna(r.t_tender):
            sealed = sealed[sealed.sealed.map(lambda p: int(p[:4]) + (0.08 if p.endswith("a") else 0.58)) >= r.t_tender]
        latest = "2026a"
        if r.kind.startswith("control"):
            seg = g.iloc[len(g) // 2]
            k_ = min(len(order) - 2, max(0, int(round((r.t_min_raw - 2017.08) * 2)))) if pd.notna(r.t_min_raw) else order.index("2024b")
            mid, latest = order[k_], "2026b"
            if mid >= latest:
                mid = order[order.index(latest) - 2]
        elif len(sealed):
            sealed = sealed.assign(k=sealed.sealed.map(order.index)).sort_values("k")
            seg = sealed.iloc[len(sealed) // 2]
            mid = seg.sealed
            if mid >= latest:
                latest = "2026b"
        else:
            seg = g.iloc[len(g) // 2]
            tt = r.t_tender if pd.notna(r.t_tender) else 2020
            mid = order[min(len(order) - 3, int(round((tt + 2026.08) / 2 - 2017.08) * 2))]
        t0 = max(2017.08, r.t_tender - 0.6) if pd.notna(r.t_tender) else 2017.08
        first = next(p for p in order if 2017 + (0.08 if p.endswith("a") else 0.58) + int(p[:4]) - 2017 >= t0 - 0.01)
        trip = [first, mid, latest]
        imgs = []
        for per in trip:
            s_ = segs.loc[int(seg.seg)]
            img, sid = None, None
            for alt in [per] + [order[min(len(order) - 1, order.index(per) + d)] for d in (1, -1, 2, -2)]:
                img, sid = lk.crop(s_.lon, s_.lat, alt)
                if img is not None:
                    per = alt
                    break
            imgs.append((img, per))
        if all(i[0] is not None for i in imgs):
            rows.append(dict(r=r, imgs=imgs, km=seg.km_at))
            print("crops", rid, [p for _, p in imgs], flush=True)
    n = len(rows)
    row_h, top, bot = 1.95, 1.45, 0.45
    H = top + n * row_h + bot
    fig = plt.figure(figsize=(9, H))
    title(fig, "See for yourself: before, during and after",
          "Sentinel-2 true colour, 1.2 km across, around one kilometre of each road. 'During' = when that kilometre\n"
          "sealed (rehabilitations: when the surface was stripped). Gravel looks orange or tan; asphalt grey.",
          source="Imagery: Copernicus Sentinel-2 (ESA) via Earth Search")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.72) / H) if y > 0.5 else t.set_y(0.18 / H)
    img_w = 0.2
    for i, row in enumerate(rows):
        r = row["r"]
        y0 = 1 - (top + (i + 1) * row_h) / H
        h = (row_h - 0.25) / H
        name = r.road.replace(" (control)", "")
        if len(name) > 22:
            name = name.replace("–", "–\n", 1)
        fig.text(0.04, y0 + h - 0.02 / H, name, fontsize=10.5, fontweight="bold", color=INK, va="top")
        k = status_key(r.status)
        stat = {"completed": "completed", "substantial": "substantially complete", "incomplete": "overdue, still being sealed",
                "nopaving": "no paving past expected date", "progress": "not yet due",
                "control": "rehabilitation" + (": stripped, not restored" if "under way" in r.status else ": restored" if "restored" in r.status else "")}.get(k, k)
        fig.text(0.04, y0 + h - 0.62 / H, f"tender {r.tender_date[:7] if r.tender_date else 'n/a'}\n{stat}\nkm {row['km']:.0f} of the route",
                 fontsize=8.3, color=STATUS_COL.get(k, INK2) if k != "control" else INK2, va="top", linespacing=1.45)
        for j, (img, per) in enumerate(row["imgs"]):
            ax = fig.add_axes([0.335 + j * (img_w + 0.018), y0, img_w, h])
            ax.imshow(cr.stretch([img])[0], interpolation="lanczos")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            lab = ["before", "during", "latest"][j]
            mon = "Jan" if per.endswith("a") else "Jul"
            ax.set_title(f"{lab} · {mon} {per[:4]}", fontsize=8, color=INK2, loc="left", pad=3)
    save(fig, "chart_road_progress_crops.png", "p3_f3b_crops.png")

# ---------------------------------------------------------------- 11. validation summary (labels + OSM)
def chart_validation():
    import json
    fl, fo = ROOT / "paper" / "road_progress_labels_score.json", ROOT / "paper" / "road_progress_osmcheck.json"
    if not (fl.exists() and fo.exists()):
        return
    L, O = json.loads(fl.read_text()), json.loads(fo.read_text())
    H = 5.0
    fig = plt.figure(figsize=(9, H))
    title(fig, "Two independent checks of the classifier",
          "Left: 160 random kilometre-dates judged by eye from true-colour crops, blind to the\n"
          "classifier (40 could not be told). Right: satellite state in Jul 2026 vs OpenStreetMap tags\n"
          "(ways not used in training); 30 of 84 disagreements are two roads paved 2023–25 still tagged gravel.",
          source="paper/road_progress_labels_score.json · paper/road_progress_osmcheck.json")
    for tt in fig.texts:
        x, y = tt.get_position()
        tt.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else tt.set_y(0.2 / H)
    c = L["confusion"]
    m1 = np.array([[c["visual_asphalt_pred_asphalt"], c["visual_asphalt_pred_unpaved"]],
                   [c["visual_unpaved_pred_asphalt"], c["visual_unpaved_pred_unpaved"]]])
    o = O["confusion"]
    m2 = np.array([[o["osm_sealed_sat_sealed"], o["osm_sealed_sat_unpaved"]],
                   [o["osm_unpaved_sat_sealed"], o["osm_unpaved_sat_unpaved"]]])
    for j, (m, rows, cols, lab) in enumerate([
            (m1, ["seen: asphalt", "seen: unpaved"], ["classifier:\nasphalt", "classifier:\nunpaved"],
             f"Visual check: {L['coder1']['accuracy']:.0%} agree (n={L['coder1']['n']})"),
            (m2, ["OSM: sealed", "OSM: unpaved"], ["satellite:\nsealed", "satellite:\nunpaved"],
             f"OpenStreetMap: {O['agreement']:.0%} agree (n={O['n_segments']} km, {O['n_roads']} roads)")]):
        ax = fig.add_axes([0.14 + j * 0.47, 0.85 / H, 0.3, 1 - 2.45 / H])
        ax.imshow(m / m.sum(), cmap="Blues", vmin=0, vmax=0.7)
        for a in range(2):
            for b in range(2):
                ax.text(b, a, int(m[a, b]), ha="center", va="center", fontsize=13, color="white" if m[a, b] / m.sum() > 0.35 else INK)
        ax.set_xticks([0, 1], cols, fontsize=8)
        ax.set_yticks([0, 1], rows, fontsize=8)
        ax.set_title(lab, fontsize=9, loc="left", color=INK)
        ax.grid(False)
    save(fig, "chart_road_progress_validation.png", "p3_f10_validation.png")


def main():
    import json
    R, S, G = load()
    chart_curves(R, S)
    chart_strips(R, G)
    chart_timeline(R)
    km = chart_km(R)
    rk = chart_risk(R)
    mv = chart_media(R, S)
    chart_map(R)
    chart_decomp(R)
    chart_sensitivity()
    chart_validation()
    if "--crops" in sys.argv:
        chart_crops(R, G)
    f = ROOT / "paper" / "road_progress.json"
    d = json.loads(f.read_text())
    d["kaplan_meier"], d["delay_vs_risk"], d["media_validation"] = km, rk, mv
    f.write_text(json.dumps(d, indent=2))
    print(km, rk, mv)


if __name__ == "__main__":
    main()
