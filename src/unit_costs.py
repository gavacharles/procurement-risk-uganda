"""(2) Unit-cost benchmarking: cost per latrine stance, classroom, borehole, staff-house unit, km.

NLP extraction from titles, deliberately conservative: a title is used only when it
describes ONE countable item type, built new (no renovation/repair/completion), with a
clearly stated quantity. Mixed contracts ("classroom block and latrine") are skipped.

Benchmark: each contract's unit cost is compared with the median for its category in
the same year (handles inflation without a CPI series). Flags:
  * price_ratio = unit cost / category-year median
  * outlier     = price_ratio > 2.0
  * excess_over_p75 = max(0, award - P75 unit cost x quantity)   (conservative "at stake")
Also: an estimate-side check, are *estimates* (not just awards) unusually high per unit?
That speaks to the hypothesis that bunching at inflated estimates hides overpricing.

Out: data/processed/unit_costs.parquet, paper/unit_costs.json, posts/media/chart_unit_costs.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
WORDNUM = {"one": 1, "a": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20}
NUM = r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)"
NEW_BUILD = re.compile(r"\b(construct\w*|erection|drilling|building of|upgrading|proposed)\b", re.I)
NOT_NEW = re.compile(r"\b(renovat\w*|rehabilit\w*|repair\w*|complet\w*|finish\w*|maintenance|remodel\w*|"
                     r"extension|phase|balance|additional|retention|supervision|consult\w*)\b", re.I)
MULTI = re.compile(r"\b(\d+|two|three|four|five|six)\s*(no\.?s?|nos|blocks?\s+of|units?\s+of)\b|\(\s*\d+\s*\)\s*no", re.I)


def n(x: str) -> int | None:
    x = x.lower()
    return int(x) if x.isdigit() else WORDNUM.get(x)


def extract(title: str) -> tuple[str, float, str] | None:
    """Return (category, quantity, unit) or None."""
    if not isinstance(title, str):
        return None
    t = " " + re.sub(r"\s+", " ", title.lower().replace("&amp;", "&")) + " "
    has = lambda w: re.search(w, t) is not None  # noqa: E731
    # --- latrines: per stance (single latrine block only)
    if has(r"stance") and has(r"latrine|toilet") and not has(r"class ?room|staff house|house|ward|office block|dormitor|kitchen|borehole|fence"):
        m = re.findall(NUM + r"\s*-?\s*stance", t)
        if len(m) == 1 and not MULTI.search(t) and not has(r"latrines|toilets") and NEW_BUILD.search(t) and not NOT_NEW.search(t):
            q = n(m[0])
            if q and 1 <= q <= 20:
                return "latrine", q, "per stance"
    # --- classroom blocks: per classroom
    if has(r"class ?r?oom") and not has(r"latrine|toilet|staff house|house|dormitor|laborator|library|kitchen|desks|furniture|borehole"):
        m = re.findall(NUM + r"\s*-?\s*(?:in\s*-?\s*1\s*)?class ?r?oom", t) or re.findall(r"(\d)\s*-?\s*in\s*-?\s*1\s*class", t)
        if len(m) == 1 and not MULTI.search(t) and not has(r"blocks") and NEW_BUILD.search(t) and not NOT_NEW.search(t):
            q = n(m[0])
            if q and 1 <= q <= 8:
                return "classroom", q, "per classroom"
    # --- staff houses: per unit
    if has(r"staff house") and not has(r"latrine|class ?room|toilet|kitchen|ward|houses"):
        m = re.findall(NUM + r"\s*-?\s*(?:unit|in\s*-?\s*1)\s*staff house", t)
        if len(m) == 1 and NEW_BUILD.search(t) and not NOT_NEW.search(t):
            q = n(m[0])
            if q and 1 <= q <= 12:
                return "staff_house", q, "per housing unit"
    # --- boreholes: per borehole (hand-pump/deep vs production kept apart)
    if has(r"borehole") and has(r"drill") and not has(r"rehabilit|repair|motoris|solar|pipe|tank|piped|survey|siting only"):
        m = re.findall(r"(?:drilling|construction)[^.]*?\(?\b" + NUM + r"\b\)?\s*(?:no\.?\s*)?(?:\(\d+\)\s*)?(?:deep |hand[- ]?pump(?:ed)? |production |new )*boreholes?", t)
        kind = "borehole_production" if has(r"production") else "borehole_handpump"
        if len(m) == 1:
            q = n(m[0])
            if q and 1 <= q <= 60:
                return kind, q, "per borehole"
    # --- roads: per km, by intervention type
    if has(r"\broads?\b") and not has(r"pipe|dn\d|water|sewer|power|line|fibre|cable|complet|additional|balance|phase"):
        m = re.findall(r"(\d+(?:\.\d+)?)\s*km", t)
        if len(m) == 1:
            q = float(m[0])
            if 0.3 <= q <= 300:
                if has(r"upgrad") and has(r"bitumen|paved|tarmac|asphalt|class ii|class ll"):
                    return "road_upgrade_paved", q, "per km"
                if has(r"periodic maintenance|reseal|overlay"):
                    return "road_periodic_maint", q, "per km"
                if has(r"rehabilit|reconstruct"):
                    return "road_rehab", q, "per km"
    # --- pipelines: per km
    if has(r"pipe ?line|dn\s?\d+|pipe laying|laying of") and has(r"water|mains|dn\s?\d"):
        m = re.findall(r"(\d+(?:\.\d+)?)\s*km", t)
        if len(m) == 1:
            q = float(m[0])
            if 0.2 <= q <= 200:
                return "pipeline", q, "per km"
    return None


def main() -> None:
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    rows = []
    for r in w.itertuples(index=False):
        e = extract(r.title)
        if e and r.award_value_ugx and r.award_value_ugx > 0:
            yr = r.tender_start.year if pd.notna(r.tender_start) and r.tender_start.year >= 2014 else None
            rows.append(dict(ocid=r.ocid, buyer_name=r.buyer_name, title=r.title, category=e[0], quantity=e[1], unit=e[2],
                             award=r.award_value_ugx, estimate=r.estimate_ugx if r.estimate_ugx and r.estimate_ugx > 0 else np.nan,
                             year=yr, risk_score=r.risk_score, method=r.method, award_to_est=r.award_to_est))
    u = pd.DataFrame(rows)
    u["unit_cost"] = u.award / u.quantity
    u["unit_est"] = u.estimate / u.quantity
    # implausible tails (data entry errors): drop the extreme 1% each side within category
    lo = u.groupby("category").unit_cost.transform(lambda s: s.quantile(0.01))
    hi = u.groupby("category").unit_cost.transform(lambda s: s.quantile(0.99))
    u = u[(u.unit_cost >= lo) & (u.unit_cost <= hi)].copy()
    # category-year benchmark (fall back to category-wide where a year is thin)
    key = u.category + "|" + u.year.fillna(0).astype(int).astype(str)
    med_cy = u.groupby(key).unit_cost.transform("median")
    n_cy = u.groupby(key).unit_cost.transform("size")
    med_c = u.groupby("category").unit_cost.transform("median")
    u["benchmark"] = np.where(n_cy >= 8, med_cy, med_c)
    p75 = u.groupby("category").unit_cost.transform(lambda s: s.quantile(0.75)) * (u.benchmark / med_c)
    u["price_ratio"] = u.unit_cost / u.benchmark
    u["outlier"] = u.price_ratio > 2.0
    u["excess_over_p75"] = np.clip(u.award - p75 * u.quantity, 0, None)
    u["est_ratio"] = u.unit_est / u.groupby(key).unit_est.transform("median").where(n_cy >= 8, u.groupby("category").unit_est.transform("median"))
    u.to_parquet(P / "unit_costs.parquet", index=False)

    cats = u.groupby(["category", "unit"]).agg(
        contracts=("ocid", "size"), median_unit_cost_ugx=("unit_cost", "median"),
        p25=("unit_cost", lambda s: s.quantile(.25)), p75=("unit_cost", lambda s: s.quantile(.75)),
        outliers_over_2x=("outlier", "sum"), excess_over_p75_ugx=("excess_over_p75", "sum")).reset_index()
    corr = u.dropna(subset=["est_ratio"])
    RELIABLE = ["latrine", "classroom", "borehole_production", "borehole_handpump", "staff_house", "pipeline"]
    rel = u[u.category.isin(RELIABLE)]
    res = dict(
        contracts_with_units=int(len(u)),
        categories=cats.round(0).to_dict("records"),
        reliable_categories=RELIABLE,
        note_roads="Road categories have 6-13 heterogeneous contracts each: reported descriptively, excluded from totals.",
        reliable_contracts=int(len(rel)),
        total_outliers_over_2x=int(rel.outlier.sum()),
        outlier_award_value_ugx=float(rel.loc[rel.outlier, "award"].sum()),
        excess_over_p75_total_ugx=float(rel.excess_over_p75.sum()),
        excess_over_p75_outliers_only_ugx=float(rel.loc[rel.outlier, "excess_over_p75"].sum()),
        # does a high estimate predict a high price? (inflated-estimate channel)
        spearman_est_vs_price_ratio=float(corr[["est_ratio", "price_ratio"]].corr(method="spearman").iloc[0, 1]),
        bunched_vs_not_median_price_ratio=dict(
            bunched_99_100=float(u[u.award_to_est.between(0.99, 0.999999)].price_ratio.median()),
            not_bunched=float(u[~u.award_to_est.between(0.99, 0.999999)].price_ratio.median())),
        outlier_rate_by_method=u.groupby("method").outlier.mean().round(3).to_dict(),
        mean_risk_outlier_vs_not=u.groupby("outlier").risk_score.mean().round(2).to_dict(),
    )
    (ROOT / "paper" / "unit_costs.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()


def chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visuals import BLUE, ORANGE, INK2, title  # shared style
    u = pd.read_parquet(P / "unit_costs.parquet")
    fig = plt.figure(figsize=(8, 4.6))
    specs = [("latrine", "Latrine: cost per stance (UGX m)"), ("classroom", "Classroom block: cost per classroom (UGX m)")]
    title(fig, "What should a latrine or a classroom cost?",
          "Unit costs extracted from contract titles (NLP); orange = more than 2× the same-year median")
    rng = np.random.default_rng(1)
    for i, (cat, lab) in enumerate(specs):
        ax = fig.add_axes([0.08 + i * 0.47, 0.17, 0.40, 0.60])
        g = u[u.category == cat]
        y = rng.uniform(-0.35, 0.35, len(g))
        ax.scatter(g.unit_cost / 1e6, y, s=10, color=np.where(g.outlier, ORANGE, BLUE), alpha=0.75, linewidths=0)
        med = g.unit_cost.median() / 1e6
        ax.axvline(med, color=INK2, lw=1, ls="--")
        ax.text(med, 0.48, f"median {med:.1f}m", ha="center", fontsize=9.5, color=INK2)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab)
        ax.text(0.98, 0.04, f"n = {len(g)} · outliers {int(g.outlier.sum())}", transform=ax.transAxes, ha="right",
                fontsize=9, color=INK2)
    out = ROOT / "posts" / "media" / "chart_unit_costs.png"
    fig.savefig(out, dpi=200)
    import shutil
    shutil.copy(out, ROOT / "docs" / "media" / "chart_unit_costs.png")


if __name__ == "__main__" and "--chart" in sys.argv:
    chart()
