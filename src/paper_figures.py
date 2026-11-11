"""Publication figures for both papers (vector PDF + PNG, journal style: no in-figure titles,
panel letters, one palette). All numbers read from paper/*.json and data/processed.

Paper 1 (paper/figures/p1_*): the price-anchoring argument
  F1 bunching + counterfactual      F2 robustness (24 specs + ML sample + placebos)
  F3 heterogeneity                  F4 mechanism (entry timing, edits) + theory
  F5 not overpricing (unit cost & hedonic residuals)   F6 cross-country
  F7 threshold bunching             F8 elections / fiscal-year coefficients
Paper 2 (paper/figures/p2_*): the late-and-untargeted-oversight argument
  F1 publication lag + share entered after award by year
  F2 audit selection vs risk        F3 rules vs machine learning agreement
  F4 audit coverage curve (independent targets)   F5 toolkit pipeline schematic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
J = ROOT / "paper"
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "src"))

BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 8,
})
pct = matplotlib.ticker.PercentFormatter(1, decimals=0)


def jload(name):
    return json.loads((J / name).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")


# ================================================================ PAPER 1
def p1_f1():
    from analysis import bunching
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    b = bunching(w.award_to_est.dropna(), n_boot=50)
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.bar(b["mids"], b["cnt"], width=0.0045, color=BLUE, label="Observed contracts")
    ax.plot(b["mids"], b["pred"], color=ORANGE, lw=1.8, label="Smooth counterfactual (degree-5 polynomial)")
    ax.axvspan(0.97, 1.0, color=ORANGE, alpha=0.08, lw=0, label="Excluded window [0.97, 1.00)")
    ax.set_xlim(0.8, 1.2)
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("Award value / official estimate")
    ax.set_ylabel("Works contracts per 0.5% bin")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.annotate(f"b = {b['excess_ratio']:.2f}\n95% CI {b['ci95'][0]:.2f}–{b['ci95'][1]:.2f}",
                xy=(1.0, b["cnt"].max() * 0.9), xytext=(1.06, b["cnt"].max() * 0.75), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    save(fig, "p1_f1_bunching")


def p1_f2():
    rob = pd.read_csv(J / "tables" / "t5_bunching_robustness.csv")
    rob["lo"] = rob.ci.str.split("-").str[0].astype(float)
    rob["hi"] = rob.ci.str.split("-").str[1].astype(float)
    mc = jload("ml_classifier.json")["bunching_by_sample"]
    rows = [("Baseline (rule-based works)", mc["rule_works"]["bunching_b"], *mc["rule_works"]["ci95"], INK),
            ("ML-classified works", mc["ml_works"]["bunching_b"], *mc["ml_works"]["ci95"], ORANGE),
            ("ML-only new works", mc["ml_only_new"]["bunching_b"], *mc["ml_only_new"]["ci95"], ORANGE)]
    for r in rob.itertuples():
        rows.append((f"deg {r.deg} · window {r.window} · range {r.range}", r.excess, r.lo, r.hi, BLUE))
    fig, ax = plt.subplots(figsize=(6.3, 6.4))
    for i, (lab, b, lo, hi, c) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=c, lw=1.4)
        ax.plot(b, i, "o", color=c, ms=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Excess mass b (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.text(0.98, 0.02, "Placebo windows at 0.90, 0.93, 1.07, 1.10: b < 0 (no bunching)", transform=ax.transAxes,
            ha="right", fontsize=7.5, color=INK2)
    save(fig, "p1_f2_robustness")


def p1_f3():
    h = pd.read_csv(J / "tables" / "t2_bunching_heterogeneity.csv")
    lab = {"method": "Method", "subsector": "Sub-sector", "value_q": "Value quartile"}
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y, ticks, names = 0, [], []
    for dim, g in h.groupby("dimension", sort=False):
        for r in g.itertuples():
            ax.plot([r.ci_low, r.ci_high], [y, y], color=BLUE, lw=1.4)
            ax.plot(r.excess_ratio, y, "o", color=BLUE, ms=4)
            ticks.append(y)
            names.append(f"{lab[dim]}: {r.group} (n={r.n:,})")
            y += 1
        y += 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Excess mass b (95% CI)")
    ax.grid(axis="y", visible=False)
    save(fig, "p1_f3_heterogeneity")


def p1_f4():
    from scipy.stats import beta
    et = jload("estimate_timing.json")["random_sample_timing"]
    th = jload("theory_sim.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})

    def ci(k, n):
        return beta.ppf(0.025, k, n - k + 1) if k else 0, beta.ppf(0.975, k + 1, n - k)
    t2 = et["T2_table"]  # rows: ante, post ; cols: not bunched, bunched (sorted labels)
    t3 = et["T3_table_rows_edited_FT_cols_bunched_FT"]
    groups = [("Entered before\nbid opening", t2[0][1], sum(t2[0])), ("Entered after\naward", t2[1][1], sum(t2[1])),
              ("Not edited\nafter award", t3[0][1], sum(t3[0])), ("Edited\nafter award", t3[1][1], sum(t3[1]))]
    for i, (lab, k, n) in enumerate(groups):
        lo, hi = ci(k, n)
        a.bar(i, k / n, color=[BLUE, BLUE, AQUA, ORANGE][i], width=0.62)
        a.plot([i, i], [lo, hi], color=INK, lw=1)
        a.text(i, 0.008, f"{k}/{n}", ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")
    a.set_xticks(range(4))
    a.set_xticklabels([g[0] for g in groups], fontsize=7)
    a.yaxis.set_major_formatter(pct)
    a.set_ylabel("Awards at 99–100% of estimate")
    a.grid(axis="x", visible=False)
    top = a.get_ylim()[1]
    for (x0, x1, pv) in [(0, 1, et["T2_fisher_p"]), (2, 3, et["T3_fisher_p"])]:
        yb = top * 0.97
        a.plot([x0, x0, x1, x1], [yb - top * 0.02, yb, yb, yb - top * 0.02], color=INK2, lw=0.8)
        a.text((x0 + x1) / 2, yb + top * 0.01, f"p = {pv:.2f}" if pv >= 0.01 else f"p = {pv:.4f}", ha="center", fontsize=7, color=INK2)
    a.set_ylim(0, top * 1.08)
    panel(a, "A")
    obs = th["observed_uganda"]["share_99_100"]
    vals = [("Observed", obs, INK), ("Known\nestimate", th["known_estimate"]["share_99_100"], BLUE)]
    for s, d in th["secret_estimate"].items():
        vals.append((f"Secret\n±{float(s.split('_')[-1]):.0%}", d["share_99_100"], ORANGE))
    for i, (lab, v, c) in enumerate(vals):
        b.bar(i, v, color=c, width=0.62)
        b.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.5, color=INK2)
    b.set_xticks(range(len(vals)))
    b.set_xticklabels([v[0] for v in vals], fontsize=7)
    b.set_ylabel("Awards at 99–100% (model vs data)")
    b.yaxis.set_major_formatter(pct)
    b.grid(axis="x", visible=False)
    panel(b, "B")
    fig.tight_layout()
    save(fig, "p1_f4_mechanism_theory")


def p1_f5():
    u = pd.read_parquet(P / "unit_costs.parquet")
    r = pd.read_parquet(P / "price_residuals.parquet").merge(
        pd.read_parquet(P / "works_tenders_scored.parquet")[["ocid", "award_to_est"]], on="ocid", how="left")
    r["bunched"] = r.award_to_est.between(0.99, 0.999999)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for i, (mask, lab, c) in enumerate([(~u.award_to_est.between(0.99, 0.999999), "Not bunched", BLUE),
                                        (u.award_to_est.between(0.99, 0.999999), "Bunched (99–100%)", ORANGE)]):
        v = np.log2(u[mask].price_ratio.clip(0.125, 8))
        a.hist(v, bins=np.arange(-3, 3.01, 0.2), density=True, histtype="step", lw=1.6, color=c, label=f"{lab} (n={mask.sum()})")
    a.set_xticks([-2, -1, 0, 1, 2])
    a.set_xticklabels(["¼×", "½×", "1×", "2×", "4×"])
    a.set_xlabel("Unit cost / same-year category median")
    a.set_ylabel("Density")
    a.legend(fontsize=7)
    panel(a, "A")
    for mask, lab, c in [(~r.bunched & r.award_to_est.notna(), "Not bunched", BLUE), (r.bunched, "Bunched", ORANGE)]:
        b.hist(r[mask].resid.clip(-3, 3), bins=np.arange(-3, 3.01, 0.15), density=True, histtype="step", lw=1.6, color=c, label=lab)
    b.axvline(0, color=INK2, lw=0.8)
    b.set_xlabel("log(award) − ML benchmark (cross-fitted)")
    b.legend(fontsize=7)
    panel(b, "B")
    fig.tight_layout()
    save(fig, "p1_f5_not_overpricing")


def p1_f6():
    r = jload("cross_country.json")
    rows = []
    for k, v in r.items():
        m = v.get("works") or v.get("all")
        if m and m.get("share_99_100") is not None and m["n"] >= 200 and not m.get("artefact"):
            rows.append((k, m["share_99_100"], m["n"]))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.barh([x[0] for x in rows], [x[1] for x in rows], color=[ORANGE if x[0] == "Uganda" else BLUE for x in rows], height=0.6)
    for i, x in enumerate(rows):
        ax.text(x[1] + 0.004, i, f"{x[1]:.0%} (n={x[2]:,})", va="center", fontsize=7.5, color=INK2)
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlim(0, max(x[1] for x in rows) * 1.4)
    ax.set_xlabel("Share of awards at 99–100% of estimate")
    ax.grid(axis="y", visible=False)
    save(fig, "p1_f6_cross_country")


def p1_f7():
    t = jload("threshold_bunching.json")
    keys = ["5m", "10m", "20m", "30m", "50m", "100m", "200m", "300m", "500m"]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    x = np.arange(len(keys))
    for off, (name, lab, c) in zip([-0.27, 0, 0.27], [("all_estimate", "All estimates", GREY), ("works_estimate", "Works estimates", BLUE),
                                                      ("works_award", "Works awards", ORANGE)]):
        vals = [t[name].get(k, {}).get("ratio", np.nan) for k in keys]
        ax.bar(x + off, vals, width=0.26, color=c, label=lab)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.axhspan(1.0, 1.4, color=GREY, alpha=0.12, lw=0)
    ax.text(len(keys) - 0.5, 1.2, "charm-pricing range", ha="right", va="center", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_xlabel("Round value (UGX)")
    ax.set_ylim(0, 6.3)
    ax.set_ylabel("Count just below ÷ just above")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center")
    save(fig, "p1_f7_threshold_bunching")


def p1_f8():
    ne = jload("natural_experiments.json")["all_sectors"]
    lab = {"direct": "Direct procurement", "non_open": "Not openly bid", "bunched": "Award 99–100% of estimate",
           "exact": "Award = estimate", "above_est": "Award >10% above estimate", "short_window": "Bid window <10 days"}
    wins = [("pre_election", "6 months before election", BLUE), ("fy_q4", "Fiscal Q4 (Apr–Jun)", ORANGE), ("covid", "COVID-19 (2020)", AQUA)]
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    ys = np.arange(len(lab))
    for j, (k, name, c) in enumerate(wins):
        for i, y in enumerate(lab):
            r = ne[y][k]
            ax.plot(r["ci95"], [i + (j - 1) * 0.22] * 2, color=c, lw=1.4)
            ax.plot(r["coef"], i + (j - 1) * 0.22, "o", color=c, ms=3.5, label=name if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(list(lab.values()), fontsize=7.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f} pts"))
    ax.set_xlabel("Effect (percentage points), entity + year fixed effects, 95% CI clustered by entity")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "p1_f8_elections_fiscal")


# ================================================================ PAPER 2
def p2_f1():
    from estimate_timing import load_invitations
    r = jload("estimate_timing.json")
    t = r["random_sample_timing"]
    by = r["works_entered_after_award_by_fy"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    a.barh(0, t["cycle_median"], color=BLUE, height=0.5)
    a.barh(0, t["lag_median"], left=t["cycle_median"], color=ORANGE, height=0.5)
    a.text(t["cycle_median"] / 2, 0, f"Procurement\n{t['cycle_median']:.0f} days", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    a.text(t["cycle_median"] + t["lag_median"] / 2, 0, f"Until public: {t['lag_median']:.0f} days\n(IQR {t['lag_p25']:.0f}–{t['lag_p75']:.0f})",
           ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    a.set_yticks([])
    a.set_xlabel("Median days from invitation (works, random sample)")
    a.grid(axis="y", visible=False)
    panel(a, "A")
    ys = sorted(k for k in by if k >= "2015")
    b.bar(range(len(ys)), [by[k] for k in ys], color=ORANGE, width=0.7)
    b.set_xticks(range(len(ys)))
    b.set_xticklabels([k[2:4] + "/" + k[7:9] for k in ys], fontsize=6.5, rotation=45)
    b.yaxis.set_major_formatter(pct)
    b.set_ylabel("Entered on portal after award")
    b.set_ylim(0, 1)
    b.grid(axis="x", visible=False)
    panel(b, "B")
    fig.tight_layout()
    save(fig, "p2_f1_publication_lag")


def p2_f2():
    F = pd.read_parquet(P / "oag_engineering.parquet")
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    prof = w.groupby("buyer_name").agg(n=("ocid", "size"), mean_risk=("risk_score", "mean")).reset_index()
    prof = prof[prof.n >= 5]
    audited = set(F.buyer_name.dropna())
    prof["audited"] = prof.buyer_name.isin(audited)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    rng = np.random.default_rng(1)
    for i, (flag, lab, c) in enumerate([(False, "Never audited", BLUE), (True, "Engineering-audited", ORANGE)]):
        v = prof[prof.audited == flag].mean_risk
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=10, color=c, alpha=0.7, linewidths=0)
        ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.5)
        ax.text(v.median(), i + 0.36, f"median {v.median():.1f} (n={len(v)})", ha="center", fontsize=7.5, color=INK2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Never audited", "Engineering-audited"])
    ax.set_xlabel("Entity mean risk score (works, 0–100)")
    ax.grid(axis="y", visible=False)
    save(fig, "p2_f2_audit_selection")


def p2_f3():
    m = pd.read_parquet(P / "ml_anomaly.parquet")
    m["rule_pct"] = m.risk_score.rank(pct=True, method="first")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    hb = ax.hexbin(m.rule_pct, m.ml_anomaly_pct, gridsize=20, cmap="Blues", mincnt=1, linewidths=0.2)
    ax.axvline(0.9, color=ORANGE, lw=1, ls="--")
    ax.axhline(0.9, color=ORANGE, lw=1, ls="--")
    r = jload("ml_results.json")["isolation_forest"]
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    ax.text(0.95, 0.95, f"overlap\n{r['top_decile_overlap']:.0%}", ha="center", va="center", fontsize=7.5, color=INK, bbox=box)
    ax.text(0.45, 0.95, f"ML only\n({r['ml_only_top_decile']})", ha="center", va="center", fontsize=7.5, color=INK2, bbox=box)
    ax.set_xlabel("Rule-based risk (percentile)")
    ax.set_ylabel("Isolation Forest anomaly (percentile)")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    fig.colorbar(hb, ax=ax, label="Contracts", shrink=0.8)
    save(fig, "p2_f3_rules_vs_ml")


def p2_f4():
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    w = w[w.award_value_ugx > 0].copy()
    ml = pd.read_parquet(P / "ml_anomaly.parquet")[["ocid", "ml_anomaly_pct"]]
    uc = pd.read_parquet(P / "unit_costs.parquet")[["ocid", "outlier"]]
    sp = pd.read_parquet(P / "split_candidates.parquet")
    sp = sp[sp.category == "possible_split"]
    split_ids = set(sp.ocid_a) | set(sp.ocid_b)
    w = w.merge(ml, on="ocid", how="left").merge(uc, on="ocid", how="left")
    # independent targets: not part of the rule score
    w["target"] = (w.ml_anomaly_pct >= 0.9) | (w.outlier == True) | w.ocid.isin(split_ids)  # noqa: E712
    w["cost"] = 5 + 3 * np.maximum(0, np.log10(w.award_value_ugx) - 6)
    total = w.target.sum()

    tot_cost = w.cost.sum()

    def curve(order):
        d = w.loc[order]
        return d.cost.cumsum().to_numpy() / tot_cost, d.target.cumsum().to_numpy() / total
    risk_order = (w.risk_score * w.award_value_ugx / w.cost).sort_values(ascending=False).index
    big_order = w.award_value_ugx.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for order, lab, c in [(risk_order, "Risk-based (rule score × value / cost)", ORANGE), (big_order, "Largest contracts first", GREY)]:
        x, y = curve(order)
        ax.plot(x, y, color=c, lw=1.8, label=lab)
    rng = np.random.default_rng(0)
    rs = [curve(rng.permutation(w.index)) for _ in range(30)]
    grid = np.linspace(0, 1, 300)
    ys = np.array([np.interp(grid, x, y) for x, y in rs])
    ax.fill_between(grid, ys.min(0), ys.max(0), color=BLUE, alpha=0.2, lw=0)
    ax.plot(grid, ys.mean(0), color=BLUE, lw=1.4, label="Random selection (30 draws)")
    ax.set_xlim(0, 0.5)
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel(f"Share of total audit effort (100% = {tot_cost:,.0f} auditor-days)")
    for order, c in [(risk_order, ORANGE)]:
        x, y = curve(order)
        y20 = float(np.interp(0.2, x, y))
        ax.annotate(f"risk-based: {y20:.0%} at 20% effort", xy=(0.2, y20), xytext=(0.22, y20 + 0.14), fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    r20 = float(np.interp(0.2, grid, ys.mean(0)))
    ax.annotate(f"random: {r20:.0%}", xy=(0.2, r20), xytext=(0.26, r20 - 0.08), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylabel("Share of independently flagged\ncontracts covered")
    ax.yaxis.set_major_formatter(pct)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=7)
    ax.text(0.02, 0.95, f"Targets: ML anomalies (top 10%), unit-cost outliers, possible splits (n={total:,})",
            transform=ax.transAxes, fontsize=7, color=INK2, va="top")
    save(fig, "p2_f4_audit_coverage")


def p2_f5():
    fig, ax = plt.subplots(figsize=(8.2, 2.7))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    boxes = [
        (0.1, "Open data", "PPDA OCDS\nbid invitations\nOAG audits\nSentinel-2 · OSM", GREY),
        (2.1, "Screens", "Red flags\nBenford · networks\nIsolation Forest\nTF-IDF · unit costs", BLUE),
        (4.1, "Targeting", "Risk score\nknapsack audit plan\nfast lane for\nlow-risk", ORANGE),
        (6.1, "Verification", "Satellite change\ndetection\ndocument checks\nfield inspection", AQUA),
        (8.1, "Privacy-safe", "Blind bid codes\nprivate conflict\ncheck\n(number theory)", YELLOW),
    ]
    for x, head, body, c in boxes:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 0.35), 1.75, 2.3, boxstyle="round,pad=0.03,rounding_size=0.12",
                                                        fc="white", ec=c, lw=1.8))
        ax.text(x + 0.875, 2.3, head, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(x + 0.875, 1.3, body, ha="center", va="center", fontsize=7, color=INK2, linespacing=1.4)
    for x in [1.85, 3.85, 5.85, 7.85]:
        ax.annotate("", xy=(x + 0.25, 1.5), xytext=(x, 1.5), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.text(5, 0.08, "Binding constraint: real-time publication (82% of works tenders are entered only after award)",
            ha="center", fontsize=7.5, color=ORANGE)
    save(fig, "p2_f5_toolkit_pipeline")


def main():
    for f in [p1_f1, p1_f2, p1_f3, p1_f4, p1_f5, p1_f6, p1_f7, p1_f8, p2_f1, p2_f2, p2_f3, p2_f4, p2_f5]:
        try:
            f()
        except Exception as e:
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
