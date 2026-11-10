"""Paper analyses: produces tables (CSV/Markdown) and figures in paper/ for both papers.

Paper 1 (integrity): estimate bunching, competition, bidder rotation.
Paper 2 (efficiency/oversight): procurement cycle times, risk tiers, audit optimisation,
                                validation against PPDA suspensions.

All numbers in the paper drafts should come from paper/results.json written here.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entities import canonical  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "paper"
FIG = OUT / "figures"
RES: dict = {}


# ------------------------------------------------------------------ bunching
def bunching(ratio: pd.Series, lo=0.80, hi=1.20, width=0.005, excl=(0.97, 1.0), deg=5, n_boot=500, seed=0):
    """Chetty-style bunching estimator on award/estimate ratios.

    Fit a degree-`deg` polynomial to bin counts outside the exclusion window,
    predict the counterfactual inside it, excess mass b = (observed - predicted) / predicted.
    Bootstrap CIs by resampling contracts.
    """
    r = ratio[(ratio >= lo) & (ratio < hi)].to_numpy()
    edges = np.arange(lo, hi + 1e-9, width)
    mids = (edges[:-1] + edges[1:]) / 2
    inside = (mids >= excl[0]) & (mids < excl[1])

    def est(sample):
        cnt, _ = np.histogram(sample, edges)
        x = mids - 1.0
        coef = np.polyfit(x[~inside], cnt[~inside], deg)
        pred = np.polyval(coef, x)
        obs_in, pred_in = cnt[inside].sum(), max(pred[inside].sum(), 1e-9)
        return cnt, pred, obs_in, pred_in, (obs_in - pred_in) / pred_in

    cnt, pred, obs_in, pred_in, b = est(r)
    rng = np.random.default_rng(seed)
    boots = [est(rng.choice(r, len(r), replace=True))[4] for _ in range(n_boot)]
    return dict(n=int(len(r)), observed_in_window=int(obs_in), counterfactual_in_window=float(pred_in),
                excess_ratio=float(b), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                mids=mids, cnt=cnt, pred=pred)


def paper1(w: pd.DataFrame, bw: pd.DataFrame, sets: pd.DataFrame) -> None:
    ratio = w.award_to_est.dropna()
    main = bunching(ratio)
    RES["bunching_main"] = {k: v for k, v in main.items() if k not in ("mids", "cnt", "pred")}
    RES["share_within_1pct_below"] = float(ratio.between(0.99, 0.999999).mean())
    RES["share_exactly_equal"] = float(np.isclose(ratio, 1.0).mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(main["mids"], main["cnt"], width=0.005, color="#4C72B0", label="Observed")
    ax.plot(main["mids"], main["pred"], color="#C44E52", lw=2, label="Counterfactual (polynomial)")
    ax.axvspan(0.97, 1.0, color="grey", alpha=0.15, label="Excluded window")
    ax.set_xlabel("Award value / official estimate")
    ax.set_ylabel("Works contracts per 0.5% bin")
    ax.legend()
    ax.set_title(f"Excess mass just below the estimate: b = {main['excess_ratio']:.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_bunching.png", dpi=200)
    plt.close(fig)

    # heterogeneity
    rows = []
    w2 = w.assign(year=w.award_date.dt.year, ratio=w.award_to_est)
    w2["value_q"] = pd.qcut(w2.award_value_ugx, 4, labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"])
    for dim in ["method", "subsector", "value_q"]:
        for g, d in w2.groupby(dim, observed=True):
            r = d.ratio.dropna()
            if len(r) < 150:
                continue
            res = bunching(r, n_boot=200)
            rows.append(dict(dimension=dim, group=str(g), n=res["n"], excess_ratio=round(res["excess_ratio"], 2),
                             ci_low=round(res["ci95"][0], 2), ci_high=round(res["ci95"][1], 2),
                             share_99_100=round(float(r.between(0.99, 0.999999).mean()), 3)))
    het = pd.DataFrame(rows)
    het.to_csv(OUT / "tables" / "t2_bunching_heterogeneity.csv", index=False)
    RES["bunching_heterogeneity"] = rows

    # competition
    RES["works_tenders"] = int(len(w))
    RES["works_bid_records"] = int(len(bw))
    RES["works_award_value_ugx"] = float(w.award_value_ugx.sum())
    RES["method_shares"] = w.method.value_counts(normalize=True).round(4).to_dict()
    RES["single_bid_share_bidrecords"] = float(bw.f_single_bidder.mean())
    RES["n_bids_distribution"] = bw.n_bids.value_counts().sort_index().head(10).to_dict()
    RES["repeated_sets"] = int(len(sets))
    RES["rotating_sets"] = int(sets.rotating.sum())
    RES["rotating_sets_with_works"] = int((sets.rotating & sets.any_works).sum())
    top = sets[sets.rotating].sort_values("n_procurements", ascending=False).iloc[0]
    RES["largest_rotating_set"] = dict(n_procurements=int(top.n_procurements), n_firms=int(top.set_size),
                                       n_distinct_winners=int(top.n_distinct_winners), n_buyers=int(top.n_buyers))

    # flag co-occurrence table
    fl = w.filter(regex="^f_").astype(object).fillna(False).astype(bool)
    fb = bw.filter(regex="^f_").astype(object).fillna(False).astype(bool)
    t1 = pd.concat([fl.mean().rename("tender_records"), fb.mean().rename("bid_records")], axis=1).round(3)
    t1.to_csv(OUT / "tables" / "t1_flag_rates.csv")


# ------------------------------------------------------------------ paper 2
def paper2(w: pd.DataFrame, bw: pd.DataFrame) -> None:
    w = w.copy()
    w["bid_window"] = (w.tender_end - w.tender_start).dt.days
    w["post_close"] = (w.award_date - w.tender_end).dt.days
    w["cycle"] = (w.award_date - w.tender_start).dt.days
    ok = w[w.cycle.between(0, 1500) & w.post_close.between(0, 1500) & w.bid_window.between(0, 800)
           & (w.award_date.dt.year >= 2015)].copy()
    RES["cycle_n"] = int(len(ok))
    RES["cycle_median"] = {c: float(ok[c].median()) for c in ["bid_window", "post_close", "cycle"]}
    RES["post_close_share_of_cycle_median"] = float((ok.post_close / ok.cycle).median())
    by_m = ok.groupby("method")[["bid_window", "post_close", "cycle"]].median()
    by_m["n"] = ok.method.value_counts()
    by_m.to_csv(OUT / "tables" / "t3_cycle_by_method.csv")
    RES["cycle_by_method"] = by_m.reset_index().to_dict("records")

    ok["tier"] = pd.cut(ok.risk_score, [-1, 10, 20, 100], labels=["Low (≤10)", "Medium (10–20)", "High (>20)"])
    tiers = ok.groupby("tier", observed=True).agg(n=("cycle", "size"), median_post_close=("post_close", "median"),
                                                  median_cycle=("cycle", "median"),
                                                  value_share=("award_value_ugx", "sum"))
    tiers["share_of_records"] = tiers.n / tiers.n.sum()
    tiers["value_share"] = tiers.value_share / tiers.value_share.sum()
    tiers.to_csv(OUT / "tables" / "t4_risk_tiers.csv")
    RES["risk_tiers"] = tiers.reset_index().astype({"tier": str}).to_dict("records")

    # scenario: low-risk post-close time reduced to the 25th percentile of its own distribution
    low = ok[ok.tier == "Low (≤10)"]
    p25 = float(low.post_close.quantile(0.25))
    saved = (low.post_close - np.minimum(low.post_close, p25)).clip(lower=0)
    RES["scenario_fasttrack"] = dict(
        target_post_close_days=p25,
        median_days_saved_low_risk=float(saved.median()),
        mean_days_saved_low_risk=float(saved.mean()),
        low_risk_share=float(len(low) / len(ok)),
        overall_median_cycle_after=float(np.median(np.where(ok.tier == "Low (≤10)",
                                                            ok.cycle - (ok.post_close - np.minimum(ok.post_close, p25)),
                                                            ok.cycle))),
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, d in ok.groupby("tier", observed=True):
        ax.hist(d.post_close.clip(upper=600), bins=40, alpha=0.5, label=f"{lab} (n={len(d)})", density=True)
    ax.set_xlabel("Days from bid closing to award")
    ax.set_ylabel("Density")
    ax.set_title("Low-risk procurements wait as long as risky ones")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig2_postclose_by_tier.png", dpi=200)
    plt.close(fig)

    yr = ok.groupby(ok.award_date.dt.year)[["bid_window", "post_close"]].median()
    ax = yr.plot(kind="bar", stacked=True, figsize=(8, 4), color=["#55A868", "#C44E52"])
    ax.set_ylabel("Median days")
    ax.set_xlabel("Award year")
    ax.set_title("Where the time goes: bidding window vs post-closing")
    ax.legend(["Bidding window", "Bid closing → award"])
    plt.tight_layout()
    plt.savefig(FIG / "fig3_cycle_by_year.png", dpi=200)
    plt.close()

    # validation against PPDA suspensions
    s = pd.read_parquet(P / "suspended.parquet")
    sc = set(s.provider_name.map(canonical))
    allw = pd.concat([w[["supplier_name", "risk_score"]], bw[["supplier_name", "risk_score"]]])
    allw["suspended"] = allw.supplier_name.map(canonical).isin(sc)
    v = allw.groupby("suspended").risk_score.agg(["mean", "median", "count"])
    RES["validation_suspended"] = {str(k): {kk: float(vv) for kk, vv in r.items()} for k, r in v.iterrows()}
    t = pd.read_parquet(P / "tenders.parquet")
    b = pd.read_parquet(P / "bid_procs.parquet")
    winners = set(pd.concat([t.supplier_name, b.supplier_name]).map(canonical))
    RES["suspended_total"] = int(len(sc))
    RES["suspended_found_as_winners"] = int(len(sc & winners))
    br = s.breach_committed.fillna("").str.lower()
    RES["suspension_breach_types"] = {
        "forged_or_false_documents": int(br.str.contains("forg|false|unauthentic|fake|misrepresent").sum()),
        "financial_loss": int(br.str.contains("loss").sum()),
        "other": int((~br.str.contains("forg|false|unauthentic|fake|misrepresent|loss")).sum()),
    }

    # variations (thin data, descriptive only)
    vp = pd.read_parquet(P / "variations_price.parquet")
    vt = pd.read_parquet(P / "variations_time.parquet")
    RES["variations"] = dict(price_records=int(len(vp)), time_records=int(len(vt)),
                             time_ext_median_days=float(vt.extension_days[vt.extension_days > 0].median()))


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    bw = pd.read_parquet(P / "works_bids_scored.parquet")
    sets = pd.read_parquet(P / "bidder_sets.parquet")
    paper1(w, bw, sets)
    paper2(w, bw)
    (OUT / "results.json").write_text(json.dumps(RES, indent=2, default=float))
    print(json.dumps(RES, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
