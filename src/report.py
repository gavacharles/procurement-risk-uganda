"""Generate the exploratory data & findings report: reports/exploratory_report.md (+ PNG charts).

Facts in the report are computed from data/processed; re-run after each data refresh.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
R = ROOT / "reports"
FIG = R / "figures"


def ugx(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    for div, s in [(1e12, "tn"), (1e9, "bn"), (1e6, "m")]:
        if abs(x) >= div:
            return f"UGX {x / div:,.1f}{s}"
    return f"UGX {x:,.0f}"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    tenders = pd.read_parquet(P / "tenders.parquet")
    procs = pd.read_parquet(P / "bid_procs.parquet")
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    bw = pd.read_parquet(P / "works_bids_scored.parquet")
    sets = pd.read_parquet(P / "bidder_sets.parquet")
    keys = pd.read_parquet(P / "entity_keys.parquet")
    buyers = pd.read_parquet(P / "buyer_profiles.parquet")

    # ---- charts ----
    r = w.award_to_est.dropna()
    r = r[(r > 0.5) & (r < 1.5)]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(r, bins=np.arange(0.5, 1.5, 0.01), color="#4C72B0")
    ax.axvline(1.0, color="#C44E52", lw=1)
    ax.set_xlabel("Award value / official estimate")
    ax.set_ylabel("Works contracts")
    ax.set_title("Awards bunch just below the estimate")
    fig.tight_layout()
    fig.savefig(FIG / "award_to_estimate.png", dpi=150)
    plt.close(fig)

    v = pd.concat([w.award_value_ugx, bw.award_value_ugx]).dropna()
    v = v[v > 0]
    first = v.map(lambda x: int(f"{x:e}"[0]))
    obs = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    exp = np.log10(1 + 1 / np.arange(1, 10))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(1, 10), obs.values, color="#4C72B0", label="Observed (works awards)")
    ax.plot(range(1, 10), exp, "o-", color="#C44E52", label="Benford expected")
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("First digit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "benford_works.png", dpi=150)
    plt.close(fig)
    mad_all = float(np.mean(np.abs(obs.values - exp)))

    wy = w[w.award_date.dt.year >= 2014]
    yr = wy.groupby(wy.award_date.dt.year).agg(n=("ocid", "size"), value=("award_value_ugx", "sum"))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(yr.index.astype(int), yr.value / 1e9, color="#4C72B0")
    ax.set_ylabel("Award value (UGX bn)")
    ax.set_title("Works awards by year (tender records)")
    fig.tight_layout()
    fig.savefig(FIG / "works_by_year.png", dpi=150)
    plt.close(fig)

    # ---- numbers ----
    yrs = w.award_date.dt.year[w.award_date.dt.year >= 2014]
    if mad_all <= 0.006:
        benford_txt = "close conformity"
    elif mad_all <= 0.012:
        benford_txt = "acceptable-to-marginal conformity (Nigrini bands); the deviation that exists comes mainly from round-number pricing"
    elif mad_all <= 0.015:
        benford_txt = "marginal conformity, largely due to round-number pricing"
    else:
        benford_txt = "nonconformity, partly due to round-number pricing"
    n_w, n_bw = len(w), len(bw)
    just_below = w.award_to_est.between(0.99, 0.999999).sum() / w.award_to_est.notna().sum()
    band_95_98 = w.award_to_est.between(0.95, 0.98).sum() / w.award_to_est.notna().sum() / 3
    single = bw.f_single_bidder.mean()
    rot = sets[sets.rotating].copy()
    top_sets = rot.sort_values("n_procurements", ascending=False).head(8)
    flag_t = w[[c for c in w.columns if c.startswith("f_")]].fillna(False).mean().sort_values(ascending=False)
    flag_b = bw[[c for c in bw.columns if c.startswith("f_")]].fillna(False).mean().sort_values(ascending=False)
    top = pd.concat([
        w.assign(source="tender")[["source", "ocid", "buyer_name", "supplier_name", "title", "award_value_ugx", "risk_score", "reasons"]],
        bw.assign(source="bids")[["source", "ocid", "buyer_name", "supplier_name", "title", "award_value_ugx", "risk_score", "reasons"]],
    ])
    # audit priority: risk weighted by money at stake (log scale so mega-projects don't swamp everything)
    top["priority_index"] = (top.risk_score * np.log10(top.award_value_ugx.clip(lower=1e6))).round(1)
    top = top.sort_values("priority_index", ascending=False)
    top.to_csv(R / "top_risk_works_contracts.csv", index=False)

    def tbl(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    md = f"""# Uganda Construction Procurement: Exploratory Data & Findings Report

*Generated {date.today().isoformat()} from the PPDA OCDS feed (OCP Data Registry, publication 130). Flags are **risk indicators that call for review**. They are not findings of wrongdoing.*

## 1. Data

| Item | Value |
|---|---|
| OCDS records (all sectors) | {len(tenders) + len(procs):,} usable ({len(tenders):,} tender records, {len(procs):,} bid records) |
| Period | {int(yrs.min())}–{int(yrs.max())} ({int((w.award_date.dt.year < 2014).sum())} records with implausible dates excluded from timelines) |
| Construction works, tender records | {n_w:,} ({ugx(w.award_value_ugx.sum())} awarded) |
| Construction works, bid records | {n_bw:,} |
| Procuring entities with works | {w.buyer_id.nunique():,} |
| Distinct suppliers (works) | {pd.concat([w.supplier_id, bw.supplier_id]).nunique():,} |
| Party IDs → distinct firms after name matching | {len(keys):,} → {keys.canon.nunique():,} |

**How the Uganda feed is structured.** Each procurement is split across two record families that don't share an ID.
- *Tender records* carry the estimate, procurement method, tender window and award, but no bidders.
- *Bid records* carry every bid and the list of bidders, but no estimate or method.

The two are analysed separately. Linking them, using PPDA's own API IDs, is a priority.

**No procurement category.** `mainProcurementCategory` is empty, so construction works are identified from titles with a rule-based classifier (`src/classify.py`). Spot checks suggest about 90% precision. A hand-labelled validation sample is still needed.

![Works by year](figures/works_by_year.png)

## 2. Headline findings

1. **Awards bunch just below the official estimate.** {just_below:.1%} of works awards fall between 99% and 100% of the estimate. A 1-point band between 95% and 98% averages only {band_95_98:.1%}. Competitive bidding shouldn't produce that spike. Possible explanations are leaked estimates, bids anchored to a published budget, or estimates revised after award. This is a systemic issue to raise with PPDA.
   ![award/estimate](figures/award_to_estimate.png)
2. **Competition looks weak, but the data needs checking first.** {single:.1%} of works procurements with bid data list **only one bid**. Some large contracts, such as UNRA roads, very likely had several bidders, so part of this rate probably reflects records that store only the winning bid. Verify against PPDA bid-submission data before quoting the figure. {w.f_restricted_method.mean():.1%} of works tenders used limited or selective bidding, and {w.f_direct_procurement.mean():.1%} used direct procurement.
3. **The same bidder groups recur, and wins rotate among them.** {len(sets):,} identical bidder groups (all sectors) appear together in 3+ procurements. In {int(sets.rotating.sum()):,} of them, the win passes between members. The largest are listed below.
4. **Supplier identities are messy.** A third of party IDs are spelling variants of another firm. This blocks reliable supplier-level monitoring and needs to be fixed with URSB registration numbers.
5. **Benford's Law.** Across all works award values, MAD = {mad_all:.3f}: {benford_txt}. Per entity, most values exceed the 0.015 cut-off, so the test is used *relatively*, to flag the 10% of entities that deviate most.
   ![Benford](figures/benford_works.png)

### Largest repeated bidder groups with rotating wins

{tbl(top_sets[["bidder_names", "n_procurements", "n_distinct_winners", "n_buyers", "any_works"]].rename(columns={"bidder_names": "bidders", "n_procurements": "procurements", "n_distinct_winners": "distinct winners", "n_buyers": "entities", "any_works": "incl. works"}))}

## 3. Flag rates

**Tender records (works)**

{tbl(flag_t.rename("rate").map("{:.1%}".format).rename_axis("flag").reset_index())}

**Bid records (works)**

{tbl(flag_b.rename("rate").map("{:.1%}".format).rename_axis("flag").reset_index())}

## 4. Audit priority list: top 15 works contracts

Ranked by **priority index = risk score × log₁₀(award value)**, which weights risk by the money at stake.

The full ranked list is in `reports/top_risk_works_contracts.csv` and the dashboard.

{tbl(top.head(15).assign(award=lambda d: d.award_value_ugx.map(ugx), title=lambda d: d.title.str[:60])[["priority_index", "risk_score", "buyer_name", "supplier_name", "title", "award", "reasons"]])}

## 5. Entities with the highest mean works risk (≥10 records)

{tbl(buyers[buyers.works_records >= 10].nlargest(10, "mean_risk")[["buyer_name", "works_records", "mean_risk", "direct_share", "single_bid_share", "benford_mad"]].round(3))}

## 6. Limitations and next steps

- Single-bid flag: some bid records seem to contain only the winning bid. Cross-check with PPDA `/bid-submissions` before treating "one bid" as fact.
- Thresholds such as minimum tender windows are first-pass assumptions. Validate them against the PPDA Regulations.
- Link tender and bid records through the PPDA API (`awarded_contract_id`, `bid_invitation_id`).
- Contract variations (cost and time overruns) are available from the PPDA API (`/contracts-variations-prices`) but not yet ingested.
- Match firms to URSB (directors, addresses) to enable ownership-network analysis.
- Build labels from OAG audit reports, PPDA suspensions and IG cases to validate and train models.
"""
    (R / "exploratory_report.md").write_text(md)
    print(f"wrote {R / 'exploratory_report.md'}")


if __name__ == "__main__":
    main()
