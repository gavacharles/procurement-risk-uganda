"""Explainable red-flag engine for Ugandan public construction (works) procurement.

Every flag is a boolean column `f_<name>` with a weight in FLAGS. A record's
risk score is the weighted sum of fired flags, scaled to 0-100, and the
`reasons` column lists the fired flags in plain language.

Thresholds are first-pass assumptions; each one marked (VALIDATE) must be
checked against the PPDA Act / Regulations and with practitioners.

Outputs (data/processed/):
  works_tenders_scored.parquet   tender records (works) with flags + score
  works_bids_scored.parquet      bid records (works) with flags + score
  bidder_sets.parquet            repeated identical bidder groups (all sectors)
  supplier_profiles.parquet      supplier-level indicators
  buyer_profiles.parquet         procuring-entity indicators incl. Benford MAD
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import tag  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

# name: (weight, plain-language reason)
FLAGS: dict[str, tuple[float, str]] = {
    # --- tender records ---
    "direct_procurement": (3, "Direct procurement (no competition) for works"),
    "restricted_method": (1, "Restricted method (limited/selective bidding)"),
    "short_window": (2, "Bidding window shorter than expected minimum"),
    "award_just_below_est": (1, "Award within 1% below the official estimate (possible leaked estimate)"),
    "award_above_est": (2, "Award more than 10% above the official estimate"),
    "contract_above_award": (2, "Contract value more than 10% above award value"),
    "high_value_outlier": (1, "Award value an extreme outlier for this sub-sector"),
    "new_supplier_big_award": (2, "Supplier's first appearance in the data is a large award"),
    "buyer_supplier_concentration": (2, "Supplier won >40% of this entity's works value"),
    "repeat_award_same_buyer": (1, "Supplier won 3+ works awards from same entity within 12 months"),
    "missing_key_data": (1, "Key transparency fields missing (award value, dates)"),
    # --- bid records ---
    "single_bidder": (3, "Only one bid received"),
    "winner_not_lowest": (2, "Award higher than the lowest bid"),
    "bids_too_close": (2, "Bids suspiciously close together (coefficient of variation <3%)"),
    "repeated_bidder_set": (2, "Same group of bidders appears together in 3+ procurements"),
    "rotating_winners": (3, "Repeated bidder group where the win rotates among members"),
    "name_variant_cobid": (1, "Two bidders look like name-variants of the same firm (review)"),
    "buyer_benford_nonconform": (1, "Entity's award values deviate strongly from Benford's Law (top 10% of entities)"),
}

# Data start in 2015: a supplier's "first appearance" before this is not meaningful.
BURN_IN = pd.Timestamp("2017-01-01", tz="UTC")
SHORT_WINDOW_DAYS ={"open": 21, "selective": 10, "limited": 10, "direct": 0}  # (VALIDATE)


def _days(a, b):
    d = (b - a).dt.days
    return d.where(d >= 0)


def benford_mad(values: pd.Series) -> float | None:
    v = values[(values > 0) & values.notna()]
    if len(v) < 50:
        return None
    first = v.astype("float").map(lambda x: int(f"{x:e}"[0]))
    obs = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    exp = np.log10(1 + 1 / np.arange(1, 10))
    return float(np.mean(np.abs(obs.values - exp)))


def score(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("f_")]
    max_w = sum(FLAGS[c[2:]][0] for c in cols)
    raw = sum(df[c].fillna(False).astype(float) * FLAGS[c[2:]][0] for c in cols)
    df["risk_raw"] = raw
    df["risk_score"] = (100 * raw / max_w).round(1)
    df["n_flags"] = df[cols].fillna(False).sum(axis=1)
    df["reasons"] = df[cols].fillna(False).apply(
        lambda r: "; ".join(FLAGS[c[2:]][1] for c in cols if r[c]), axis=1
    )
    return df


def main() -> None:
    tenders = tag(pd.read_parquet(P / "tenders.parquet"))
    procs = tag(pd.read_parquet(P / "bid_procs.parquet"))
    bids = pd.read_parquet(P / "bids.parquet")
    tdr = pd.read_parquet(P / "tenderers.parquet")

    # ---------- supplier first appearance (all sectors, both families) ----------
    allaw = pd.concat(
        [tenders[["supplier_id", "award_date", "buyer_id", "award_value_ugx"]],
         procs[["supplier_id", "award_date", "buyer_id", "award_value_ugx"]]]
    ).dropna(subset=["supplier_id"])
    first_seen = allaw.groupby("supplier_id")["award_date"].min()

    # ---------- buyer Benford (all awards of the entity) ----------
    buyer_mad = allaw.groupby("buyer_id")["award_value_ugx"].apply(benford_mad).rename("benford_mad")
    # Nigrini's absolute cut-off (MAD > 0.015) flags ~all entities because procurement
    # values are habitually rounded, so use a relative cut-off: top decile of entities.
    benford_bad = set(buyer_mad[buyer_mad > buyer_mad.quantile(0.90)].index)

    # =============== TENDER RECORDS (works) ===============
    w = tenders[tenders.is_works].copy()
    w["tender_days"] = _days(w.tender_start, w.tender_end)
    est = w.estimate_ugx.where(w.estimate_ugx > 0)
    w["award_to_est"] = w.award_value_ugx / est
    w["contract_to_award"] = w.contract_value_ugx / w.award_value_ugx.where(w.award_value_ugx > 0)

    w["f_direct_procurement"] = w.method.eq("direct")
    w["f_restricted_method"] = w.method.isin(["limited", "selective"])
    thr = w.method.map(SHORT_WINDOW_DAYS)
    w["f_short_window"] = (w.tender_days < thr) & w.tender_days.notna()
    w["f_award_just_below_est"] = w.award_to_est.between(0.99, 0.999999, inclusive="left")
    w["f_award_above_est"] = w.award_to_est > 1.10
    w["f_contract_above_award"] = w.contract_to_award > 1.10

    lv = np.log10(w.award_value_ugx.where(w.award_value_ugx > 0))
    grp = lv.groupby(w.subsector)
    z = (lv - grp.transform("median")) / (grp.transform(lambda s: (s - s.median()).abs().median()) * 1.4826)
    w["f_high_value_outlier"] = z > 3.5

    big = w.award_value_ugx > w.award_value_ugx.quantile(0.75)
    fs = w.supplier_id.map(first_seen)
    w["f_new_supplier_big_award"] = big & (w.award_date <= fs + pd.Timedelta(days=1)) & (
        w.award_date >= BURN_IN)

    tot = w.groupby("buyer_id").award_value_ugx.transform("sum")
    n_b = w.groupby("buyer_id").award_value_ugx.transform("count")
    sup_share = w.groupby(["buyer_id", "supplier_id"]).award_value_ugx.transform("sum") / tot
    w["supplier_share_of_buyer"] = sup_share
    w["f_buyer_supplier_concentration"] = (sup_share > 0.4) & (n_b >= 5)

    w = w.sort_values("award_date")
    def _rolling_count(g):
        s = pd.Series(1, index=g.award_date)
        return s.rolling("365D").sum().values
    w["awards_12m_same_pair"] = np.nan
    ok = w.award_date.notna() & w.supplier_id.notna()
    w.loc[ok, "awards_12m_same_pair"] = (
        w[ok].groupby(["buyer_id", "supplier_id"], group_keys=False)
        .apply(lambda g: pd.Series(_rolling_count(g), index=g.index))
    )
    w["f_repeat_award_same_buyer"] = w.awards_12m_same_pair >= 3
    w["f_missing_key_data"] = w.award_value_ugx.isna() | w.tender_start.isna() | w.tender_end.isna()
    w["f_buyer_benford_nonconform"] = w.buyer_id.isin(benford_bad)
    w = score(w)
    w.to_parquet(P / "works_tenders_scored.parquet", index=False)

    # =============== BIDDER SETS (all sectors) ===============
    sets = (
        tdr.groupby("ocid").party_id.apply(lambda s: tuple(sorted(set(s.dropna()))))
        .rename("bidder_set").reset_index()
    )
    sets = sets[sets.bidder_set.map(len) >= 2]
    sets = sets.merge(procs[["ocid", "supplier_id", "buyer_id", "award_date", "title", "is_works"]], on="ocid")
    agg = sets.groupby("bidder_set").agg(
        n_procurements=("ocid", "nunique"),
        n_distinct_winners=("supplier_id", "nunique"),
        n_buyers=("buyer_id", "nunique"),
        any_works=("is_works", "any"),
        first=("award_date", "min"),
        last=("award_date", "max"),
    ).reset_index()
    agg["set_size"] = agg.bidder_set.map(len)
    agg["rotating"] = (agg.n_procurements >= 3) & (agg.n_distinct_winners >= 2)
    agg = agg[agg.n_procurements >= 3].sort_values("n_procurements", ascending=False)
    names = tdr.drop_duplicates("party_id").set_index("party_id").party_name
    agg["bidder_names"] = agg.bidder_set.map(lambda t: " | ".join(str(names.get(i, i)) for i in t))
    agg["bidder_set"] = agg.bidder_set.map(lambda t: ",".join(t))
    agg.to_parquet(P / "bidder_sets.parquet", index=False)
    rep = sets.merge(agg[["bidder_set", "rotating"]].assign(
        bidder_set=lambda d: d.bidder_set.str.split(",").map(tuple)), on="bidder_set", how="inner")
    repeated_ocids = set(rep.ocid)
    rotating_ocids = set(rep.loc[rep.rotating, "ocid"])

    # =============== BID RECORDS (works) ===============
    bw = procs[procs.is_works].copy()
    bstat = bids.groupby("ocid").bid_value_ugx.agg(
        min_bid="min", max_bid="max", mean_bid="mean", std_bid="std", n_valid="count")
    bw = bw.merge(bstat, left_on="ocid", right_index=True, how="left")
    bw["bid_cv"] = bw.std_bid / bw.mean_bid
    bw["f_single_bidder"] = bw.n_bids == 1
    bw["f_winner_not_lowest"] = (bw.n_valid >= 2) & (bw.award_value_ugx > bw.min_bid * 1.01)
    bw["f_bids_too_close"] = (bw.n_valid >= 3) & (bw.bid_cv < 0.03)
    bw["f_repeated_bidder_set"] = bw.ocid.isin(repeated_ocids)
    bw["f_rotating_winners"] = bw.ocid.isin(rotating_ocids)
    vpath = P / "variant_cobids.parquet"
    variant_ocids = set(pd.read_parquet(vpath).ocid) if vpath.exists() else set()
    bw["f_name_variant_cobid"] = bw.ocid.isin(variant_ocids)
    fs = bw.supplier_id.map(first_seen)
    bw["f_new_supplier_big_award"] = (bw.award_value_ugx > bw.award_value_ugx.quantile(0.75)) & (
        bw.award_date <= fs + pd.Timedelta(days=1)) & (bw.award_date >= BURN_IN)
    bw["f_buyer_benford_nonconform"] = bw.buyer_id.isin(benford_bad)
    bw = score(bw)
    bw.to_parquet(P / "works_bids_scored.parquet", index=False)

    # =============== PROFILES ===============
    both = pd.concat([
        w[["supplier_id", "supplier_name", "buyer_id", "award_value_ugx", "risk_score"]],
        bw[["supplier_id", "supplier_name", "buyer_id", "award_value_ugx", "risk_score"]],
    ]).dropna(subset=["supplier_id"])
    sup = both.groupby("supplier_id").agg(
        supplier_name=("supplier_name", "first"),
        works_awards=("risk_score", "size"),
        works_value_ugx=("award_value_ugx", "sum"),
        n_buyers=("buyer_id", "nunique"),
        mean_risk=("risk_score", "mean"),
        max_risk=("risk_score", "max"),
    )
    sup["first_seen"] = sup.index.map(first_seen)
    sup.reset_index().to_parquet(P / "supplier_profiles.parquet", index=False)

    buy = pd.concat([
        w[["buyer_id", "buyer_name", "award_value_ugx", "risk_score", "f_direct_procurement"]],
        bw[["buyer_id", "buyer_name", "award_value_ugx", "risk_score", "f_single_bidder"]],
    ]).groupby("buyer_id").agg(
        buyer_name=("buyer_name", "first"),
        works_records=("risk_score", "size"),
        works_value_ugx=("award_value_ugx", "sum"),
        mean_risk=("risk_score", "mean"),
        direct_share=("f_direct_procurement", "mean"),
        single_bid_share=("f_single_bidder", "mean"),
    )
    buy = buy.join(buyer_mad, how="left")
    buy.reset_index().to_parquet(P / "buyer_profiles.parquet", index=False)

    print(f"works tenders scored: {len(w):,}  | works bid records scored: {len(bw):,}")
    print(f"repeated bidder sets (>=3 procs, all sectors): {len(agg):,} | rotating: {int(agg.rotating.sum()):,}")
    for name, df in [("tenders", w), ("bids", bw)]:
        fl = [c for c in df.columns if c.startswith("f_")]
        print(f"\nflag rates ({name}):")
        print(df[fl].fillna(False).mean().sort_values(ascending=False).map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
