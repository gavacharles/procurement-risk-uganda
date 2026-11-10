"""Machine-learning and NLP layer.

1. Isolation Forest (unsupervised anomaly detection) on works tender records.
   Learns what a "normal" works contract looks like from ~5k contracts and scores how
   isolated each one is. Compared against the rule-based risk score: agreement shows
   the rules capture real structure, disagreement surfaces cases the rules miss.

2. NLP contract-splitting detector (TF-IDF + cosine similarity).
   Splitting a project into several smaller contracts keeps each one under a method
   threshold (e.g., to avoid open bidding). Signature: same entity, near-identical
   titles, awarded within a short window, combined value much larger than each part.

3. (Placeholder) Supervised model: needs labels (OAG findings, IG cases). Not trained
   yet -- PPDA suspensions (n=43, mostly document forgery) are too few and the wrong
   fraud type for collusion screens.

Outputs: data/processed/ml_anomaly.parquet, data/processed/split_candidates.parquet,
         paper/ml_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "paper"

FEATURES = ["log_value", "log_est", "award_to_est_c", "tender_days_c", "is_direct", "is_restricted",
            "supplier_share_of_buyer", "awards_12m_same_pair", "supplier_age_days", "buyer_n_works"]


def features(w: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=w.index)
    X["log_value"] = np.log10(w.award_value_ugx.clip(lower=1e5))
    X["log_est"] = np.log10(w.estimate_ugx.clip(lower=1e5))
    X["award_to_est_c"] = w.award_to_est.clip(0, 3)
    X["tender_days_c"] = w.tender_days.clip(0, 365)
    X["is_direct"] = (w.method == "direct").astype(int)
    X["is_restricted"] = w.method.isin(["limited", "selective"]).astype(int)
    X["supplier_share_of_buyer"] = w.supplier_share_of_buyer
    X["awards_12m_same_pair"] = w.awards_12m_same_pair
    first = w.groupby("supplier_id").award_date.transform("min")
    X["supplier_age_days"] = (w.award_date - first).dt.days.clip(lower=0)
    X["buyer_n_works"] = w.groupby("buyer_id").ocid.transform("count")
    return X.fillna(X.median(numeric_only=True))


def isolation_forest(w: pd.DataFrame) -> dict:
    ok = w.award_value_ugx.notna() & (w.award_value_ugx > 0)
    d = w[ok].copy()
    X = features(d)[FEATURES]
    model = IsolationForest(n_estimators=500, contamination="auto", random_state=42).fit(X)
    d["ml_anomaly"] = -model.score_samples(X)  # higher = more anomalous
    d["ml_anomaly_pct"] = d.ml_anomaly.rank(pct=True)

    rho, pval = spearmanr(d.ml_anomaly, d.risk_score)
    top_ml = set(d.nlargest(int(len(d) * 0.1), "ml_anomaly").ocid)
    top_rule = set(d.nlargest(int(len(d) * 0.1), "risk_score").ocid)
    overlap = len(top_ml & top_rule) / len(top_ml)

    # what drives anomalies: mean feature z-score in top decile vs rest
    z = (X - X.mean()) / X.std()
    drivers = z[d.ocid.isin(top_ml).values].mean().sort_values(ascending=False)

    d[["ocid", "ml_anomaly", "ml_anomaly_pct", "risk_score"]].to_parquet(P / "ml_anomaly.parquet", index=False)
    return dict(n=int(len(d)), features=FEATURES, spearman_rho=float(rho), spearman_p=float(pval),
                top_decile_overlap=float(overlap), expected_overlap_if_random=0.10,
                anomaly_drivers_top_decile_z=drivers.round(2).to_dict(),
                ml_only_top_decile=int(len(top_ml - top_rule)))


def contract_splitting(w: pd.DataFrame, sim=0.85, window_days=60) -> dict:
    d = w.dropna(subset=["title", "award_date"]).copy()
    d["t"] = d.title.str.lower().str.replace(r"\d+", " ", regex=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    rows = []
    for buyer, g in d.groupby("buyer_id"):
        if len(g) < 2:
            continue
        M = vec.fit_transform(g.t)
        S = cosine_similarity(M)
        idx = g.index.to_list()
        dates = g.award_date.to_numpy()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if S[i, j] >= sim:
                    gap = abs((dates[i] - dates[j]) / np.timedelta64(1, "D"))
                    if gap <= window_days:
                        a, b = g.loc[idx[i]], g.loc[idx[j]]
                        rows.append(dict(buyer_id=buyer, buyer_name=a.buyer_name, ocid_a=a.ocid, ocid_b=b.ocid,
                                         title_a=a.title, title_b=b.title, similarity=float(S[i, j]), gap_days=gap,
                                         value_a=a.award_value_ugx, value_b=b.award_value_ugx,
                                         same_supplier=a.supplier_id == b.supplier_id,
                                         method_a=a.method, method_b=b.method))
    sp = pd.DataFrame(rows)
    if sp.empty:
        return dict(pairs=0)
    lot = sp.title_a.str.contains(r"\blot\b", case=False) | sp.title_b.str.contains(r"\blot\b", case=False)
    dup = sp.same_supplier & np.isclose(sp.value_a, sp.value_b, rtol=0.001)
    open_both = sp.method_a.eq("open") & sp.method_b.eq("open")
    sp["category"] = np.select(
        [dup, lot & open_both, sp.same_supplier | ~open_both],
        ["duplicate_record", "declared_lots_open", "possible_split"], default="similar_open_tenders")
    sp.to_parquet(P / "split_candidates.parquet", index=False)
    return dict(pairs=int(len(sp)), entities=int(sp.buyer_id.nunique()),
                by_category=sp.category.value_counts().to_dict(),
                possible_split_entities=int(sp[sp.category == "possible_split"].buyer_id.nunique()),
                same_supplier_pairs=int(sp.same_supplier.sum()),
                both_non_open=int((~sp.method_a.eq("open") & ~sp.method_b.eq("open")).sum()),
                contracts_involved=int(len(set(sp.ocid_a) | set(sp.ocid_b))),
                params=dict(cosine_threshold=sim, window_days=window_days))


def main() -> None:
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    res = dict(isolation_forest=isolation_forest(w), contract_splitting=contract_splitting(w),
               supervised="not trained: awaiting labelled cases (OAG/IG); PPDA suspensions n=43 are mostly forgery")
    OUT.mkdir(exist_ok=True)
    (OUT / "ml_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
