"""Experiment C: ML price benchmark ("hedonic" model) for all works contracts.

Predict log(award value) WITHOUT the official estimate, from:
  * title text (TF-IDF word 1-2 grams) - captures what is being built and its scale words
  * extracted quantities (stances, classrooms, km, boreholes, units) where present
  * procuring entity, tender year, method, sub-sector (one-hot)
Model: ridge regression on sparse features, 5-fold CROSS-FITTED (each contract predicted by a model
that never saw it). Residual e = log(actual) - log(predicted): e > log(2) means ~2x what comparable
work cost.

Checks: do residuals differ for bunched awards, rotating-group contracts, rule-flagged high-risk
contracts, and unit-cost outliers? This generalises unit-cost screening to every works contract.

Out: data/processed/price_residuals.parquet, paper/price_model.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import subsector  # noqa: E402
from unit_costs import extract  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


def build():
    t = pd.read_parquet(P / "tenders.parquet")
    f = pd.read_parquet(P / "works_ml_flags.parquet")
    w = t.merge(f, on="ocid")
    w = w[w.ml_works & (w.award_value_ugx > 1e5)].copy()
    w["y"] = np.log(w.award_value_ugx)
    w["year"] = w.tender_start.dt.year.where(w.tender_start.dt.year >= 2014).fillna(0).astype(int).astype(str)
    w["sub"] = w.title.map(subsector).fillna("other")
    w["method"] = w.method.fillna("na")
    q = w.title.map(extract)
    w["q_cat"] = q.map(lambda e: e[0] if e else "none")
    w["log_q"] = q.map(lambda e: np.log(e[1]) if e else 0.0)
    nums = w.title.fillna("").str.findall(r"(\d+(?:\.\d+)?)\s*km").map(lambda m: float(m[0]) if m else 0.0)
    w["log_km"] = np.log1p(nums)
    return w.reset_index(drop=True)


def features(tr, te):
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=3)
    cats = ["buyer_name", "year", "method", "sub", "q_cat"]
    Xtr = hstack([tf.fit_transform(tr.title.fillna("")), oh.fit_transform(tr[cats].astype(str)),
                  csr_matrix(tr[["log_q", "log_km"]].to_numpy())]).tocsr()
    Xte = hstack([tf.transform(te.title.fillna("")), oh.transform(te[cats].astype(str)),
                  csr_matrix(te[["log_q", "log_km"]].to_numpy())]).tocsr()
    return Xtr, Xte


def main():
    w = build()
    pred = np.zeros(len(w))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(w):
        Xtr, Xte = features(w.iloc[tr], w.iloc[te])
        pred[te] = Ridge(alpha=3.0).fit(Xtr, w.y.iloc[tr]).predict(Xte)
    w["pred"], w["resid"] = pred, w.y - pred
    r2 = 1 - ((w.y - pred) ** 2).sum() / ((w.y - w.y.mean()) ** 2).sum()
    mae_x = float(np.exp(np.median(np.abs(w.resid))))

    # join context
    scored = pd.read_parquet(P / "works_tenders_scored.parquet")[["ocid", "risk_score", "f_award_just_below_est"]]
    w = w.merge(scored, on="ocid", how="left")
    w["ratio"] = w.award_value_ugx / w.estimate_ugx.where(w.estimate_ugx > 0)
    w["bunched"] = w.ratio.between(0.99, 0.999999)
    w["over2x"] = w.resid > np.log(2)
    uc = pd.read_parquet(P / "unit_costs.parquet")[["ocid", "outlier"]]
    w = w.merge(uc.rename(columns={"outlier": "unit_outlier"}), on="ocid", how="left")
    # does the estimate also sit above the benchmark? (inflated-estimate channel)
    w["est_resid"] = np.log(w.estimate_ugx.where(w.estimate_ugx > 0)) - w.pred

    res = dict(
        n=int(len(w)), cv_r2=float(r2), typical_error_factor=mae_x,
        share_over_2x=float(w.over2x.mean()), n_over_2x=int(w.over2x.sum()),
        value_over_2x_ugx=float(w.loc[w.over2x, "award_value_ugx"].sum()),
        excess_over_benchmark_for_over2x_ugx=float((w.award_value_ugx - np.exp(w.pred))[w.over2x].sum()),
        median_resid_bunched=float(w[w.bunched].resid.median()),
        median_resid_not_bunched=float(w[~w.bunched & w.ratio.notna()].resid.median()),
        median_est_resid_bunched=float(w[w.bunched].est_resid.median()),
        median_est_resid_not_bunched=float(w[~w.bunched & w.ratio.notna()].est_resid.median()),
        corr_resid_risk=float(w[["resid", "risk_score"]].corr(method="spearman").iloc[0, 1]),
        unit_outlier_agreement=dict(
            share_over2x_among_unit_outliers=float(w[w.unit_outlier == True].over2x.mean()) if (w.unit_outlier == True).any() else None,  # noqa: E712
            share_over2x_among_unit_normal=float(w[w.unit_outlier == False].over2x.mean()) if (w.unit_outlier == False).any() else None),  # noqa: E712
        by_method=w.groupby("method").over2x.mean().round(3).to_dict(),
    )
    w[["ocid", "buyer_name", "title", "award_value_ugx", "pred", "resid", "over2x", "est_resid"]].to_parquet(P / "price_residuals.parquet", index=False)
    (ROOT / "paper" / "price_model.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
