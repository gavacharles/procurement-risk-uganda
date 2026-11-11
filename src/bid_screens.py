"""Experiment D: machine-learning bid-rigging screens (after Huber & Imhof, 2019).

Unit: tender with >= 3 bids (bid records, all sectors).
Features (from bid PRICES only):
  cv        coefficient of variation of bids
  spread    (max - min) / min
  kurt      kurtosis of bids
  skew      skewness of bids
  rd        relative distance: (b2 - b1) / sd(losing bids)   (cover bidding -> b1 isolated, losers clustered)
  diff_pct  (b2 - b1) / b1
  n_bids, log_mean
Distant labels (from bidder IDENTITY, an independent channel): tender run by a repeated bidder group
whose wins rotate (bidder_sets.rotating) -> 1, else 0.
Model: random forest; grouped CV by procuring entity (no entity in both train and test). Metric: ROC AUC.
If AUC >> 0.5, price patterns alone recognise rotation tenders -> screens can flag collusion where
identities don't repeat. Then score unlabeled tenders.

Out: paper/bid_screens.json, data/processed/bid_screen_scores.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


def screens(v: np.ndarray) -> dict:
    v = np.sort(v[v > 0])
    if len(v) < 3:
        return {}
    losers = v[1:]
    sd_l = losers.std(ddof=1) if len(losers) > 1 else np.nan
    return dict(n_bids=len(v), log_mean=float(np.log(v.mean())), cv=float(v.std(ddof=1) / v.mean()),
                spread=float((v[-1] - v[0]) / v[0]), kurt=float(stats.kurtosis(v)) if len(v) > 3 else 0.0,
                skew=float(stats.skew(v)), rd=float((v[1] - v[0]) / sd_l) if sd_l and sd_l > 0 else np.nan,
                diff_pct=float((v[1] - v[0]) / v[0]))


def main():
    bids = pd.read_parquet(P / "bids.parquet")
    procs = pd.read_parquet(P / "bid_procs.parquet")[["ocid", "buyer_id"]]
    tdr = pd.read_parquet(P / "tenderers.parquet")
    sets = pd.read_parquet(P / "bidder_sets.parquet")
    rot = set(sets[sets.rotating].bidder_set)
    bset = tdr.groupby("ocid").party_id.apply(lambda s: ",".join(sorted(set(s.dropna()))))
    label = bset.isin(rot).astype(int).rename("rot")

    rows = {}
    for oc, s in bids.groupby("ocid").bid_value_ugx:
        f = screens(s.to_numpy(dtype=float))
        if f:
            rows[oc] = f
    F = pd.DataFrame.from_dict(rows, orient="index").dropna(subset=["cv"])
    F = F.join(label, how="inner").join(procs.set_index("ocid"), how="left")
    F = F.replace([np.inf, -np.inf], np.nan)
    cols = ["n_bids", "log_mean", "cv", "spread", "kurt", "skew", "rd", "diff_pct"]
    F[cols] = F[cols].fillna(F[cols].median())
    y, g = F.rot.to_numpy(), F.buyer_id.fillna("na").to_numpy()
    print(f"tenders with >=3 bids: {len(F):,}; rotation-labelled: {y.sum():,}; entities: {len(set(g))}", flush=True)

    oof = np.full(len(F), np.nan)
    for tr, te in GroupKFold(5).split(F, y, g):
        if y[tr].sum() < 10:
            continue
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1)
        m.fit(F.iloc[tr][cols], y[tr])
        oof[te] = m.predict_proba(F.iloc[te][cols])[:, 1]
    ok = ~np.isnan(oof)
    auc = roc_auc_score(y[ok], oof[ok]) if len(set(y[ok])) == 2 else None
    # a price-only model WITHOUT n_bids/log_mean (pure distribution screens)
    pure = ["cv", "spread", "kurt", "skew", "rd", "diff_pct"]
    oof2 = np.full(len(F), np.nan)
    for tr, te in GroupKFold(5).split(F, y, g):
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1)
        m.fit(F.iloc[tr][pure], y[tr])
        oof2[te] = m.predict_proba(F.iloc[te][pure])[:, 1]
    auc_pure = roc_auc_score(y, oof2)
    full = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1).fit(F[cols], y)
    imp = permutation_importance(full, F[cols], y, n_repeats=5, random_state=0, scoring="roc_auc")
    F["screen_score"] = full.predict_proba(F[cols])[:, 1]
    top_unlabelled = F[F.rot == 0].screen_score.quantile(0.99)
    res = dict(n=int(len(F)), positives=int(y.sum()), grouped_cv_auc=float(auc) if auc else None,
               grouped_cv_auc_pure_screens=float(auc_pure),
               importance=dict(sorted(zip(cols, imp.importances_mean.round(4)), key=lambda x: -x[1])),
               mean_screens_rot_vs_not=F.groupby("rot")[pure].median().round(4).to_dict("index"),
               unlabelled_flagged_top1pct=int((F[F.rot == 0].screen_score >= top_unlabelled).sum()))
    F.rename_axis("ocid").reset_index()[["ocid", "rot", "screen_score"] + cols].to_parquet(P / "bid_screen_scores.parquet", index=False)
    (ROOT / "paper" / "bid_screens.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
