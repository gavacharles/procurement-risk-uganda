"""Experiment B: machine-learning works classifier trained on PPDA's official procurement type.

Labels: procurement_type == "WRKS" from PPDA /bid-invitations (random sample + main crawl, deduplicated).
Model: TF-IDF (word 1-2 grams + character 3-5 grams) -> logistic regression (class-balanced).
Evaluation: 5-fold stratified cross-validation; compared with the rule-based classifier (src/classify.py).
Application: score every OCDS tender title; re-estimate the headline bunching on the ML works sample.

Out: paper/ml_classifier.json, data/processed/works_ml_flags.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import bunching  # noqa: E402
from classify import is_works  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "ppda_api"


def load_labels() -> pd.DataFrame:
    parts = [pd.read_json(f, lines=True, dtype=False) for f in [RAW / "bid_invitations_sample.jsonl", RAW / "bid_invitations.jsonl"]
             if f.exists() and f.stat().st_size]
    d = pd.concat(parts).drop_duplicates("id")
    d = d.dropna(subset=["subject", "procurement_type"])
    d = d[d.subject.str.len() > 5].drop_duplicates("subject")
    d["y"] = (d.procurement_type == "WRKS").astype(int)
    return d[["subject", "y", "procurement_type"]].reset_index(drop=True)


class Model:
    def __init__(self):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, lowercase=True)
        self.m = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)

    def fit(self, x, y):
        self.m.fit(hstack([self.w.fit_transform(x), self.c.fit_transform(x)]), y)
        return self

    def proba(self, x):
        return self.m.predict_proba(hstack([self.w.transform(x), self.c.transform(x)]))[:, 1]


def prf(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return dict(precision=float(pr), recall=float(rc), f1=float(f1))


def main():
    d = load_labels()
    print(f"labelled titles: {len(d):,} (works {d.y.sum():,})", flush=True)
    oof = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(d.subject, d.y):
        oof[te] = Model().fit(d.subject.iloc[tr], d.y.iloc[tr]).proba(d.subject.iloc[te])
    rule = d.subject.map(is_works).astype(int)
    res = dict(n=int(len(d)), works=int(d.y.sum()), rule_based=prf(d.y, rule))
    best = max(((t, prf(d.y, (oof >= t).astype(int))["f1"]) for t in np.arange(0.2, 0.9, 0.05)), key=lambda x: x[1])
    res["ml_threshold"] = float(best[0])
    res["ml"] = prf(d.y, (oof >= best[0]).astype(int))
    res["ml_at_0.5"] = prf(d.y, (oof >= 0.5).astype(int))
    print(json.dumps(res, indent=1), flush=True)

    # errors the ML fixes / makes (for the paper)
    d["oof"], d["rule"] = oof, rule
    fixed = d[(d.y == 1) & (d.rule == 0) & (d.oof >= best[0])].subject.head(10).tolist()
    res["examples_recovered_by_ml"] = fixed

    # apply to all OCDS tender titles; re-run headline bunching on the ML works sample
    full = Model().fit(d.subject, d.y)
    t = pd.read_parquet(P / "tenders.parquet")
    t["ml_p"] = full.proba(t.title.fillna(""))
    t["ml_works"] = t.ml_p >= best[0]
    t["rule_works"] = t.title.map(is_works)
    t[["ocid", "ml_p", "ml_works", "rule_works"]].to_parquet(P / "works_ml_flags.parquet", index=False)
    r = t.award_value_ugx / t.estimate_ugx.where(t.estimate_ugx > 0)
    out = {}
    for name, mask in [("rule_works", t.rule_works), ("ml_works", t.ml_works), ("ml_only_new", t.ml_works & ~t.rule_works)]:
        rr = r[mask].dropna()
        b = bunching(rr, n_boot=200) if len(rr[(rr >= .8) & (rr < 1.2)]) > 400 else None
        out[name] = dict(n_tenders=int(mask.sum()), n_ratio=int(len(rr)),
                         share_99_100=float(rr.between(0.99, 0.999999).mean()),
                         bunching_b=(b["excess_ratio"] if b else None), ci95=(b["ci95"] if b else None))
    res["bunching_by_sample"] = out
    (ROOT / "paper" / "ml_classifier.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
