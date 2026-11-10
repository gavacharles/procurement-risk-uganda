"""Paper 1 mechanism test: when was the estimate recorded, and does that predict bunching?

Data: PPDA /bid-invitations (invitation-stage estimate, portal entry timestamp, process dates,
official procurement type), joined to OCDS tender records (award value) on the OCDS id.

Tests
  T1  Same estimate?  Invitation-stage estimate vs the estimate in the OCDS award-stage record.
  T2  Entry timing.   Was the invitation (with its estimate) entered on the portal before bid
                      opening ("ex ante") or only after the contract award ("ex post")?
                      If estimates are back-filled once the award is known, ex-post entries
                      should bunch more just below the estimate.
  T3  Later edits.    Was the record updated after the award date?
  V   Classifier validation: PPDA's official WRKS type vs our title-based works classifier.

Uses the random-page sample (target 500 pages ≈ 5,000 records) by default. Writes
paper/estimate_timing.json and paper/tables/t6_estimate_timing.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import is_works  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ppda_api"
P = ROOT / "data" / "processed"
OUT = ROOT / "paper"


def load_invitations(sample_only: bool = True) -> pd.DataFrame:
    """Default: the random-page sample only (a clean random design across all years).
    sample_only=False also adds the newest-first main crawl, which over-represents recent years."""
    files = [RAW / "bid_invitations_sample.jsonl"] + ([] if sample_only else [RAW / "bid_invitations.jsonl"])
    parts = [pd.read_json(f, lines=True, dtype=False) for f in files if f.exists() and f.stat().st_size]
    bi = pd.concat(parts, ignore_index=True).drop_duplicates("id", keep="last")
    for c in ["created_at", "updated_at", "bid_openning_date", "bid_submission_deadline_date", "contract_award_date",
              "invitation_to_bid_date", "display_of_beb_notice_date", "contract_signature_date"]:
        bi[c] = pd.to_datetime(bi[c], errors="coerce", utc=True)
    bi["est_inv"] = pd.to_numeric(bi.estimated_amount, errors="coerce") * pd.to_numeric(
        bi.estimated_amount_exchange_rate, errors="coerce").fillna(1)
    return bi


def share_bunched(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.between(0.99, 0.999999).mean()) if len(r) else float("nan")


def boot_diff(a: pd.Series, b: pd.Series, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = a.dropna().between(0.99, 0.999999).to_numpy(), b.dropna().between(0.99, 0.999999).to_numpy()
    if len(a) < 10 or len(b) < 10:
        return None
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main() -> None:
    bi = load_invitations()
    res: dict = {"design": "random sample of PPDA /bid-invitations pages (all years)",
                 "invitations_loaded": int(len(bi)),
                 "financial_years": bi.financial_year.value_counts().sort_index().to_dict()}

    # ---- V: classifier validation on official type
    v = bi.dropna(subset=["subject", "procurement_type"]).copy()
    v["pred"] = v.subject.map(is_works)
    v["true"] = v.procurement_type.eq("WRKS")
    tp, fp, fn = int((v.pred & v.true).sum()), int((v.pred & ~v.true).sum()), int((~v.pred & v.true).sum())
    res["classifier_validation"] = dict(n=int(len(v)), true_works=int(v.true.sum()), tp=tp, fp=fp, fn=fn,
                                        precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))
    fps = v[v.pred & ~v.true][["subject", "procurement_type"]].head(15)
    fns = v[~v.pred & v.true][["subject"]].head(15)

    # ---- join to OCDS tender records (award values) on OCDS id
    # join key: PPDA bid-invitation id == OCDS tender.id (ocds_id rarely matches the OCDS ocid)
    t = pd.read_parquet(P / "tenders.parquet")[["ocid", "tender_id", "estimate_ugx", "award_value_ugx", "method", "award_date"]]
    t = t.assign(tender_id=t.tender_id.astype(str)).drop_duplicates("tender_id")
    j = bi.assign(tid=bi.id.astype(str)).merge(t, left_on="tid", right_on="tender_id", how="inner")
    res["joined_to_ocds"] = int(len(j))
    w = j[j.procurement_type.eq("WRKS")].copy()
    res["joined_works"] = int(len(w))
    res["plan_estimate_populated"] = int(pd.to_numeric(bi.plan_estimated_amount, errors="coerce").notna().sum())
    res["works_entered_after_award_share"] = None  # filled below

    # ---- T1: same estimate?
    both = w.dropna(subset=["est_inv", "estimate_ugx"])
    both = both[(both.est_inv > 0) & (both.estimate_ugx > 0)]
    rel = (both.estimate_ugx / both.est_inv)
    res["T1_estimate_identity"] = dict(n=int(len(both)), share_identical=float(np.isclose(rel, 1, rtol=1e-6).mean()),
                                       share_within_1pct=float(rel.between(0.99, 1.01).mean()))

    # ---- award / invitation-estimate ratio
    w["ratio_inv"] = w.award_value_ugx / w.est_inv.where(w.est_inv > 0)

    # ---- T2: entry timing
    w["entered_vs_opening_days"] = (w.created_at - w.bid_openning_date).dt.days
    award_ref = w.contract_award_date.fillna(w.award_date)
    w["entered_vs_award_days"] = (w.created_at - award_ref).dt.days
    w["timing"] = np.select(
        [w.entered_vs_opening_days <= 0, w.entered_vs_award_days > 0],
        ["ex_ante (entered before bid opening)", "ex_post (entered after award)"],
        default="between opening and award")
    g = w.groupby("timing").agg(n=("ratio_inv", "size"), n_with_ratio=("ratio_inv", "count"), share_99_100=("ratio_inv", share_bunched),
                                share_exact=("ratio_inv", lambda r: float(np.isclose(r.dropna(), 1).mean()) if r.notna().any() else np.nan),
                                median_ratio=("ratio_inv", "median"))
    res["T2_by_entry_timing"] = g.reset_index().to_dict("records")
    res["works_entered_after_award_share"] = float(w.timing.str.startswith("ex_post").mean())
    res["works_entered_after_award_by_fy"] = w.groupby("financial_year").timing.apply(
        lambda s: round(float(s.str.startswith("ex_post").mean()), 3)).to_dict()
    exa = w[w.timing.str.startswith("ex_ante")].ratio_inv
    exp = w[w.timing.str.startswith("ex_post")].ratio_inv
    res["T2_diff_expost_minus_exante_share_99_100"] = dict(
        point=float(share_bunched(exp) - share_bunched(exa)) if len(exa) and len(exp) else None,
        ci95=boot_diff(exp, exa))

    # ---- T3: later edits
    w["edited_after_award"] = w.updated_at > award_ref + pd.Timedelta(days=1)
    g3 = w.groupby("edited_after_award").agg(n=("ratio_inv", "size"), n_with_ratio=("ratio_inv", "count"), share_99_100=("ratio_inv", share_bunched))
    res["T3_by_edit_after_award"] = g3.reset_index().to_dict("records")

    # ---- process timeline (Paper 2): stage medians
    stages = dict(
        invitation_to_deadline=(w.bid_submission_deadline_date - pd.to_datetime(w.invitation_to_bid_date, utc=True, errors="coerce")).dt.days,
        deadline_to_beb_notice=(w.display_of_beb_notice_date - w.bid_submission_deadline_date).dt.days,
        beb_notice_to_award=(w.contract_award_date - w.display_of_beb_notice_date).dt.days,
        award_to_signature=(w.contract_signature_date - w.contract_award_date).dt.days,
    )
    res["stage_medians_days_works"] = {k: (float(s[(s >= 0) & (s < 1500)].median()) if s.notna().any() else None)
                                       for k, s in stages.items()}

    OUT.mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    g.to_csv(OUT / "tables" / "t6_estimate_timing.csv")
    (OUT / "estimate_timing.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    print("\nclassifier false positives (sample):\n", fps.to_string(index=False))
    print("\nclassifier false negatives (sample):\n", fns.to_string(index=False))


if __name__ == "__main__":
    main()
