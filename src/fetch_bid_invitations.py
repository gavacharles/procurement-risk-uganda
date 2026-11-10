"""Resumable, polite crawl of PPDA public /bid-invitations (≈95k records, 10 per page).

Each record carries the invitation-stage estimate, PPDA's official procurement type,
the procurement method (with its statutory bidding period), every process date
(initiation -> invitation -> deadline -> evaluation -> BEB notice -> award -> signature),
the OCDS id, and `created_at` (when the record was entered on the portal).

Uses:
  Paper 1: was the estimate entered before or after the award was known?
  Paper 2: stage-by-stage procurement timelines.
  Classifier validation: official WRKS type vs our title classifier.

Run:  python src/fetch_bid_invitations.py        (resumes where it stopped)
Out:  data/raw/ppda_api/bid_invitations.jsonl   (one compact row per record)
      data/processed/bid_invitations.parquet     (deduplicated, written at the end)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ppda_api"
OUT = RAW / "bid_invitations.jsonl"
STATE = RAW / "bid_invitations.state"
API = "https://cdn.ppda.go.ug/api/bid-invitations"
UA = "procurement-risk-research (python urllib)"
PAUSE = 1.0

KEEP = ["id", "ocds_id", "pde_id", "financial_year", "procurement_reference_no", "procurement_plan_entry_id",
        "procurement_method_id", "estimated_amount", "estimated_amount_exchange_rate", "estimated_amount_currency_id",
        "initiation_date", "confirmation_of_funds_date", "cc_approval_date", "invitation_to_bid_date",
        "bid_submission_deadline_date", "bid_openning_date", "bid_evaluation_start_date", "bid_evaluation_end_date",
        "display_of_beb_notice_date", "contract_award_date", "contract_signature_date", "approval_date",
        "cancelled_date", "is_framework", "has_lots", "created_at", "updated_at"]


def get(url: str, tries: int = 6) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(15 * (i + 1))


def compact(r: dict) -> dict:
    row = {k: r.get(k) for k in KEEP}
    pe = r.get("procurement_plan_entry") or {}
    pt = pe.get("procurement_type") or {}
    pm = r.get("procurement_method") or {}
    row.update(
        subject=pe.get("subject_of_procurement"),
        plan_estimated_amount=pe.get("estimated_amount"),
        procurement_type=pt.get("code"),
        procurement_type_ocds=pt.get("ocds_code"),
        method_code=pm.get("code"),
        method_title=pm.get("title"),
        method_bidding_period=pm.get("bidding_period"),
        pde_title=(r.get("pdes") or {}).get("title"),
        pde_category=(r.get("pdes") or {}).get("category"),
    )
    return row


def crawl() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    page = int(STATE.read_text()) + 1 if STATE.exists() else 1
    last = None
    with OUT.open("a") as fh:
        while last is None or page <= last:
            d = get(f"{API}?page={page}")["data"]
            last = d["last_page"]
            for r in d["data"]:
                fh.write(json.dumps(compact(r)) + "\n")
            fh.flush()
            STATE.write_text(str(page))
            if page % 100 == 0:
                print(f"page {page}/{last}", flush=True)
            page += 1
            time.sleep(PAUSE)
    print("crawl complete", flush=True)


def sample(n_pages: int, seed: int = 7) -> None:
    """Random-page sample across all years (for a preliminary, representative analysis)."""
    import random

    out = RAW / "bid_invitations_sample.jsonl"
    done_f = RAW / "bid_invitations_sample.pages"
    done = set(map(int, done_f.read_text().split())) if done_f.exists() else set()
    last = get(f"{API}?page=1")["data"]["last_page"]
    pages = random.Random(seed).sample(range(1, last + 1), n_pages)
    with out.open("a") as fh, done_f.open("a") as df:
        for i, page in enumerate(pages, 1):
            if page in done:
                continue
            try:
                recs = get(f"{API}?page={page}", tries=3)["data"]["data"]
            except Exception as e:  # skip pages the server cannot serve; keep going
                print(f"skip page {page}: {e}", flush=True)
                time.sleep(30)
                continue
            for r in recs:
                fh.write(json.dumps(compact(r)) + "\n")
            fh.flush()
            df.write(f"{page}\n")
            df.flush()
            if i % 25 == 0:
                print(f"sample {i}/{n_pages}", flush=True)
            time.sleep(PAUSE)
    print("sample complete", flush=True)


def build() -> None:
    import pandas as pd

    df = pd.read_json(OUT, lines=True, dtype=False).drop_duplicates("id", keep="last")
    df.to_parquet(ROOT / "data" / "processed" / "bid_invitations.parquet", index=False)
    print(f"bid_invitations: {len(df):,} unique records")


if __name__ == "__main__":
    if "--sample" in sys.argv:
        sample(int(sys.argv[sys.argv.index("--sample") + 1]))
        sys.exit()
    if "--build-only" not in sys.argv:
        crawl()
    build()
