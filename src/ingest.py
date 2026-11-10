"""Flatten the Uganda PPDA OCDS bulk file into analysis tables.

Source: OCP Data Registry, publication 130 (Uganda PPDA), full.jsonl.gz
        https://data.open-contracting.org/en/publication/130

The Uganda feed splits each procurement across two record families that
do not share an ocid:
  * "tender" records  - tender (title, estimate, method, window) + awards + contracts
  * "bid" records     - bids (every bid price) + parties (tenderers) + awards + contracts
They are kept as separate tables; each supports different red flags.

Outputs (data/processed/*.parquet):
  tenders.parquet     one row per tender record (with its first award/contract)
  bid_procs.parquet   one row per bid record (procurement with bid list)
  bids.parquet        one row per bid
  tenderers.parquet   one row per (procurement, tenderer)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "uganda_ppda_ocds_full.jsonl.gz"
OUT = ROOT / "data" / "processed"

# Approximate conversion to UGX for the few non-UGX values (assumption; refine later).
FX_TO_UGX = {"UGX": 1.0, None: 1.0, "USD": 3700.0, "EUR": 4000.0, "GBP": 4700.0, "KES": 28.0}


def amt(v: dict | None) -> tuple[float | None, str | None]:
    if not v:
        return None, None
    a = v.get("amount")
    return (float(a) if a is not None else None), v.get("currency")


def to_ugx(a, cur):
    if a is None:
        return None
    rate = FX_TO_UGX.get(cur)
    return a * rate if rate else None


def main(path: Path = RAW) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tenders, procs, bids, tenderers = [], [], [], []

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ocid = r["ocid"]
            buyer = r.get("buyer") or {}
            awards = r.get("awards") or []
            contracts = r.get("contracts") or []
            a0 = awards[0] if awards else {}
            c0 = contracts[0] if contracts else {}
            sup0 = (a0.get("suppliers") or [{}])[0]
            aw_amt, aw_cur = amt(a0.get("value"))
            co_amt, co_cur = amt(c0.get("value"))
            common = dict(
                ocid=ocid,
                release_date=r.get("date"),
                buyer_id=buyer.get("id"),
                buyer_name=buyer.get("name"),
                award_id=a0.get("id"),
                award_date=a0.get("date"),
                award_value_ugx=to_ugx(aw_amt, aw_cur),
                award_currency=aw_cur,
                n_awards=len(awards),
                supplier_id=sup0.get("id"),
                supplier_name=sup0.get("name"),
                n_suppliers=len(a0.get("suppliers") or []),
                contract_id=c0.get("id"),
                contract_title=c0.get("title"),
                contract_value_ugx=to_ugx(co_amt, co_cur),
                contract_status=c0.get("status"),
            )

            t = r.get("tender")
            if t:
                est, est_cur = amt(t.get("value"))
                tp = t.get("tenderPeriod") or {}
                tenders.append(
                    dict(
                        common,
                        tender_id=t.get("id"),
                        title=t.get("title"),
                        description=t.get("description"),
                        tender_status=t.get("status"),
                        method=t.get("procurementMethod"),
                        estimate_ugx=to_ugx(est, est_cur),
                        estimate_currency=est_cur,
                        tender_start=tp.get("startDate"),
                        tender_end=tp.get("endDate"),
                        bid_opening=(t.get("bidOpening") or {}).get("date"),
                        n_lots=len(t.get("lots") or []),
                        is_framework=bool((t.get("techniques") or {}).get("hasFrameworkAgreement")),
                    )
                )

            b = r.get("bids")
            if b:
                details = b.get("details") or []
                vals = []
                for d in details:
                    v, cur = amt(d.get("value"))
                    vu = to_ugx(v, cur)
                    vals.append(vu)
                    bids.append(
                        dict(ocid=ocid, bid_id=d.get("id"), bid_date=d.get("date"),
                             bid_value_ugx=vu, bid_status=d.get("status"))
                    )
                parties = r.get("parties") or []
                tdr = [p for p in parties if "tenderer" in (p.get("roles") or [])]
                for p in tdr:
                    tenderers.append(
                        dict(ocid=ocid, party_id=p.get("id"), party_name=p.get("name"),
                             is_supplier="supplier" in (p.get("roles") or []),
                             buyer_id=buyer.get("id"), release_date=r.get("date"))
                    )
                procs.append(
                    dict(common, title=c0.get("title"), n_bids=len(details),
                         n_tenderers=len(tdr))
                )

    for name, rows in [("tenders", tenders), ("bid_procs", procs), ("bids", bids), ("tenderers", tenderers)]:
        df = pd.DataFrame(rows)
        for col in [c for c in df.columns if c.endswith(("_date", "_start", "_end")) or c in ("bid_opening",)]:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name:10s} {len(df):8,d} rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RAW)
