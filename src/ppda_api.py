"""Fetch supplementary public datasets from the PPDA API (no authentication).

  /contracts-variations-prices                 price variations (cost overruns)
  /contracts-variations-dates                  time extensions
  /suspended-providers/egp-suspended-providers PPDA suspended providers (labels)

Polite: one request per second, plain GETs only.
Outputs: data/raw/ppda_api/*.json and data/processed/{variations_price,variations_time,suspended}.parquet
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ppda_api"
P = ROOT / "data" / "processed"
API = "https://cdn.ppda.go.ug/api"
UA = "procurement-risk-research (python urllib)"


def get(url: str, tries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except (TimeoutError, OSError):
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def fetch_all(path: str) -> list[dict]:
    rows, page = [], 1
    while True:
        j = get(f"{API}/{path}{'&' if '?' in path else '?'}page={page}")
        d = j["data"]
        rows += d["data"]
        if page >= d["last_page"]:
            return rows
        page += 1
        time.sleep(1)


def _ac(r: dict) -> dict:
    ac = r.get("awarded_contract") or {}
    return dict(awarded_contract_id=r.get("awarded_contract_id"), ocid=ac.get("ocds_id"),
                pde_id=r.get("pde_id"), contract_amount=ac.get("contract_amount"),
                contract_exchange_rate=ac.get("exchange_rate"), signed_date=ac.get("signed_date"),
                commencement_date=ac.get("commencement_date"), completion_date=ac.get("completion_date"),
                financial_year=ac.get("financial_year"))


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    P.mkdir(parents=True, exist_ok=True)
    for name, path in [("variations_price", "contracts-variations-prices"),
                       ("variations_time", "contracts-variations-dates"),
                       ("suspended", "suspended-providers/egp-suspended-providers")]:
        rows = fetch_all(path)
        (RAW / f"{name}.json").write_text(json.dumps(rows))
        if name == "variations_price":
            df = pd.DataFrame([dict(_ac(r), variation_amount=r.get("amount"), variation_fx=r.get("exchange_rate"),
                                    variation_type=r.get("price_variation_type"), created_at=r.get("created_at"))
                               for r in rows])
            df["variation_ugx"] = df.variation_amount * df.variation_fx.fillna(1)
            df["contract_ugx"] = df.contract_amount * df.contract_exchange_rate.fillna(1)
            df["variation_pct"] = df.variation_ugx / df.contract_ugx.where(df.contract_ugx > 0)
        elif name == "variations_time":
            df = pd.DataFrame([dict(_ac(r), initial_completion=r.get("initial_completion_date"),
                                    new_completion=r.get("new_plan_completion_date"),
                                    variation_type=r.get("date_variation_type"), details=r.get("details"))
                               for r in rows])
            df["extension_days"] = (pd.to_datetime(df.new_completion, errors="coerce")
                                    - pd.to_datetime(df.initial_completion, errors="coerce")).dt.days
        else:
            df = pd.DataFrame(rows)
        df.to_parquet(P / f"{name}.parquet", index=False)
        print(f"{name:18s} {len(df):5d} rows")


if __name__ == "__main__":
    main()
