"""(9) Ownership/contact network from PPDA's public provider register (/providers).

PRIVACY: phone, email, address and registration numbers are personal/business contact data.
They are normalised and immediately replaced by salted SHA-256 hashes; raw values are never
written to disk or published. Hashes are used only to find firms that share contact details.

Modes
  --sample N   random sample of N pages (coverage check)
  --crawl      full resumable crawl (≈9,300 pages; days at current server speed)

Out: data/processed/providers_hashed.parquet  (id, name, has_* flags, h_tel, h_email, h_addr, h_reg)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path.home() / ".cache" / "procurement-risk" / "providers"  # outside cloud-synced folder (file locks)
OUT = ROOT / "data" / "processed" / "providers_hashed.parquet"
API = "https://cdn.ppda.go.ug/api/providers"
UA = "procurement-risk-research (python urllib)"
SALT_FILE = RAW / ".provider_salt"


def salt() -> bytes:
    RAW.mkdir(parents=True, exist_ok=True)
    if not SALT_FILE.exists():
        SALT_FILE.write_bytes(os.urandom(32))
    return SALT_FILE.read_bytes()


def h(v: str | None, kind: str) -> str | None:
    if not v:
        return None
    s = str(v).lower()
    if kind == "tel":
        d = re.sub(r"\D", "", s)
        d = d[-9:] if len(d) >= 9 else None  # last 9 digits (drops +256 / 0 prefixes)
        s = d
    elif kind == "email":
        s = s.strip()
    else:
        s = re.sub(r"[^a-z0-9]", "", s)
    if not s or len(s) < 5:
        return None
    return hashlib.sha256(salt() + kind.encode() + s.encode()).hexdigest()[:20]


def get(page: int, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(f"{API}?page={page}", headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["data"]
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def row(r: dict) -> dict:
    return dict(id=r["id"], name=r.get("name"), created_at=r.get("created_at"),
                has_tel=bool(r.get("tel")), has_email=bool(r.get("email")), has_addr=bool(r.get("address")),
                has_reg=bool(r.get("identifier_id")),
                h_tel=h(r.get("tel"), "tel"), h_email=h(r.get("email"), "email"),
                h_addr=h(r.get("address"), "addr"), h_reg=h(r.get("identifier_id"), "reg"),
                n_bids=len(r.get("bid_submissions") or []), n_wins=len(r.get("successful_submissions") or []))


def run(pages: list[int], tag: str):
    RAW.mkdir(parents=True, exist_ok=True)
    part = RAW / f"providers_{tag}.jsonl"
    done = set()
    if part.exists():
        done = {json.loads(l)["_page"] for l in part.read_text().splitlines() if l.strip()}
    with part.open("a") as fh:
        for i, p in enumerate(pages, 1):
            if p in done:
                continue
            try:
                d = get(p)
            except Exception as e:
                print("skip", p, e, flush=True)
                continue
            for r in d["data"]:
                fh.write(json.dumps(dict(row(r), _page=p)) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"{tag} {i}/{len(pages)}", flush=True)
            time.sleep(1)
    df = pd.read_json(part, lines=True).drop_duplicates("id")
    return df


def main():
    last = get(1)["last_page"]
    if "--sample" in sys.argv:
        n = int(sys.argv[sys.argv.index("--sample") + 1])
        df = run(random.Random(11).sample(range(1, last + 1), n), "sample")
    else:
        df = run(list(range(1, last + 1)), "full")
        df.to_parquet(OUT, index=False)
    cov = {k: float(df[k].mean()) for k in ["has_tel", "has_email", "has_addr", "has_reg"]}
    shared = {k: int(df[k].dropna().duplicated(keep=False).sum()) for k in ["h_tel", "h_email", "h_addr", "h_reg"]}
    print(json.dumps(dict(providers=int(len(df)), last_page=last, coverage=cov,
                          providers_sharing_value_within_sample=shared), indent=1))


if __name__ == "__main__":
    main()
