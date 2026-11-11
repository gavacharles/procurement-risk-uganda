"""(7) Labels from the Auditor General's engineering audits -> validate (and later train) risk models.

Source: OAG "All Engineering Audits" (https://www.oag.go.ug/allengineeringaudits), 66 value-for-money
audits of construction works, 2017/18-2024/25.

Steps
  1. Parse listing -> (title, financial year, report page); report page -> PDF URL.
  2. Download PDFs (cached), extract text (pypdf).
  3. NLP features per report: counts of finding terms (overpayment, defects/shoddy, incomplete/
     abandoned, delays/time overrun, cost overrun/variations, non-compliance) and the largest
     UGX amount mentioned near "overpa" (a crude money-at-issue measure).
  4. Link each audit to a procuring entity (fuzzy match of names in the title to PPDA buyer names)
     and financial year.
  5. Validation: entity-year mean risk score / flag rates for audited entities with severe findings
     vs audited entities with mild findings.

Out: data/processed/oag_engineering.parquet, paper/oag_labels.json
"""
from __future__ import annotations

import html
import io
import json
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
CACHE = Path.home() / ".cache" / "procurement-risk" / "oag"  # outside cloud-synced folder
BASE = "https://www.oag.go.ug"
UA = "procurement-risk-research (github.com/gavacharles/procurement-risk-uganda)"
TERMS = {
    "overpayment": r"over-?pa(?:id|yment|yments)",
    "defects": r"defect|shoddy|poor (?:quality|workmanship)|substandard|cracks?\b",
    "incomplete": r"incomplete|abandon|not (?:been )?completed|stalled",
    "delay": r"delay|time overrun|behind schedule|extension of time",
    "cost_overrun": r"cost overrun|variation|price adjustment",
    "noncompliance": r"not in (?:accordance|compliance)|non-?complian|contrary to",
}


def fetch(url: str, binary=False, tries=4):
    """Fetch via curl: it verifies TLS against the macOS trust store, which completes OAG's
    certificate chain (Python's bundled store rejects it). Verification stays ON."""
    import subprocess
    for i in range(tries):
        r = subprocess.run(["curl", "-sSL", "--fail", "-m", "300", "-A", UA, url], capture_output=True)
        if r.returncode == 0:
            return r.stdout if binary else r.stdout.decode("utf-8", "ignore")
        if i == tries - 1:
            raise RuntimeError(f"curl failed ({r.returncode}): {r.stderr.decode()[:200]}")
        time.sleep(10 * (i + 1))


def listing() -> pd.DataFrame:
    s = fetch(f"{BASE}/allengineeringaudits")
    rows = []
    for block in s.split("<h5")[1:]:
        t = re.search(r">(.*?)</h5>", block, re.S)
        link = re.search(r'href="(/(?:view\w*|reports)/\d+)"', block)
        if not t or not link:
            continue
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", block)).split())
        fy = re.search(r"Financial Year:\s*(\d{4}/\d{2})", text)
        yr = re.search(r"Audit Year:\s*(\d{4})", text)
        rows.append(dict(title=" ".join(html.unescape(re.sub(r"<[^>]+>", " ", t.group(1))).split()),
                         page=link.group(1), fy=fy.group(1) if fy else None, audit_year=yr.group(1) if yr else None))
    return pd.DataFrame(rows).drop_duplicates("page")


def pdf_text(page: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^a-z0-9]+", "_", page.lower())
    txt = CACHE / f"{key}.txt"
    if txt.exists():
        return txt.read_text()
    v = fetch(BASE + page)
    m = re.search(r'(https?://[^"\s]+\.pdf)', v, re.I)
    if not m:
        txt.write_text("")
        return ""
    import pypdf
    data = fetch(m.group(1).replace(" ", "%20"), binary=True)
    reader = pypdf.PdfReader(io.BytesIO(data))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    txt.write_text(text)
    return text


def features(text: str) -> dict:
    t = text.lower()
    f = {k: len(re.findall(v, t)) for k, v in TERMS.items()}
    f["pages_chars"] = len(t)
    amts = []
    for m in re.finditer(r"over-?pa\w*", t):
        win = t[max(0, m.start() - 250): m.end() + 250]
        for a in re.findall(r"(?:ugx|shs\.?|ushs\.?)\s*([\d,]{6,})", win):
            try:
                amts.append(float(a.replace(",", "")))
            except ValueError:
                pass
    f["max_overpayment_ugx"] = max(amts) if amts else 0.0
    f["severity"] = (f["overpayment"] > 0) + (f["defects"] > 2) + (f["incomplete"] > 2)
    return f


def match_entity(title: str, buyers: list[str]) -> tuple[str | None, float]:
    """Exact town match: '... implemented by <Town> Municipal Council' -> PPDA buyer whose name
    starts with <Town> (Municipal/City/District). Municipalities later upgraded to cities map to
    '<Town> City Council'. Programme-wide audits (several entities) stay unmatched."""
    t = re.sub(r"Report of the Auditor General.*$", "", title, flags=re.I)
    m = re.search(r"(?:implemented by|undertaken by|executed by)\s+(?:the\s+)?([A-Z][a-z]+(?:[- ][A-Z][a-z]+)?)\s+"
                  r"(Municipal|Municipality|City|District|Town)", t)
    if not m:
        return None, 0.0
    town, kind = m.group(1).lower(), m.group(2).lower()
    pool = [b for b in buyers if b.lower().startswith(town + " ")]
    for pref in ([kind, "municipal", "city"] if kind != "district" else ["district"]):
        hit = [b for b in pool if pref in b.lower()]
        if hit:
            return sorted(hit, key=len)[0], 1.0
    return None, 0.0


def fy_of(ts: pd.Timestamp) -> str | None:
    if pd.isna(ts):
        return None
    y = ts.year if ts.month >= 7 else ts.year - 1
    return f"{y}/{str(y + 1)[2:]}"


def main():
    L = listing()
    print(f"{len(L)} engineering audit reports", flush=True)
    feats = []
    for r in L.itertuples():
        try:
            t = pdf_text(r.page)
        except Exception as e:
            print("  fail", r.page, e, flush=True)
            t = ""
        feats.append(features(t) if t else {})
        print(f"  {r.fy} {r.title[:70]} chars={len(t)}", flush=True)
    F = pd.concat([L.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    buyers = sorted(w.buyer_name.dropna().unique())
    m = F.title.map(lambda t: match_entity(t, buyers))
    F["buyer_name"], F["match_score"] = [x[0] for x in m], [x[1] for x in m]
    F.to_parquet(P / "oag_engineering.parquet", index=False)

    # validation: entity x FY risk profile for audited entities
    w["fy"] = w.tender_start.map(fy_of)
    prof = w.groupby(["buyer_name", "fy"]).agg(n=("ocid", "size"), mean_risk=("risk_score", "mean"),
                                               bunched=("f_award_just_below_est", "mean"),
                                               non_open=("f_restricted_method", "mean")).reset_index()
    J = F.dropna(subset=["buyer_name"]).merge(prof, on=["buyer_name", "fy"], how="left")
    J = J[J.n >= 3]
    out = dict(reports=int(len(F)), with_text=int((F.pages_chars.fillna(0) > 1000).sum()),
               matched_to_entity=int(F.buyer_name.notna().sum()), matched_with_risk_data=int(len(J)),
               term_prevalence={k: float((F[k].fillna(0) > 0).mean()) for k in TERMS},
               overpayment_reports=int((F.max_overpayment_ugx.fillna(0) > 0).sum()),
               total_max_overpayment_ugx=float(F.max_overpayment_ugx.fillna(0).sum()))
    if len(J) >= 10:
        J["severe"] = J.severity >= 2
        out["risk_by_severity"] = J.groupby("severe")[["mean_risk", "bunched", "non_open"]].mean().round(3).to_dict("index")
        from scipy.stats import spearmanr
        rho, p = spearmanr(J.mean_risk, J.severity)
        out["spearman_risk_vs_severity"] = dict(rho=float(rho), p=float(p), n=int(len(J)))
    (ROOT / "paper" / "oag_labels.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
