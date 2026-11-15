"""Follow-on step 5: formal disputes and parliamentary attention for the satellite-sample roads.

Source: ULII (Uganda Legal Information Institute) search API, which indexes court judgments (incl. the
Commercial Division, the Centre for Arbitration and Dispute Resolution, the PPDA Appeals Tribunal, the Court
of Appeal and Supreme Court) and the Hansard of Parliament.

For each road, documents that mention two of its distinctive place names are collected:
  disputes   judgments in which the roads authority, the works ministry or the Attorney General is a party,
             or that sit in a commercial/arbitration/procurement court (land cases between private parties that
             merely mention the towns are dropped); each is then checked by hand
             (data/manual/road_disputes_checked.csv: relevant yes/no, what the dispute is about)
  hansard    dated Hansard sittings that mention the road (a measure of political attention)

A dispute is not evidence of delay, and a delay is not evidence of fault; the link is descriptive.

Out: data/processed/road_disputes_raw.parquet, paper/road_disputes.json
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
import satellite as sat  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (academic research; procurement-risk-uganda)"}
QUERIES = {  # road id -> place-name pairs (both must appear)
    "tirinyi": [("Tirinyi", "Pallisa")], "kamonkoli": [("Pallisa", "Kamonkoli")],
    "bumbobi": [("Bumbobi", "Lwakhakha")], "muyembe": [("Muyembe", "Nakapiripirit")],
    "kigumba": [("Kigumba", "Bulima"), ("Bulima", "Kabwoya")], "seeta": [("Seeta", "Namugongo")],
    "rwenkunye": [("Rwenkunye", "Apac"), ("Apac", "Puranga")], "fortkamwenge": [("Fort Portal", "Kamwenge")],
    "bukakata": [("Masaka", "Bukakata")], "hoima": [("Butiaba", "Wanseko")], "kalule": [("Kalule", "Bamunanika")],
    "mgahinga": [("Kisoro", "Mgahinga")], "koboko": [("Koboko", "Yumbe")], "luwero": [("Luwero", "Butalangu")],
    "namagumba": [("Namagumba", "Budadiri")], "kyenjojo": [("Kyenjojo", "Bwizi"), ("Bwizi", "Rwamwanja")],
    "kalangala": [("Kalangala", "Mulabana")], "bududa": [("Bubulo", "Bududa")],
}
PARTY = re.compile(r"Roads?\s+Authou?rity|\bUNRA\b|Ministry\s+of\s+Works|Attorney\s+General", re.I)
COURTS = ("ugcommc", "ugcader", "ugppdpaat", "ugca", "ugsc", "ughccd")


def search(q, pages=3):
    def go():
        out = []
        for page in range(1, pages + 1):
            u = "https://ulii.org/search/api/documents/?" + urllib.parse.urlencode({"search": q, "page_size": 20, "page": page})
            try:
                with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90) as r:
                    d = json.load(r)
            except Exception as e:  # noqa: BLE001
                print("  fail", q, e)
                break
            for href, t in re.findall(r'<a class="h5 text-primary"\s+href="(/en/akn/ug/[^"]+)"[^>]*>(.*?)</a>',
                                      d["results_html"], re.S):
                out.append(dict(uri=href, title=re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(t))).strip()))
            if page * 20 >= d.get("count", 0):
                break
            time.sleep(1)
        return out
    return sat.cached(f"ulii {q} {pages}", go)


def kind(uri):
    if "/judgment/" in uri:
        return "judgment", uri.split("/judgment/")[1].split("/")[0]
    if "parliamentary-debate" in uri:
        return "hansard", None
    return "other", None


def main():
    rows = []
    for rid, pairs in QUERIES.items():
        for a, b in pairs:
            for h in search(f'"{a}" "{b}"'):
                k, court = kind(h["uri"])
                m = re.search(r"(\d{4}-\d{2}-\d{2})", h["uri"])
                rows.append(dict(road_id=rid, query=f"{a} + {b}", kind=k, court=court, date=m.group(1) if m else None,
                                 **h))
    D = pd.DataFrame(rows).drop_duplicates(["road_id", "uri"])
    D["party_state"] = D.title.str.contains(PARTY)
    D["dispute_candidate"] = (D.kind == "judgment") & (D.party_state | D.court.isin(COURTS))
    D.to_parquet(P / "road_disputes_raw.parquet", index=False)
    J = D[D.dispute_candidate]
    H = D[D.kind == "hansard"]
    out = dict(n_roads=len(QUERIES), n_dispute_candidates=int(len(J)),
               candidates=J[["road_id", "court", "date", "title", "uri"]].to_dict("records"),
               hansard_sittings=H.groupby("road_id").date.nunique().to_dict())
    (ROOT / "paper" / "road_disputes.json").write_text(json.dumps(out, indent=2))
    print(J[["road_id", "court", "date", "title"]].to_string())
    print(H.groupby("road_id").date.nunique().to_string())


if __name__ == "__main__":
    main()
