"""Follow-on step 2: reported physical progress of road projects from government monitoring reports.

Sources (data/raw/reports/, downloaded from finance.go.ug and works.go.ug; see SOURCES): the Ministry of
Finance Budget Monitoring and Accountability Unit (BMAU) roads / ITIS programme monitoring reports, semi-annual
(status at 31 December) and annual (status at 30 June), FY2016/17 to FY2024/25, plus Ministry of Works and
roads-authority annual performance reports. Monitors visit each project and report cumulative physical progress,
planned progress, time progress (share of contract period elapsed), right-of-way (RoW) acquisition, contract
dates and the reasons for delay.

Extraction: PyMuPDF table detection; a row = project name cell + remarks cell (continuation rows and rows
split across pages are appended to the previous project). Prose-only pages are scanned too, with the project
taken as the last alias seen before the sentence. Regular expressions then pull the numbers. Every value keeps
its source file, page and a text snippet so it can be audited; regex extraction is imperfect, so the matched
panel for the paper's roads is checked by hand (data/manual/reported_progress_checked.csv overrides).

Out: data/processed/reported_progress.parquet (one row per project mention), paper/reported_progress.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "reports"
P = ROOT / "data" / "processed"
CACHE = Path.home() / ".cache" / "procurement-risk" / "reports"

SOURCES = {
    "Semi-Annual_Budget_Monitoring_Report_FY2016-17.pdf": ("2016-12-31", "MoFPED BMAU"),
    "Annual_Monitoring_Report_FY2016-17.pdf": ("2017-06-30", "MoFPED BMAU"),
    "Roads_Sub-Sector_Semi-Annual__Monitoring_Report_FY_2018-19.pdf": ("2018-12-31", "MoFPED BMAU"),
    "Roads_Sub-Sector_Semi-Annual_Monitoring_Report_FY2019-20.pdf": ("2019-12-31", "MoFPED BMAU"),
    "Roads_Sub-Sector_Annual_Budget_Monitoring_Report_FY2019-20.pdf": ("2020-06-30", "MoFPED BMAU"),
    "UNRA_APR_FY2019-20.pdf": ("2020-06-30", "UNRA APR"),
    "MoWT_ASPR_FY2019-20.pdf": ("2020-06-30", "MoWT ASPR"),
    "Roads_Sub-Sector_Semi-Annual_Monitoring_Report_FY2020-21.pdf": ("2020-12-31", "MoFPED BMAU"),
    "MoWT_APPR_FY2020-21.pdf": ("2021-06-30", "MoWT APPR"),
    "UNRA_APR_FY2021-22.pdf": ("2022-06-30", "UNRA APR"),
    "MoWT_APPR_FY2021-22.pdf": ("2022-06-30", "MoWT APPR"),
    "ITIS_Programme_Semi-Annual_Monitoring_Report_FY2022-23_.pdf": ("2022-12-31", "MoFPED BMAU"),
    "ITIS_Annual_Monitoring_Report_FY2022-23.pdf": ("2023-06-30", "MoFPED BMAU"),
    "Integrated_Transport_Infrastructure_Services_Annual_Budget_Monitoring_report_FY_2023-24.pdf": ("2024-06-30", "MoFPED BMAU"),
    "MoWT_APPR_FY2023-24.pdf": ("2024-06-30", "MoWT APPR"),
    "Integrated_Transport_and_Infrastructure_Services_Semi_Annual_Report_FY_2024-25.pdf": ("2024-12-31", "MoFPED BMAU"),
    "Integrated_Transport_Infrastructure_Services_Annual_Monitoring_FY2024-25_Report.pdf": ("2025-06-30", "MoFPED BMAU"),
}

# satellite-sample road id -> regex identifying it in report text (distinctive place names)
ALIASES = {
    "tirinyi": r"Tirinyi", "kamonkoli": r"Kamonkoli", "bumbobi": r"Bumbobi|Lwakhakha",
    "muyembe": r"Muyembe|Nakapiripirit", "kigumba": r"Kigumba", "seeta": r"Seeta|Namugongo",
    "rwenkunye": r"Rwenkunye|Puranga|Apac\s*[-–]\s*Lira", "fortkamwenge": r"Kamwenge",
    "bukakata": r"Bukakata", "hoima": r"Butiaba|Wanseko", "kalule": r"Kalule", "mgahinga": r"Mgahinga",
    "bunyonyi": r"Bunyonyi", "koboko": r"Koboko|Yumbe", "luwero": r"Butalangu",
    "namagumba": r"Namagumba|Budadiri", "kyenjojo": r"Bwizi|Rwamwanja", "kalangala": r"Kalangala",
    "senge": r"Senge", "namulanda": r"Namulanda", "kisozi": r"Kisozi", "katooke": r"Katooke", "bududa": r"Bududa",
    "mityana": r"Mityana", "mutukula": r"Mutukula", "matugga": r"Matugga|Kasangati", "ishaka": r"Katunguru",
}
NAME_RE = re.compile(r"[A-Z][a-z]+(?:\s*[-–—]\s*[A-Z][a-z]+)+")  # "Place–Place(–Place)" project names

NUM = r"(\d{1,3}(?:\.\d+)?)\s*%"
PAT = {
    "physical": re.compile(r"(?:cumulative\s+)?physical\s+progress\s+(?:of\s+the\s+\w+\s+)?(?:was|of|stood\s+at|is|at|"
                           r"had\s+reached|reached)?\s*(?:at|about|estimated\s+at)?\s*" + NUM, re.I),
    "planned": re.compile(r"against\s+(?:a\s+|the\s+)?(?:planned|target(?:ed)?|plan(?:ned)?\s+of)\s*(?:progress\s+of\s+|"
                          r"target\s+of\s+|of\s+)?" + NUM, re.I),
    "time": re.compile(r"time\s+(?:progress|lapse|elapsed|progression)\s*(?:of|was|at)?\s*(?:at\s+)?" + NUM, re.I),
    "row": re.compile(r"(?:RoW|right[\s-]of[\s-]way)[^.%]{0,90}?" + NUM, re.I),
    "financial": re.compile(r"financial\s+progress\s+(?:of\s+the\s+project\s+)?(?:was|of|at|stood\s+at)?\s*(?:at\s+)?"
                            + NUM, re.I),
}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = rf"(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS}),?\s+\d{{4}}|(?:{MONTHS})\s+\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{4}})"
DPAT = {
    "commencement": re.compile(rf"(?:commencement\s+date|start\s+date|commenced\s+(?:on|in)|works\s+started\s+(?:on|in)|"
                               rf"commencement\s+(?:on|of\s+works\s+on))\s*(?:of|was|:)?\s*(?:on\s+)?{DATE}", re.I),
    "completion": re.compile(rf"(?:(?<!revised\s)completion\s+date|end\s+date|expected\s+to\s+(?:be\s+)?complete\w*\s+(?:by|on|in))"
                             rf"\s*(?:of|was|is|:)?\s*(?:on\s+)?{DATE}", re.I),
    "revised_completion": re.compile(rf"(?:revised\s+(?:completion|end)\s+date|extended\s+to|extension\s+of\s+time[^.]{{0,60}}?to)"
                                     rf"\s*(?:of|was|is|:)?\s*(?:on\s+)?{DATE}", re.I),
}
CAUSES = {
    "cause_row": r"right[\s-]of[\s-]way|\bRoW\b|compensat|\bPAPs?\b|land\s+acquisition",
    "cause_contractor": r"cash\s*flow|contractor'?s?\s+(?:poor|low|inadequate|slow)|poor\s+(?:contractor|mobili[sz]ation)|"
                        r"inadequate\s+(?:equipment|plant)|low\s+(?:equipment|productivity)",
    "cause_payment": r"delayed\s+payment|late\s+payment|arrears|unpaid\s+certificate|interest\s+on\s+delayed|"
                     r"inadequate\s+(?:funding|release)|budget\s+cut|funding\s+gap",
    "cause_weather": r"rain|weather|flood",
    "cause_design": r"design\s+(?:change|review|error)|scope\s+change|variation|redesign",
    "cause_materials": r"material|quarr|borrow\s+pit|bitumen\s+shortage",
    "cause_covid": r"covid|pandemic|lockdown",
}


def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def table_rows(doc):
    """(page, name, remarks) per project row; rows without a name continue the previous project."""
    out, cur = [], None
    for i, page in enumerate(doc):
        try:
            tabs = page.find_tables().tables
        except Exception:  # noqa: BLE001
            tabs = []
        for t in tabs:
            for r in t.extract():
                cells = [clean(c) for c in r]
                rest = [c for c in cells if c]
                if not rest:
                    continue
                longest = max(rest, key=len)
                name = next((c for c in cells[:3] if c and c is not longest and re.search(r"[A-Za-z]{3}", c)
                             and len(c) < 160), None)
                if name and re.search(r"[–\-—]|road|bypass|expressway|upgrad|construct", name, re.I):
                    cur = dict(page=i + 1, name=name, text=longest if len(longest) > 40 else " ".join(rest))
                    out.append(cur)
                elif cur is not None and len(longest) > 40:
                    cur["text"] += " " + longest
    return out


def prose_rows(doc):
    """Sentences with 'physical progress' outside tables, attributed to the last project-like name seen."""
    out, last = [], None
    sents = []  # (page, sentence) across the whole document, so a narrative can run over a page break
    for i, page in enumerate(doc):
        sents += [(i + 1, x) for x in re.split(r"(?<=[.;])\s+", clean(page.get_text()))]
    for k, (pg, sent) in enumerate(sents):
        names = NAME_RE.findall(sent)
        if names:
            last = max(names, key=len)
        if re.search(r"physical\s+progress", sent, re.I) and last:
            # progress figures from the sentence itself; causes from it and up to 6 following sentences
            # (stopping at the next project name or progress statement)
            ctx = []
            for _, nx in sents[k + 1:k + 7]:
                if NAME_RE.search(nx) or re.search(r"physical\s+progress", nx, re.I):
                    break
                ctx.append(nx)
            out.append(dict(page=pg, name=last, text=sent, context=" ".join(ctx)))
    return out


def extract(text, context=""):
    rec = {}
    for k, p in PAT.items():
        m = p.search(text)
        rec[k] = float(m.group(1)) if m and float(m.group(1)) <= 100 else None
    for k, p in DPAT.items():
        m = p.search(text)
        rec[k] = pd.to_datetime(re.sub(r"(st|nd|rd|th)\b", "", m.group(1)), errors="coerce", dayfirst=True) if m else None
    for k, p in CAUSES.items():
        rec[k] = bool(re.search(p, text + " " + context, re.I))
    return rec


def road_id(name, text):
    for rid, a in ALIASES.items():
        if re.search(a, name):
            return rid
    return None


def main():
    rows = []
    for fn, (asof, src) in SOURCES.items():
        f = RAW / fn
        if not f.exists():
            print("missing", fn)
            continue
        doc = fitz.open(f)
        cache = CACHE / (fn + ".tables.json")
        if cache.exists():
            tabs = json.loads(cache.read_text())
        else:
            tabs = table_rows(doc)
            CACHE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(tabs))
        cand = [dict(r, how="table") for r in tabs] + [dict(r, how="prose") for r in prose_rows(doc)]
        n = 0
        for r in cand:
            e = extract(r["text"], r.get("context", ""))
            if e["physical"] is None and e["commencement"] is None and e["completion"] is None:
                continue
            rows.append(dict(file=fn, source=src, asof=pd.Timestamp(asof), page=r["page"], how=r["how"],
                             project=r["name"][:150], road_id=road_id(r["name"], r["text"]),
                             snippet=(r["text"] + " " + r.get("context", ""))[:900], **e))
            n += 1
        print(f"{n:4d}  {fn}")
    D = pd.DataFrame(rows)
    # the same project in the same report via table and prose: keep the table row
    D["key"] = D.project.str.lower().str.replace(r"[^a-z]", "", regex=True).str[:25]
    D = D.sort_values("how", key=lambda s: s.eq("prose")).drop_duplicates(["file", "key", "physical"])
    D.drop(columns="key").to_parquet(P / "reported_progress.parquet", index=False)
    M = D[D.road_id.notna()]
    summ = dict(n_reports=int(D.file.nunique()), n_mentions=int(len(D)),
                n_projects_named=int(D.project.str.lower().nunique()),
                n_sample_roads_matched=int(M.road_id.nunique()),
                matched_mentions_with_physical=int(M.physical.notna().sum()),
                by_road=M.groupby("road_id").agg(n=("physical", "count"), first=("asof", "min"),
                                                  last=("asof", "max")).astype(str).to_dict("index"))
    (ROOT / "paper" / "reported_progress.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "by_road"}, indent=2))
    print(M.groupby("road_id").physical.count().to_string())


if __name__ == "__main__":
    sys.exit(main())
