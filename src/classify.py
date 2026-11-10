"""Rule-based classifier: is a procurement a construction *works* contract?

The Uganda OCDS feed has no mainProcurementCategory, so works are identified
from the title. Rules: must match a works pattern AND not match an exclusion
(consultancy, supply-only, design/supervision, media, etc.).

Sub-sector is assigned by first matching group (roads > water > buildings > other).
Precision should be checked against a hand-labelled sample (see reports/).
"""
from __future__ import annotations

import re

import pandas as pd

WORKS = re.compile(
    r"\b(construct\w*|rehabilit\w*|renovat\w*|refurbish\w*|civil works?|"
    r"periodic maint\w*|routine maint\w*|mechani[sz]ed maint\w*|road works?|"
    r"upgrad\w* (?:of|to)|tarmac\w*|paving|gravel\w*|reseal\w*|re-?graveling|"
    r"culvert\w*|bridge\w*|drainage|borehole drilling|drilling of|fenc\w*|"
    r"erection of|completion of|extension of|expansion of|"
    r"latrine\w*|classroom\w*|dormitor\w*|staff houses?|maternity ward|"
    r"building works?|works? (?:at|for|on|in)|maintenance works?|repair works?|"
    r"water supply (?:system|scheme|extension)|piped water|gravity flow)",
    re.I,
)

EXCLUDE = re.compile(
    r"\b(consult\w*|feasibility|design (?:of|and|&|review)|detailed design|"
    r"supervision|technical assistance|study|studies|survey of|audit|"
    r"supply(?:ing)? of|purchase of|procurement of|provision of (?:radio|tv|"
    r"advert|airtime|talk show)|radio|talk show|road show|z-cards|fuel|"
    r"stationery|printing|training|workshop|hire of|insurance|cleaning services|"
    r"materials?|lease|rental|advert\w*|publicity|deliver\w*|"
    r"motor ?vehicles?|vehicles?|generators?|ford|toyota|"
    r"u[a-z]{2} ?\d{3}[a-z]?|network|computers?|printers?|ict|software|"
    r"assorted|spare ?parts?|tyres?|tickets?|engines?|overhaul|hotel|"
    r"notice|enforcement|carpets?)\b",
    re.I,
)

# "Supply and construction/erection/laying" is works-like; let it through.
SUPPLY_WORKS = re.compile(r"supply,? (?:and|&) (?:construct|erect|lay)", re.I)

SUBSECTOR = [
    ("roads", re.compile(r"\b(road|bridge|culvert|tarmac|gravel|paving|reseal|street|"
                         r"junction|carriageway|km\b|kilomet|swamp crossing|causeway)", re.I)),
    ("water", re.compile(r"\b(borehole|water supply|piped water|gravity flow|valley tank|"
                         r"dam\b|irrigation|sewer|water ?borne|reservoir|spring protection)", re.I)),
    ("buildings", re.compile(r"\b(classroom|block|latrine|toilet|dormitor|office|house|"
                             r"ward|theatre|hall|laboratory|library|market|building|"
                             r"health cent|h/c|school|hostel|kitchen|store)", re.I)),
]


def is_works(title: str | None) -> bool:
    if not isinstance(title, str) or not title:
        return False
    if not WORKS.search(title):
        return False
    if EXCLUDE.search(title) and not SUPPLY_WORKS.search(title):
        return False
    return True


def subsector(title: str | None) -> str | None:
    if not title:
        return None
    for name, pat in SUBSECTOR:
        if pat.search(title):
            return name
    return "other_works"


def tag(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    df = df.copy()
    df["is_works"] = df[col].map(is_works)
    df["subsector"] = [subsector(t) if w else None for t, w in zip(df[col], df["is_works"])]
    return df
