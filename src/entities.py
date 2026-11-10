"""Supplier entity resolution (name-based, first pass).

PPDA party IDs are not unique per legal entity: the same firm appears under
several spellings ("Acme Build & Civil Ltd" / "AcmeBuild &
Civil Limited"). Normalised names give a canonical key.

When two *different* party IDs with the *same* canonical name bid in the same
procurement, it is either a data-entry duplicate or one firm submitting several
bids under name variants (a cover-bidding tactic). Either way it needs review.

Outputs: data/processed/entity_keys.parquet       party_id -> canonical key
         data/processed/variant_cobids.parquet    procurements with name-variant co-bidding
Next step: match canonical keys to URSB registration numbers.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

SUFFIXES = r"\b(limited|ltd|co|company|companies|corp|corporation|inc|plc|smc|\(u\)|u|uganda|enterprises?|ent)\b"


def canonical(name: str | None) -> str | None:
    if not isinstance(name, str):
        return None
    s = unicodedata.normalize("NFKD", name).lower()
    s = s.replace("&amp;", "&").replace("&", " and ")
    s = re.sub(r"\(u\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(SUFFIXES, " ", s)
    s = re.sub(r"\band\b", " ", s)
    s = re.sub(r"\s+", "", s)  # "acme build" == "acmebuild"
    return s or None


def main() -> None:
    tdr = pd.read_parquet(P / "tenderers.parquet").dropna(subset=["party_id"])
    keys = tdr.drop_duplicates("party_id")[["party_id", "party_name"]].copy()
    keys["canon"] = keys.party_name.map(canonical)
    keys["n_ids_same_canon"] = keys.groupby("canon").party_id.transform("nunique")
    keys.to_parquet(P / "entity_keys.parquet", index=False)

    t = tdr.merge(keys[["party_id", "canon"]], on="party_id")
    g = t.groupby(["ocid", "canon"]).party_id.nunique()
    hits = g[g >= 2].reset_index().rename(columns={"party_id": "n_variant_ids"})
    names = t.groupby(["ocid", "canon"]).party_name.apply(lambda s: " | ".join(sorted(set(s)))).rename("variant_names")
    hits = hits.merge(names.reset_index(), on=["ocid", "canon"])
    hits.to_parquet(P / "variant_cobids.parquet", index=False)

    print(f"party IDs: {len(keys):,} -> canonical firms: {keys.canon.nunique():,}")
    print(f"procurements where name-variants of one firm co-bid: {hits.ocid.nunique():,}")
    print(hits.variant_names.value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
