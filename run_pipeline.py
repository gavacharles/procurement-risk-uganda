"""Run the full pipeline: download (optional) -> ingest -> entities -> red flags -> network -> report -> PPDA API -> paper analyses.

Usage:
  python run_pipeline.py              # use existing data/raw file
  python run_pipeline.py --download   # fetch latest Uganda OCDS bulk file first
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

SOURCE_URL = "https://data.open-contracting.org/en/publication/130/download?name=full.jsonl.gz"
RAW = ROOT / "data" / "raw" / "uganda_ppda_ocds_full.jsonl.gz"


def download() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SOURCE_URL}")
    tmp = RAW.with_suffix(".part")
    urllib.request.urlretrieve(SOURCE_URL, tmp)
    tmp.replace(RAW)


def main() -> None:
    if "--download" in sys.argv or not RAW.exists():
        download()
    import analysis
    import entities
    import ingest
    import ppda_api
    import network
    import red_flags
    import report

    print("== ingest"); ingest.main()
    print("== entities"); entities.main()
    print("== red flags"); red_flags.main()
    print("== network"); network.main("all"); network.main("works")
    print("== report"); report.main()
    print("== PPDA API supplements"); ppda_api.main()
    print("== paper analyses"); analysis.main()


if __name__ == "__main__":
    main()
