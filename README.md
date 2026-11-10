# procurement-risk-uganda

Python pipeline, Streamlit dashboard and public site for fraud-risk analytics on public construction (works) procurement, using Uganda's procurement data as the case study.

Project site: https://gavacharles.github.io/procurement-risk-uganda/ · Citation: [CITATION.cff](CITATION.cff)

> Scores and flags produced by this code are risk indicators that call for review. They are not findings of wrongdoing.

## Contents

| Path | What it is |
|---|---|
| `run_pipeline.py` | End-to-end runner (download, build tables, report) |
| `app.py` | Streamlit dashboard |
| `src/` | Pipeline modules (see below) |
| `tests/test_blind_eval.py` | Tests for the blind-evaluation prototype |
| `docs/` | GitHub Pages site (aggregate figures and district map data only) |
| `EXECUTION.md` | Working plan |
| `CITATION.cff` | Citation metadata |
| `requirements.txt` | Python dependencies |

### Pipeline modules

| Step | File | Output (`data/processed/`) |
|---|---|---|
| Flatten OCDS | `src/ingest.py` | `tenders`, `bid_procs`, `bids`, `tenderers` |
| Supplier name resolution | `src/entities.py` | `entity_keys`, `variant_cobids` |
| Works classifier | `src/classify.py` | used by later steps |
| Red flags and risk score | `src/red_flags.py` | `works_tenders_scored`, `works_bids_scored`, `bidder_sets`, `supplier_profiles`, `buyer_profiles` |
| Co-bidding network | `src/network.py` | `network_{nodes,edges,clusters}[_works]` |
| Report | `src/report.py` | `reports/exploratory_report.md`, `reports/top_risk_works_contracts.csv` |
| Machine learning and NLP | `src/ml_models.py` | `ml_anomaly`, `split_candidates`, `paper/ml_results.json` |
| Blind evaluation prototype | `src/blind_eval.py` | tested in `tests/` |
| District map data | `src/geo.py` | `docs/data/districts.geojson` |
| PPDA bid-invitation crawl | `src/fetch_bid_invitations.py` | `bid_invitations` (long-running, resumable) |
| Helpers | `src/analysis.py`, `src/ppda_api.py` | used by the steps above |

## Setup and running

```bash
python3.12 -m venv ~/.venvs/procfraud          # keep the venv outside synced folders
~/.venvs/procfraud/bin/pip install -r requirements.txt
~/.venvs/procfraud/bin/python run_pipeline.py --download   # fetch data and rebuild (~1 min)
~/.venvs/procfraud/bin/streamlit run app.py                # open the dashboard
```

## Data

Input is the PPDA Government Procurement Portal in Open Contracting Data Standard (OCDS) format, mirrored monthly by the [OCP Data Registry, publication 130](https://data.open-contracting.org/en/publication/130). `run_pipeline.py --download` fetches it into `data/raw/`.

## What is not tracked

`data/`, `reports/` and `paper/` are git-ignored: they hold the downloaded data, firm-level results and unpublished drafts. Only `docs/` (aggregate, no firm names) is published.
