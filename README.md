# Uganda Construction Procurement Risk Monitor

Fraud-risk analytics for public construction (works) procurement, with Uganda as the first case study.
See [EXECUTION.md](EXECUTION.md) for the plan and [reports/exploratory_report.md](reports/exploratory_report.md) for current findings.

> Scores and flags are **risk indicators that call for review**. They are not findings of wrongdoing.

## Quick start

```bash
# one-time setup (the virtualenv lives outside OneDrive to avoid syncing thousands of files)
python3.12 -m venv ~/.venvs/procfraud
~/.venvs/procfraud/bin/pip install -r requirements.txt

# refresh data and rebuild everything (~1 min)
~/.venvs/procfraud/bin/python run_pipeline.py --download

# open the dashboard
~/.venvs/procfraud/bin/streamlit run app.py
```

## Data source

PPDA Government Procurement Portal, published in the Open Contracting Data Standard (OCDS) format and
mirrored monthly by the [OCP Data Registry, publication 130](https://data.open-contracting.org/en/publication/130).
About 237k records (2015 onwards).

PPDA's own API (`https://cdn.ppda.go.ug/api`) has extra public endpoints not yet used:
`/signed-contracts`, `/contracts-variations-prices`, `/contract-nature`, and the OCDS export
(`/open-data/v2/ocds/...`).

## Pipeline

| Step | File | Output (`data/processed/`) |
|---|---|---|
| Flatten OCDS | `src/ingest.py` | `tenders`, `bid_procs`, `bids`, `tenderers` |
| Supplier name resolution | `src/entities.py` | `entity_keys`, `variant_cobids` |
| Works classifier | `src/classify.py` | (used by later steps) |
| Red flags + risk score | `src/red_flags.py` | `works_tenders_scored`, `works_bids_scored`, `bidder_sets`, `supplier_profiles`, `buyer_profiles` |
| Co-bidding network | `src/network.py` | `network_{nodes,edges,clusters}[_works]` |
| Findings report | `src/report.py` | `reports/exploratory_report.md`, `reports/top_risk_works_contracts.csv` |
| Dashboard | `app.py` | overview, audit priority list, audit plan optimiser, collusion signals, supplier and entity profiles |

## Layout

```
data/raw/          downloaded OCDS bulk file (not edited)
data/processed/    parquet tables built by the pipeline
src/               pipeline modules
reports/           generated report, figures, CSV exports
app.py             Streamlit dashboard
run_pipeline.py    end-to-end runner
```

## Public site and privacy

`docs/` is the GitHub Pages site. It shows aggregate findings only and does not name firms.
Locally generated outputs (`reports/`, `paper/`, `data/`) are git-ignored: they contain firm-level
results and unpublished paper drafts. Treat all flags as risk indicators that call for review, not as findings.
