# Uganda Construction Procurement Risk Monitor

**By Charles Gava** (concept, research design and direction) · [Project site](https://gavacharles.github.io/procurement-risk-uganda/) · [How to cite](CITATION.cff)

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
| Machine learning + NLP | `src/ml_models.py` | `ml_anomaly`, `split_candidates`, `paper/ml_results.json` |
| Blind evaluation (number theory) | `src/blind_eval.py` | prototype + `tests/test_blind_eval.py` |
| District map data | `src/geo.py` | `docs/data/districts.geojson` |
| PPDA bid invitations crawl | `src/fetch_bid_invitations.py` | `bid_invitations` (≈6 h, resumable) |
| Dashboard | `app.py` | overview, audit priority list, audit plan optimiser, collusion signals, supplier and entity profiles |

## Mathematics and machine learning used

| Principle (from the concept note) | Where |
|---|---|
| Statistics & inference: bunching estimator, bootstrap, robust outliers | `analysis.py`, `red_flags.py` |
| Benford's Law | `red_flags.py` |
| Graph theory / network science: co-bidding network, Louvain communities | `network.py` |
| Linear algebra / spectral analysis: eigenvector centrality | `network.py` |
| Optimisation: knapsack audit planner; least-squares counterfactual | `app.py`, `analysis.py` |
| Machine learning (unsupervised): Isolation Forest | `ml_models.py` |
| NLP: works classifier; TF-IDF contract-splitting detector | `classify.py`, `ml_models.py` |
| Number theory & finite fields: blind bid codes, private conflict-of-interest check | `blind_eval.py` |
| Supervised ML (fraud probability) | planned: needs labelled cases |

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
