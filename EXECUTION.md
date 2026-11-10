# Procurement Fraud Analytics: Execution Plan

**Sector:** Public construction procurement
**Case study:** Uganda
**End goal:** A product that oversight institutions actually use
**Source:** `Fraud Analytics.docx` (concept notes)
**Last updated:** 2026-11-10

---

## 1. Product focus

> **A risk-scoring and audit-prioritisation tool for public construction contracts in Uganda.** It flags suspicious tenders, suppliers and contracts, explains why each was flagged, and ranks which ones auditors should look at first.

The concept notes cover a wide range: blind bid review, network analysis, spectral methods, Benford's Law, ML and NLP. The MVP narrows this to one problem: **flag and rank risky construction contracts.** The other capabilities come in later phases.

### Target users
| Priority | User | Need |
|---|---|---|
| Primary | Office of the Auditor General (OAG) | Choose which contracts to audit; cut audit costs and catch more fraud |
| Primary | PPDA | Monitor procuring entities and providers; support suspensions |
| Secondary | Inspectorate of Government (IG) | Leads for investigations |
| Secondary | CoST Uganda, CSOs, media | Transparency and monitoring of infrastructure spending |
| Secondary | Works-heavy entities: Ministry of Works & Transport (absorbed UNRA), KCCA, districts | Internal control |
| Later | Development partners (World Bank, AfDB, etc.) | Fiduciary risk on the infrastructure they fund |

### Why construction
- Contract values are high.
- Contracts change a lot after award (variation orders, time extensions).
- Costs can be compared against benchmarks (e.g., cost per km of road, cost per m² of a classroom).
- Collusion and bid rigging are known problems in the sector.

---

## 2. Data sources (Uganda)

| Source | Contents | Access | Use |
|---|---|---|---|
| PPDA Government Procurement Portal (gpp.ppda.go.ug) / eGP (egpuganda.go.ug) | Tenders, bidders, awards, contract values in the Open Contracting Data Standard (OCDS) format | Public. Both portals respond. **The open-data API endpoint is not confirmed yet** (`/adminapi/public/api/open-data/v1/releases` returned "Resource Not Found") | Core data |
| URSB business registry | Directors, shareholders, addresses, registration dates | Partly public; bulk access probably needs an agreement | Entity resolution, network graph |
| PPDA suspended-providers list | Debarred firms | Public | Labels |
| World Bank / AfDB debarment lists | Debarred firms worldwide | Public | Labels, cross-checks |
| OAG annual and value-for-money audit reports | Flagged contracts and amounts lost | Public PDFs | Labels, extracted with NLP |
| CoST Uganda disclosures | Project data: cost and time overruns, change orders | Public | Construction-specific features |
| IFMS (payments) | Payments matched to contracts | **Restricted**; needs an agreement with government | Duplicate and early payments |

---

## 3. Red flags and the method behind each

### Rule-based (explainable; build first)
- Only one bidder
- Short bidding window (below the statutory minimum)
- Direct procurement / single sourcing in works
- Contracts split just under procurement-method thresholds
- Large variation orders (e.g., more than 15–25% of the original value)
- Repeated time extensions
- Newly registered firm winning a large contract
- Winner is not the lowest compliant bidder
- Award or contract data missing or inconsistent

### Statistics and statistical inference
- Bids clustered just under the engineer's estimate
- Cost-per-unit outliers (per km of road, per m² of building) by region and contract type
- **Benford's Law** and last-digit tests on bid and contract values
- Duplicate or near-duplicate payments
- One supplier holding an unusual share of an entity's contracts (HHI)

### Graph theory and network science
- Bipartite graph: supplier – director / shareholder – address – phone – bank account
- Bidders sharing directors or addresses in the same tender, a sign of collusion
- Winners taking turns, and "cover" bidders who keep losing to the same firm
- Community detection to find bidding cartels

### Linear algebra (spectral analysis)
- Bidder × tender co-bidding matrix; **eigenvector centrality** to find influential firms in collusive networks
- Matrix and tensor anomaly detection on entity × supplier × time data

### Optimisation and calculus
- **Audit-prioritisation optimiser:** knapsack-style; picks the contracts that maximise expected recovered loss within the audit budget
- Models of how risk changes over time and across procurement systems

### Machine learning
- Unsupervised first: isolation forest and other anomaly detection
- Supervised later: fraud-probability model trained on labels from OAG, PPDA and IG findings
- NLP: extract findings from OAG reports; analyse tender and evaluation documents

### Number theory and abstract algebra (finite fields)
- Cryptographic pseudonymisation of bidder identities for blind evaluation (Phase 5)

---

## 4. Phased roadmap

### Phase 0: Discovery (3–4 weeks)
- [ ] Confirm the PPDA/eGP open-data API endpoint and the OCDS bulk download
- [ ] Assess URSB data access (bulk or API)
- [ ] Gather OAG audit reports (last 5+ years), focusing on works and roads
- [ ] Interview 3–5 target users (OAG and PPDA first) about workflow, pain points and how they pick audits today
- [ ] Legal check: Data Protection and Privacy Act 2019 (director data is personal data)
- [ ] Choose initial scope: e.g., roads and buildings works contracts, central government plus KCCA, 2018 onwards

### Phase 1: Data foundation
- [ ] OCDS ingestion pipeline (incremental, reproducible)
- [ ] Filter to works / construction contracts
- [ ] Data quality report: completeness, missing fields, coverage by entity and year
- [ ] Supplier entity resolution (name normalisation, fuzzy matching, URSB IDs)
- [ ] Database schema for tenders, bids, awards, contracts, amendments and suppliers

### Phase 2: MVP (first usable product)
- [ ] Implement 15–20 explainable red flags
- [ ] Combined contract risk score, with the contribution of each flag shown
- [ ] Supplier and procuring-entity risk profiles
- [ ] Dashboard: ranked contract list, drill-down, reasons for each flag, export for auditors
- [ ] Validate with users: review the top-ranked contracts with OAG/PPDA staff
- [ ] **Milestone: demo to OAG / PPDA / CoST Uganda**

### Phase 3: Network module
- [ ] Supplier–director–address graph
- [ ] Collusion-cluster detection, co-bidding analysis, centrality scores
- [ ] Network visualisation in the dashboard

### Phase 4: ML, NLP and optimisation
- [ ] NLP extraction of labels from OAG reports
- [ ] Supervised fraud-probability model, assessed against those labels
- [ ] Audit-prioritisation optimiser that takes the budget as a constraint
- [ ] Integrate payment data (IFMS) if a partnership has been secured

### Phase 5: Blind bid evaluation (separate product line)
- [ ] Anonymise bids: remove bidder identities, company names and logos
- [ ] Assign each bidder a code name and cryptographic identifier
- [ ] Workflow integration with eGP; needs PPDA buy-in

### Phase 6: Scale
- [ ] Extend to other sectors (health, education supplies) and other countries using OCDS
- [ ] Training and capacity building for investigators (the "human capital" point in the concept notes)

---

## 5. Technical stack

| Layer | MVP | Later |
|---|---|---|
| Language | Python | Python |
| Storage | DuckDB / Parquet | PostgreSQL (+ graph DB if needed) |
| Analytics | pandas, scipy, scikit-learn | + PyTorch / gradient boosting |
| Graph | NetworkX | Neo4j / igraph at scale |
| NLP | spaCy, PDF extraction | LLM-assisted extraction |
| Optimisation | PuLP / OR-Tools | |
| UI | Streamlit | FastAPI + React |
| Deployment | Local / single VM | Hosted, role-based access, audit logs |

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Access to data and buy-in from institutions** (biggest risk) | Build a credible demo on public OCDS data first; use it to secure partnerships with OAG, PPDA or CoST Uganda |
| Poor quality or incomplete data | Data quality report in Phase 1; adjust which red flags are used to match the fields available |
| Few labels for supervised ML | Rules and unsupervised methods first; extract labels from OAG reports and debarment lists |
| False positives damaging reputations | Present output as risk indicators, not accusations; explain every flag; keep a human in the loop |
| Data protection (director data) | Compliance review in Phase 0; data minimisation; access controls |
| Users don't adopt the product | Co-design with auditors; fit their existing workflow; measure the hit rate of top-ranked contracts |

---

## 7. Success metrics

- **Data:** share of works contracts ingested with complete key fields
- **Model:** precision of the top-N ranked contracts against known audit findings
- **Product:** number of active institutional users; audits started from the tool's rankings
- **Impact:** value of losses prevented or recovered that can be attributed to the tool's flags

---

## 8. Progress log

### 2026-11-10: Phase 1 and a first MVP built on public data
**Done**
- [x] Data source confirmed: the PPDA OCDS feed via the OCP Data Registry (publication 130), a monthly bulk download of ~237k records from 2015 onwards. The PPDA API at `cdn.ppda.go.ug/api` was also mapped (public: `/signed-contracts`, `/contracts-variations-prices`, OCDS overview).
- [x] Ingestion pipeline (`src/ingest.py`) producing tender, bid, bidder and award tables
- [x] Works classifier from titles (~90% precision on spot checks): 5,400 works tender records and 931 works bid records
- [x] Supplier name resolution: 24,707 party IDs → 16,561 firms
- [x] 18 explainable red flags, risk score, and priority index (risk × log value)
- [x] Repeated bidder groups and rotation detection; construction co-bidding network (Louvain clusters, eigenvector centrality)
- [x] Exploratory findings report (`reports/exploratory_report.md`)
- [x] Streamlit MVP dashboard (`app.py`) with priority list, audit-plan optimiser, collusion view, supplier and entity profiles, and Benford check

**Key findings to date** (to be validated)
- 23% of works awards fall within 1% below the official estimate, a sign of possible estimate leakage or estimates revised after award.
- One group of three firms bid together in 255 procurements of a single entity and rotated the wins (firms anonymised in public outputs).
- A third of supplier IDs are duplicates, so URSB matching is needed.
- The single-bid rate (41.5%) is probably inflated by records that hold only the winning bid.

**Data constraints found**
- The feed splits tender records and bid records with no shared ID.
- No procurement category field.
- Some dates are implausible (1970).
- Contract variations are not in the OCDS feed.

### 2026-11-10 (later): ML, cryptography, map, credit
- [x] Isolation Forest anomaly model (32% top-decile overlap with rules, 3× chance; 298 extra anomalies)
- [x] NLP contract-splitting detector (21 possible splits across 13 entities)
- [x] Blind bid evaluation and private conflict-of-interest check (modular exponentiation over a 2048-bit prime field), with tests
- [x] Interactive district map on the public site (67 districts with data)
- [x] Site section mapping each concept-note principle to code and status; author credit and CITATION.cff
- [ ] Paper 1 test: plan/invitation-stage vs award-stage estimate, and estimates entered before vs after award (PPDA bid-invitations crawl in progress)

## 9. Immediate next steps
1. Validate thresholds against the PPDA Regulations (bidding periods, method thresholds).
2. Hand-label ~300 titles to measure classifier precision and recall.
3. Ingest `/contracts-variations-prices` from the PPDA API (cost overruns, a key construction flag).
4. Link tender and bid records using PPDA API IDs.
5. Build a label set: PPDA suspended providers and OAG works findings (NLP on audit reports).
6. Demo the dashboard to OAG / PPDA / CoST Uganda for feedback.
