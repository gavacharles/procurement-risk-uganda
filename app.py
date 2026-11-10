"""Uganda Construction Procurement Risk Monitor: MVP dashboard.

Run:  ~/.venvs/procfraud/bin/streamlit run app.py
Data: produced by `python run_pipeline.py` into data/processed/.

Flags are risk indicators for review, not findings of wrongdoing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
P = ROOT / "data" / "processed"

st.set_page_config(page_title="Uganda Works Procurement Risk Monitor", layout="wide")


def ugx(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    for div, s in [(1e12, "tn"), (1e9, "bn"), (1e6, "m")]:
        if abs(x) >= div:
            return f"UGX {x / div:,.1f}{s}"
    return f"UGX {x:,.0f}"


@st.cache_data
def load():
    w = pd.read_parquet(P / "works_tenders_scored.parquet").assign(source="Tender record")
    bw = pd.read_parquet(P / "works_bids_scored.parquet").assign(source="Bid record")
    cols = ["source", "ocid", "award_date", "buyer_id", "buyer_name", "supplier_id", "supplier_name",
            "title", "subsector", "award_value_ugx", "risk_score", "n_flags", "reasons"]
    allw = pd.concat([w[cols + ["method", "estimate_ugx", "award_to_est", "tender_days"]], bw[cols + ["n_bids", "min_bid", "bid_cv"]]],
                     ignore_index=True)
    allw["year"] = allw.award_date.dt.year.where(allw.award_date.dt.year >= 2014)
    allw["priority_index"] = (allw.risk_score * np.log10(allw.award_value_ugx.clip(lower=1e6))).round(1)
    return dict(
        w=w, bw=bw, all=allw,
        sets=pd.read_parquet(P / "bidder_sets.parquet"),
        sup=pd.read_parquet(P / "supplier_profiles.parquet"),
        buy=pd.read_parquet(P / "buyer_profiles.parquet"),
        nodes=pd.read_parquet(P / "network_nodes_works.parquet"),
        edges=pd.read_parquet(P / "network_edges_works.parquet"),
        clusters=pd.read_parquet(P / "network_clusters_works.parquet"),
        variants=pd.read_parquet(P / "variant_cobids.parquet"),
    )


d = load()
A = d["all"]

# ---------------- sidebar filters ----------------
st.sidebar.title("Filters")
years = sorted(A.year.dropna().astype(int).unique())
yr = st.sidebar.slider("Award year", min(years), max(years), (min(years), max(years)))
subs = st.sidebar.multiselect("Sub-sector", sorted(A.subsector.dropna().unique()), default=None)
buyer = st.sidebar.multiselect("Procuring entity", sorted(A.buyer_name.dropna().unique()))
min_val = st.sidebar.number_input("Minimum award value (UGX m)", 0, 1_000_000, 0, step=10)
st.sidebar.caption("Source: PPDA OCDS via OCP Data Registry (publication 130). "
                   "Scores are risk indicators that call for review, not proof of wrongdoing.")

f = A[(A.year.between(*yr)) | A.year.isna()]
if subs:
    f = f[f.subsector.isin(subs)]
if buyer:
    f = f[f.buyer_name.isin(buyer)]
if min_val:
    f = f[f.award_value_ugx >= min_val * 1e6]

st.title("Uganda Construction Procurement Risk Monitor")
st.caption("Case study: public works contracts published by PPDA (2015 to present). MVP for audit prioritisation.")

tabs = st.tabs(["Overview", "Audit priority list", "Audit plan", "Collusion signals",
                "Suppliers", "Procuring entities", "Methodology"])

# ---------------- Overview ----------------
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Works records", f"{len(f):,}")
    c[1].metric("Award value", ugx(f.award_value_ugx.sum()))
    c[2].metric("Records with 3+ flags", f"{(f.n_flags >= 3).sum():,}")
    c[3].metric("Entities", f"{f.buyer_id.nunique():,}")
    c[4].metric("Suppliers", f"{f.supplier_id.nunique():,}")

    l, r = st.columns(2)
    by = f.dropna(subset=["year"]).groupby("year").agg(value=("award_value_ugx", "sum"), mean_risk=("risk_score", "mean")).reset_index()
    l.plotly_chart(px.bar(by, x="year", y="value", title="Works award value by year (UGX)"), width="stretch")
    r.plotly_chart(px.line(by, x="year", y="mean_risk", markers=True, title="Mean risk score by year"), width="stretch")

    w = d["w"]
    ratio = w.award_to_est[(w.award_to_est > 0.5) & (w.award_to_est < 1.5)]
    fig = px.histogram(ratio, nbins=100, title="Award ÷ official estimate: bunching just below 1.0 suggests leaked or adjusted estimates")
    fig.add_vline(x=1.0, line_color="red")
    fig.update_layout(showlegend=False, xaxis_title="award / estimate")
    l.plotly_chart(fig, width="stretch")

    fl = pd.concat([d["w"].filter(regex="^f_").astype(object).fillna(False).astype(bool).mean(), d["bw"].filter(regex="^f_").astype(object).fillna(False).astype(bool).mean()])
    fl = fl.groupby(level=0).max().sort_values()
    fl.index = fl.index.str.replace("f_", "").str.replace("_", " ")
    r.plotly_chart(px.bar(fl, orientation="h", title="How often each red flag fires").update_layout(showlegend=False, xaxis_tickformat=".0%"),
                   width="stretch")

# ---------------- Priority list ----------------
with tabs[1]:
    st.subheader("Works contracts ranked for review")
    rank_by = st.radio("Rank by", ["Priority index (risk × log value)", "Risk score", "Award value"], horizontal=True)
    key = {"Priority index (risk × log value)": "priority_index", "Risk score": "risk_score", "Award value": "award_value_ugx"}[rank_by]
    min_flags = st.slider("Minimum number of flags", 0, 6, 1)
    t = f[f.n_flags >= min_flags].sort_values(key, ascending=False)
    show = t[["priority_index", "risk_score", "n_flags", "award_date", "buyer_name", "supplier_name", "title",
              "award_value_ugx", "source", "reasons", "ocid"]].copy()
    show["award_date"] = show.award_date.dt.date
    st.dataframe(show.head(500), width="stretch", hide_index=True,
                 column_config={"award_value_ugx": st.column_config.NumberColumn("Award (UGX)", format="%,.0f"),
                                "reasons": st.column_config.TextColumn("Why flagged", width="large")})
    st.download_button("Download full list (CSV)", t.to_csv(index=False).encode(), "works_priority_list.csv", "text/csv")

    st.markdown("#### Contract detail")
    pick = st.selectbox("Select a contract (OCID)", t.ocid.head(200))
    if pick:
        row = t[t.ocid == pick].iloc[0]
        st.write(f"**{row.title}**")
        st.write(f"{row.buyer_name} → {row.supplier_name} · {ugx(row.award_value_ugx)} · {row.source}")
        for reason in str(row.reasons).split("; "):
            if reason:
                st.warning(reason)
        st.caption(f"Look up on the PPDA portal: https://gpp.ppda.go.ug  (OCID {row.ocid})")

# ---------------- Audit plan ----------------
with tabs[2]:
    st.subheader("Audit plan optimiser")
    st.write("Choose the contracts that **maximise expected value at risk** within your audit capacity. "
             "Expected value at risk = award value × probability proxy (risk score / 100). "
             "Each audit has a cost that grows with contract size.")
    c1, c2, c3 = st.columns(3)
    budget_days = c1.number_input("Audit capacity (auditor-days)", 10, 5000, 200, step=10)
    base = c2.number_input("Days per audit: base", 1, 60, 5)
    per_log = c3.number_input("Extra days per 10× contract value", 0, 30, 3)
    cand = f.dropna(subset=["award_value_ugx"]).copy()
    cand = cand[cand.award_value_ugx > 0]
    cand["exp_at_risk"] = cand.award_value_ugx * cand.risk_score / 100
    cand["audit_days"] = base + per_log * np.maximum(0, np.log10(cand.award_value_ugx) - 6)
    # greedy knapsack on value density (near-optimal for many small items)
    cand["density"] = cand.exp_at_risk / cand.audit_days
    cand = cand.sort_values("density", ascending=False)
    cand["cum_days"] = cand.audit_days.cumsum()
    plan = cand[cand.cum_days <= budget_days]
    m = st.columns(4)
    m[0].metric("Contracts selected", f"{len(plan):,}")
    m[1].metric("Auditor-days used", f"{plan.audit_days.sum():,.0f}")
    m[2].metric("Award value covered", ugx(plan.award_value_ugx.sum()))
    m[3].metric("Expected value at risk covered", ugx(plan.exp_at_risk.sum()))
    st.dataframe(plan[["buyer_name", "supplier_name", "title", "award_value_ugx", "risk_score", "audit_days", "exp_at_risk", "reasons"]],
                 width="stretch", hide_index=True)
    st.download_button("Download audit plan (CSV)", plan.to_csv(index=False).encode(), "audit_plan.csv", "text/csv")

# ---------------- Collusion ----------------
with tabs[3]:
    st.subheader("Repeated bidder groups")
    st.write("Identical groups of firms bidding together in 3+ procurements. Where the win **rotates** among members, "
             "this is a classic bid-rotation pattern. It can also mean a fixed shortlist, which still undermines competition.")
    sets = d["sets"]
    only_works = st.checkbox("Only groups that bid on works", True)
    s = sets[sets.any_works] if only_works else sets
    s = s[["bidder_names", "set_size", "n_procurements", "n_distinct_winners", "n_buyers", "rotating", "first", "last"]]
    st.dataframe(s, width="stretch", hide_index=True)

    st.subheader("Construction co-bidding network")
    nodes, edges, clusters = d["nodes"], d["edges"], d["clusters"]
    cl_opts = clusters[clusters.internal_procurements >= 3].cluster.tolist()
    cl = st.selectbox("Cluster", cl_opts, format_func=lambda i: (
        lambda r: f"Cluster {i}: {r.n_firms} firms, {r.internal_procurements} joint procurements, {r.n_winners} winners")(
        clusters.set_index("cluster").loc[i]))
    sub_n = nodes[nodes.cluster == cl]
    ids = set(sub_n.party_id)
    sub_e = edges[edges.source.isin(ids) & edges.target.isin(ids)]
    if len(sub_n) and len(sub_n) <= 150:
        import networkx as nx
        G = nx.Graph()
        G.add_weighted_edges_from(sub_e[["source", "target", "weight"]].itertuples(index=False))
        pos = nx.spring_layout(G, seed=1, weight="weight")
        ex, ey = [], []
        for a, b in G.edges():
            ex += [pos[a][0], pos[b][0], None]
            ey += [pos[a][1], pos[b][1], None]
        nn = sub_n.set_index("party_id").loc[list(G.nodes)]
        fig = go.Figure([
            go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#999"), hoverinfo="none"),
            go.Scatter(x=[pos[n][0] for n in G.nodes], y=[pos[n][1] for n in G.nodes], mode="markers",
                       marker=dict(size=8 + 25 * nn.win_rate.fillna(0), color=nn.n_wins, colorscale="Reds", showscale=True,
                                   colorbar=dict(title="wins")),
                       text=[f"{r.party_name}<br>bids {r.n_bids}, wins {r.n_wins}" for r in nn.itertuples()], hoverinfo="text"),
        ])
        fig.update_layout(showlegend=False, height=520, xaxis_visible=False, yaxis_visible=False,
                          title="Node size = win rate, colour = number of wins, links = bid together")
        st.plotly_chart(fig, width="stretch")
    st.dataframe(sub_n.sort_values("eigenvector_centrality", ascending=False)
                 [["party_name", "n_bids", "n_wins", "win_rate", "degree", "co_bid_count", "eigenvector_centrality"]],
                 width="stretch", hide_index=True)

    st.subheader("Name-variant co-bidding")
    st.write("Two party IDs whose names normalise to the same firm, bidding in the same procurement. "
             "This is either a data duplicate or one firm bidding under several names.")
    st.dataframe(d["variants"], width="stretch", hide_index=True)

# ---------------- Suppliers ----------------
with tabs[4]:
    st.subheader("Supplier profiles (works)")
    sup = d["sup"]
    q = st.text_input("Search supplier")
    s = sup[sup.supplier_name.str.contains(q, case=False, na=False)] if q else sup
    st.dataframe(s.sort_values("works_value_ugx", ascending=False), width="stretch", hide_index=True,
                 column_config={"works_value_ugx": st.column_config.NumberColumn("Works value (UGX)", format="%,.0f")})
    if q and len(s):
        st.dataframe(f[f.supplier_id.isin(s.supplier_id)][["award_date", "buyer_name", "title", "award_value_ugx", "risk_score", "reasons"]],
                     width="stretch", hide_index=True)

# ---------------- Entities ----------------
with tabs[5]:
    st.subheader("Procuring entity risk profile")
    buy = d["buy"]
    min_rec = st.slider("Minimum works records", 1, 100, 10)
    b = buy[buy.works_records >= min_rec].sort_values("mean_risk", ascending=False)
    st.plotly_chart(px.scatter(b, x="works_value_ugx", y="mean_risk", size="works_records", hover_name="buyer_name",
                               log_x=True, title="Entities: works value vs mean risk"), width="stretch")
    st.dataframe(b, width="stretch", hide_index=True)

    pick = st.selectbox("Benford check for entity", b.buyer_name)
    if pick:
        bid = b[b.buyer_name == pick].buyer_id.iloc[0]
        v = A[(A.buyer_id == bid) & (A.award_value_ugx > 0)].award_value_ugx
        if len(v) >= 20:
            first = v.map(lambda x: int(f"{x:e}"[0]))
            obs = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
            exp = np.log10(1 + 1 / np.arange(1, 10))
            fig = go.Figure([go.Bar(x=list(range(1, 10)), y=obs.values, name="observed"),
                             go.Scatter(x=list(range(1, 10)), y=exp, name="Benford", mode="lines+markers")])
            fig.update_layout(title=f"First-digit distribution: {pick} (works, n={len(v)})")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Too few works awards for a Benford test (need ≥20).")

# ---------------- Methodology ----------------
with tabs[6]:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from red_flags import FLAGS
    st.markdown("""
**Data.** PPDA Government Procurement Portal OCDS feed, retrieved via the Open Contracting Data Registry.
Construction works are identified from titles (`src/classify.py`) because the feed has no procurement category.

**Two record families.** *Tender records* (estimate, method, window, award) and *bid records* (all bids and bidders)
are scored with different flag sets. Each score is the weighted sum of fired flags, scaled 0–100 against that family's
maximum.

**Priority index** = risk score × log₁₀(award value). It weights risk by the money at stake.

**Caveats.** Thresholds are first-pass assumptions to be validated against the PPDA Regulations and with auditors.
Some bid records may hold only the winning bid. Supplier IDs are deduplicated by name only, pending URSB matching.
""")
    st.table(pd.DataFrame([(k, v[0], v[1]) for k, v in FLAGS.items()], columns=["flag", "weight", "meaning"]))
