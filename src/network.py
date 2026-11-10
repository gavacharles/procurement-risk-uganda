"""Co-bidding network: find suspicious supplier clusters (possible cartels / cover bidding).

Graph: nodes = tenderers (all sectors, from bid records); an edge joins two firms
that bid in the same procurement, weighted by how often they do.

Per-firm indicators:
  eigenvector_centrality  influence within the co-bidding network (spectral method)
  degree, co_bid_count    breadth / intensity of co-bidding
  win_rate                share of its bids that it won
Per-cluster indicators (Louvain communities):
  n_firms, internal_procurements, win_gini (how evenly wins are shared -> rotation),
  repeat_pair_ratio (how often the same firms meet), any_works

Clusters where a small, closed group repeatedly bids together and wins are
shared out evenly are a classic bid-rotation signature.

Outputs: data/processed/network_{nodes,edges,clusters}.parquet (all sectors)
         data/processed/network_{nodes,edges,clusters}_works.parquet (construction firms)
"""
from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import tag  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0 or len(x) < 2:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main(scope: str = "all", min_edge_weight: int = 2) -> None:
    """scope="all": every tenderer. scope="works": only co-bidding on works procurements."""
    tdr = pd.read_parquet(P / "tenderers.parquet").dropna(subset=["party_id"])
    procs = tag(pd.read_parquet(P / "bid_procs.parquet"))
    # collapse name-variant party IDs onto one canonical firm (see entities.py)
    keys = pd.read_parquet(P / "entity_keys.parquet")
    pid2canon = keys.set_index("party_id").canon.dropna().to_dict()
    canon_name = keys.dropna(subset=["canon"]).drop_duplicates("canon").set_index("canon").party_name
    tdr["party_id"] = tdr.party_id.map(lambda x: pid2canon.get(x, x))
    procs["supplier_id"] = procs.supplier_id.map(lambda x: pid2canon.get(x, x))
    winner = procs.set_index("ocid").supplier_id
    works_ocids = set(procs.loc[procs.is_works, "ocid"])

    groups = tdr.groupby("ocid").party_id.apply(lambda s: sorted(set(s)))
    groups = groups[groups.map(len) >= 2]
    if scope == "works":
        # only co-bidding that happens on works procurements
        groups = groups[groups.index.isin(works_ocids)]
        min_edge_weight = 1

    edges: Counter = Counter()
    for members in groups:
        for a, b in itertools.combinations(members, 2):
            edges[(a, b)] += 1

    G = nx.Graph()
    for (a, b), wgt in edges.items():
        if wgt >= min_edge_weight:
            G.add_edge(a, b, weight=wgt)
    print(f"[{scope}] graph: {G.number_of_nodes():,} firms, {G.number_of_edges():,} repeat co-bidding links")

    # spectral centrality, per connected component (eigenvector centrality is
    # ill-defined on disconnected graphs)
    eig = {}
    for comp in nx.connected_components(G):
        sg = G.subgraph(comp)
        if len(sg) < 3:
            eig.update({n: 0.0 for n in sg})
            continue
        try:
            c = nx.eigenvector_centrality_numpy(sg, weight="weight")
        except Exception:
            c = {n: 0.0 for n in sg}
        # scale by component size so small isolated cliques do not all score 1.0
        eig.update({n: v * len(sg) / G.number_of_nodes() for n, v in c.items()})

    comms = nx.community.louvain_communities(G, weight="weight", seed=42)
    node_comm = {n: i for i, c in enumerate(comms) for n in c}

    bids_per = tdr.groupby("party_id").ocid.nunique()
    wins_per = procs.supplier_id.value_counts()
    names = tdr.drop_duplicates("party_id").set_index("party_id").party_name
    names.update(canon_name)
    nodes = pd.DataFrame({"party_id": list(G.nodes)})
    nodes["party_name"] = nodes.party_id.map(names)
    nodes["cluster"] = nodes.party_id.map(node_comm)
    nodes["degree"] = nodes.party_id.map(dict(G.degree()))
    nodes["co_bid_count"] = nodes.party_id.map(dict(G.degree(weight="weight")))
    nodes["eigenvector_centrality"] = nodes.party_id.map(eig)
    nodes["n_bids"] = nodes.party_id.map(bids_per).fillna(0).astype(int)
    nodes["n_wins"] = nodes.party_id.map(wins_per).fillna(0).astype(int)
    nodes["win_rate"] = nodes.n_wins / nodes.n_bids.replace(0, np.nan)

    # cluster-level stats
    ocid_members = groups.to_dict()
    rows = []
    for i, members in enumerate(comms):
        mem = set(members)
        internal = [o for o, m in ocid_members.items() if len(set(m) & mem) >= 2]
        wins = Counter(winner.get(o) for o in internal if winner.get(o) in mem)
        win_counts = np.array([wins.get(m, 0) for m in mem])
        sg = G.subgraph(mem)
        rows.append(dict(
            cluster=i,
            n_firms=len(mem),
            internal_procurements=len(internal),
            works_procurements=sum(o in works_ocids for o in internal),
            density=nx.density(sg) if len(mem) > 1 else np.nan,
            total_edge_weight=sg.size(weight="weight"),
            n_winners=int((win_counts > 0).sum()),
            win_gini=gini(win_counts),
            members=" | ".join(sorted(str(names.get(m, m)) for m in mem))[:2000],
        ))
    clusters = pd.DataFrame(rows)
    # suspicion: small dense groups, many joint procurements, wins shared out
    clusters["rotation_signal"] = (
        (clusters.n_firms.between(3, 8))
        & (clusters.density >= 0.6)
        & (clusters.internal_procurements >= 5)
        & (clusters.n_winners >= 2)
    )
    clusters = clusters.sort_values(["rotation_signal", "internal_procurements"], ascending=False)

    e = pd.DataFrame([(a, b, d["weight"]) for a, b, d in G.edges(data=True)], columns=["source", "target", "weight"])
    sfx = "" if scope == "all" else f"_{scope}"
    nodes.to_parquet(P / f"network_nodes{sfx}.parquet", index=False)
    e.to_parquet(P / f"network_edges{sfx}.parquet", index=False)
    clusters.to_parquet(P / f"network_clusters{sfx}.parquet", index=False)
    print(f"clusters: {len(clusters):,} | with rotation signal: {int(clusters.rotation_signal.sum()):,}")
    print(clusters.head(10)[["cluster", "n_firms", "density", "internal_procurements",
                             "works_procurements", "n_winners", "win_gini"]].to_string(index=False))
    print("\nmost central firms:")
    print(nodes.nlargest(10, "eigenvector_centrality")[["party_name", "degree", "co_bid_count",
                                                       "n_bids", "n_wins", "eigenvector_centrality"]].to_string(index=False))


if __name__ == "__main__":
    main("all")
    print()
    main("works")
