"""Static charts and chart animations for the posts and the site (no firm names).

Palette (validated with the dataviz validator, all checks pass):
  observed / primary  #2a78d6 (blue)    expected / secondary  #eb6834 (orange)
  surface #fcfcfb, text #0b0b0b / #52514e, grid #e4e2dc

Out: posts/media/*.png, *.mp4, *.gif     (copies of selected items -> docs/media/)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import bunching  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "posts" / "media"
SITE = ROOT / "docs" / "media"
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})
CREDIT = "Gava Charles · gavacharles.github.io/procurement-risk-uganda"


def title(fig, t, sub=None):
    fig.text(0.04, 0.95, t, fontsize=17, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.04, 0.885, sub, fontsize=11.5, color=INK2, va="top")
    fig.text(0.04, 0.025, CREDIT, fontsize=8.5, color=INK2)
    fig.text(0.96, 0.025, "Source: PPDA open contracting data", fontsize=8.5, color=INK2, ha="right")


def save_writer(anim, name, fps):
    # render outside the synced folder (cloud sync can lock files mid-write), then copy in
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mp4, gif = Path(td) / f"{name}.mp4", Path(td) / f"{name}.gif"
        anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=160)
        anim.save(gif, writer=animation.PillowWriter(fps=fps), dpi=80)
        for f in (mp4, gif):
            dst = OUT / f.name
            dst.unlink(missing_ok=True)
            shutil.copyfile(f, dst)


# ------------------------------------------------------------------ 1. bunching
def chart_bunching(animate=True):
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    b = bunching(w.award_to_est.dropna(), n_boot=50)
    x, cnt, pred = b["mids"], b["cnt"], b["pred"]

    def frame(ax, k_bars, show_line, show_note):
        ax.clear()
        ax.bar(x[:k_bars], cnt[:k_bars], width=0.0045, color=BLUE, label="Contracts observed")
        if show_line:
            ax.plot(x, pred, color=ORANGE, lw=2, label="Expected if prices were smooth")
        ax.axvspan(0.97, 1.0, color=ORANGE, alpha=0.08, lw=0)
        ax.set_xlim(0.8, 1.2)
        ax.set_ylim(0, cnt.max() * 1.15)
        ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
        ax.set_xticklabels(["80%", "90%", "100%\n(estimate)", "110%", "120%"])
        ax.set_xlabel("Winning price as a share of the government's own estimate")
        ax.set_ylabel("Works contracts")
        ax.grid(axis="x", visible=False)
        if show_line:
            ax.legend(loc="upper left", frameon=False)
        if show_note:
            ax.annotate("1,976 contracts here\nvs ~316 expected\n= 5.3× too many",
                        xy=(x[np.argmax(cnt)] + 0.003, cnt.max() * 0.97), xytext=(1.05, cnt.max() * 0.80),
                        fontsize=12, fontweight="bold", color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=1))

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_axes([0.10, 0.22, 0.86, 0.58])
    title(fig, "Winning prices pile up just below the official estimate",
          "Uganda public works tenders, 2015–2026 · statistics & calculus: bunching estimator")
    frame(ax, len(x), True, True)
    fig.savefig(OUT / "chart_bunching.png", dpi=200)
    if animate:
        n = len(x)
        steps = list(range(1, n + 1, 2)) + [n] * 12 + [-1] * 14 + [-2] * 30

        def upd(i):
            s = steps[i]
            if s > 0:
                frame(ax, s, False, False)
            elif s == -1:
                frame(ax, n, True, False)
            else:
                frame(ax, n, True, True)
        save_writer(animation.FuncAnimation(fig, upd, frames=len(steps)), "anim_bunching", fps=12)
    plt.close(fig)
    return b


# ------------------------------------------------------------------ 2. network
def chart_network(animate=True):
    tdr = pd.read_parquet(P / "tenderers.parquet").dropna(subset=["party_id"])
    keys = pd.read_parquet(P / "entity_keys.parquet")
    pid2c = keys.set_index("party_id").canon.dropna().to_dict()
    tdr["firm"] = tdr.party_id.map(lambda x: pid2c.get(x, x))
    sets = pd.read_parquet(P / "bidder_sets.parquet")
    rot = sets[sets.rotating & sets.any_works]
    rot_firms = {pid2c.get(p, p) for s in rot.bidder_set for p in s.split(",")}
    procs = pd.read_parquet(P / "bid_procs.parquet")
    sys.path.insert(0, str(ROOT / "src"))
    from classify import is_works
    works_ocids = set(procs.loc[procs.title.map(is_works), "ocid"])
    t = tdr[tdr.ocid.isin(works_ocids)]
    yr = t.groupby("ocid").release_date.min().dt.year
    groups = t.groupby("ocid").firm.apply(lambda s: sorted(set(s)))
    groups = groups[groups.map(len) >= 2]
    edges = []
    for oc, mem in groups.items():
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                edges.append((mem[i], mem[j], int(yr.get(oc, 0))))
    E = pd.DataFrame(edges, columns=["a", "b", "year"])
    W = E.groupby(["a", "b"]).agg(weight=("year", "size"), first=("year", "min")).reset_index()
    G = nx.Graph()
    G.add_weighted_edges_from(W[["a", "b", "weight"]].itertuples(index=False))
    big = max(nx.connected_components(G), key=len)
    H = G.subgraph(big).copy()
    pos = nx.spring_layout(H, seed=3, k=0.18, iterations=200, weight="weight")
    first_year = {(r.a, r.b): r.first for r in W.itertuples()}
    deg = dict(H.degree(weight="weight"))
    years = sorted({y for y in W["first"] if y > 0})

    def draw(ax, upto):
        ax.clear()
        ax.set_axis_off()
        es = [(a, b) for a, b in H.edges() if first_year.get((a, b), first_year.get((b, a), 0)) <= upto]
        ns = {n for e in es for n in e}
        nx.draw_networkx_edges(H, pos, edgelist=es, ax=ax, edge_color="#b9b7b0", width=0.5, alpha=0.7)
        other = [n for n in ns if n not in rot_firms]
        rotn = [n for n in ns if n in rot_firms]
        nx.draw_networkx_nodes(H, pos, nodelist=other, ax=ax, node_color=BLUE, edgecolors=SURF, linewidths=0.8,
                               node_size=[10 + 3 * deg[n] for n in other])
        nx.draw_networkx_nodes(H, pos, nodelist=rotn, ax=ax, node_color=ORANGE, edgecolors=SURF, linewidths=0.8,
                               node_size=[14 + 3 * deg[n] for n in rotn])
        ax.text(0.99, 0.02, f"up to {upto}", transform=ax.transAxes, ha="right", fontsize=13, color=INK2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.02, 0.07, 0.96, 0.76])
    title(fig, "Who bids against whom: the construction network",
          f"Each dot is a firm; lines join firms bidding on the same works tender ({H.number_of_nodes()} firms)")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label="Firm in a group that rotates wins"),
                        plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="Other firm")],
               loc="upper right", bbox_to_anchor=(0.97, 0.86), frameon=False, fontsize=10)
    draw(ax, max(years))
    fig.savefig(OUT / "chart_network.png", dpi=200)
    if animate:
        seq = [y for y in years for _ in range(6)] + [max(years)] * 18
        save_writer(animation.FuncAnimation(fig, lambda i: draw(ax, seq[i]), frames=len(seq)), "anim_network", fps=6)
    plt.close(fig)
    return dict(firms=H.number_of_nodes(), links=H.number_of_edges(), rotating_in_view=len(rot_firms & set(H.nodes)))


# ------------------------------------------------------------------ 3. timeline
def chart_timeline(animate=True):
    r = json.loads((ROOT / "paper" / "estimate_timing.json").read_text())
    t = r["random_sample_timing"]
    proc, lag = t["cycle_median"], t["lag_median"]
    share = r["works_entered_after_award_share"]

    def draw(ax, p_len, l_len):
        ax.clear()
        ax.set_xlim(0, proc + lag + 10)
        ax.set_ylim(-1, 1.6)
        ax.set_axis_off()
        ax.barh(0.6, p_len, height=0.5, color=BLUE)
        ax.barh(0.6, l_len, left=proc, height=0.5, color=ORANGE)
        ax.text(0, 1.05, "Invitation", fontsize=10, color=INK2)
        if p_len >= proc:
            ax.text(proc / 2, 0.6, f"Procurement\n{proc:.0f} days", ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")
            ax.text(proc, 1.05, "Award made", fontsize=10, color=INK2, ha="center")
        if l_len >= lag:
            ax.text(proc + lag / 2, 0.6, f"Waiting to be published: {lag:.0f} days", ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold")
            ax.text(proc + lag, 1.05, "Award appears on the public portal", fontsize=10, color=INK2, ha="right")
            ax.text(0, -0.35, f"{share:.0%} of works tenders are entered on the portal only after the contract is awarded.",
                    fontsize=12, color=INK)

    fig = plt.figure(figsize=(8, 3.6))
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.62])
    title(fig, "The public finds out about three months late",
          "Median works tender, random sample of PPDA records · data forensics")
    draw(ax, proc, lag)
    fig.savefig(OUT / "chart_timeline.png", dpi=200)
    if animate:
        f1 = [(proc * i / 15, 0) for i in range(1, 16)]
        f2 = [(proc, lag * i / 30) for i in range(1, 31)]
        seq = f1 + [(proc, 0)] * 6 + f2 + [(proc, lag)] * 30
        save_writer(animation.FuncAnimation(fig, lambda i: draw(ax, *seq[i]), frames=len(seq)), "anim_timeline", fps=15)
    plt.close(fig)


# ------------------------------------------------------------------ 4. Benford
def chart_benford():
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    bw = pd.read_parquet(P / "works_bids_scored.parquet")
    v = pd.concat([w.award_value_ugx, bw.award_value_ugx]).dropna()
    v = v[v > 0]
    obs = v.map(lambda x: int(f"{x:e}"[0])).value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    exp = np.log10(1 + 1 / np.arange(1, 10))
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_axes([0.08, 0.19, 0.88, 0.61])
    title(fig, "Benford's Law: do contract values look natural?",
          "First digit of works award values vs the pattern natural numbers follow")
    ax.bar(range(1, 10), obs.values, width=0.7, color=BLUE, label="Uganda works awards")
    ax.plot(range(1, 10), exp, color=ORANGE, lw=2, marker="o", ms=6, label="Benford's Law")
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("First digit")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    mad = float(np.mean(np.abs(obs.values - exp)))
    ax.text(9.4, exp[0] * 0.95, f"Mean absolute deviation {mad:.3f}\nacceptable-to-marginal fit;\nmostly round-number pricing",
            ha="right", va="top", fontsize=10, color=INK2)
    fig.savefig(OUT / "chart_benford.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ 5. ML drivers
def chart_ml():
    r = json.loads((ROOT / "paper" / "ml_results.json").read_text())["isolation_forest"]
    lab = {"supplier_share_of_buyer": "Supplier dominates the entity's works", "supplier_age_days": "Long supplier–entity relationship",
           "awards_12m_same_pair": "Repeat awards to the same supplier", "is_direct": "Direct procurement",
           "award_to_est_c": "Price relative to estimate", "tender_days_c": "Length of bidding window",
           "log_value": "Contract value", "log_est": "Estimate size", "buyer_n_works": "Entity's number of works contracts",
           "is_restricted": "Restricted bidding"}
    d = pd.Series(r["anomaly_drivers_top_decile_z"]).rename(index=lab).sort_values()
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_axes([0.40, 0.18, 0.55, 0.56])
    title(fig, "What makes a contract unusual? (machine learning)",
          "Isolation Forest: how the 10% most unusual contracts differ from the average")
    ax.barh(d.index, d.values, color=BLUE, height=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(d.values):
        ax.text(max(v, 0) + 0.03, i, f"{v:+.2f}", va="center", ha="left",
                fontsize=9, color=INK2)
    ax.set_xlabel("Difference from average (standard deviations)")
    fig.text(0.04, 0.80, f"Agrees with the red-flag rules {r['top_decile_overlap'] / 0.10:.1f}× more than chance; "
             f"finds {r['ml_only_top_decile']} further unusual contracts.", fontsize=10.5, color=INK)
    fig.savefig(OUT / "chart_ml.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    chart_bunching()
    net = chart_network()
    chart_timeline()
    chart_benford()
    chart_ml()
    for f in ["anim_bunching.mp4", "anim_network.mp4", "anim_timeline.mp4", "chart_network.png",
              "chart_timeline.png", "chart_benford.png", "chart_ml.png"]:
        shutil.copy(OUT / f, SITE / f)
    print(json.dumps(net), sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
