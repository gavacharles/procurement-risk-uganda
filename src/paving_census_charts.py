"""Chart for paper 3, extension (3): the national paving census.

Out: posts/media/chart_paving_census.png (+ docs/media, paper/figures/p3_f15_census.png)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from visuals import BLUE, ORANGE, INK, INK2, title  # noqa: E402
from road_progress_charts import save  # noqa: E402
import paving_census as pc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


def main():
    res = json.loads((ROOT / "paper" / "paving_census.json").read_text())
    pts = pd.read_parquet(P / "paving_census_points.parquet")
    new = pd.read_parquet(P / "paving_census_newly_paved.parquet")
    st = pd.read_parquet(P / "paving_census_stretches.parquet")
    matched_ids = set(st.index[st.matched_any])
    new = new[new.stretch >= 0]
    ways = [w for w in pc.osm_major() if w["id"] in set(pts.way)]
    geo = json.loads((ROOT / "data" / "raw" / "geo" / "uga_districts_ubos2020.geojson").read_text())

    H = 8.4
    fig = plt.figure(figsize=(12, H))
    wp = res["with_places"]
    title(fig, "Every major road in Uganda, 2017 to 2026: where new tarmac appeared",
          f"{res['stretch_km']:,.0f} km of new paving found from space; {wp['matched_share_km']:.0%} of it traces to a contract in the open record.",
          source="OpenStreetMap · Copernicus Sentinel-2 · PPDA open contracting data")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)

    ax = fig.add_axes([0.02, 0.5 / H, 0.56, 1 - 1.5 / H])
    for f in geo["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            xy = np.array(poly[0])
            ax.fill(xy[:, 0], xy[:, 1], color="#f3f2ee", ec="#e0ded9", lw=0.3, zorder=1)
    for w in ways:
        xy = np.array(w["geom"])
        ax.plot(xy[:, 0], xy[:, 1], color="#b9bcc2", lw=0.5, zorder=2)
    m = new.stretch.isin(matched_ids)
    ax.scatter(new.lon[m], new.lat[m], s=16, color=BLUE, zorder=4, lw=0)
    ax.scatter(new.lon[~m], new.lat[~m], s=16, color=ORANGE, zorder=4, lw=0)
    ax.set_aspect(1 / np.cos(np.radians(1.3)))
    ax.set_xlim(29.5, 35.1)
    ax.set_ylim(-1.5, 4.3)
    ax.axis("off")
    hs = [Line2D([0], [0], color="#b9bcc2", lw=1.5),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=7),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=7)]
    ax.legend(hs, ["major road network (OSM)",
                   f"newly paved, traced to a contract ({wp['matched_km']:,.0f} km)",
                   f"newly paved, no contract in the open record ({wp['unmatched_km']:,.0f} km)"],
              loc="upper left", bbox_to_anchor=(0.655, 0.17), bbox_transform=fig.transFigure, frameon=False, fontsize=8.5)

    # noise check
    bx = fig.add_axes([0.66, 0.42 + 0.0, 0.31, 0.38])
    ks = [1, 2, 3, 4]
    S = res["stretches"]
    nk = [S[f"min_pts_{k}"]["new_km"] for k in ks]
    rk = [S[f"min_pts_{k}"]["reverted_km"] for k in ks]
    x = np.arange(len(ks))
    bx.bar(x - 0.19, nk, 0.36, color=BLUE, label="unpaved → asphalt (new paving)")
    bx.bar(x + 0.19, rk, 0.36, color="#c9ccd1", label="asphalt → unpaved (noise)")
    for i in range(len(ks)):
        bx.text(x[i] - 0.19, nk[i] + 15, f"{nk[i]:,.0f}", ha="center", fontsize=7.5, color=INK2)
        bx.text(x[i] + 0.19, rk[i] + 15, f"{rk[i]:,.0f}", ha="center", fontsize=7.5, color=INK2)
    bx.set_xticks(x, ["any point", "2+ points", "3+ points", "4+ points"], fontsize=8.5)
    bx.set_ylabel("km", fontsize=9, color=INK2)
    for s_ in ("top", "right"):
        bx.spines[s_].set_visible(False)
    bx.tick_params(labelsize=8, colors=INK2)
    bx.legend(frameon=False, fontsize=8, loc="upper right")
    bx.set_title("Noise check: requiring a stretch of linked points", fontsize=10.5, color=INK, loc="left", pad=8)
    fig.text(0.66, 0.36, "Real roads almost never lose their tarmac, so apparent\n"
                         "asphalt → unpaved changes estimate classifier noise.\n"
                         f"With 2+ linked points, noise is {S['min_pts_2']['noise_ratio']:.0%} of the new-paving signal;\n"
                         f"with 3+ points, {S['min_pts_3']['noise_ratio']:.0%}.",
             fontsize=8.5, color=INK2, va="top")
    fig.text(0.66, 0.255, f"Readable at both ends of the archive: {res['km_valid']:,.0f} of {res['km_network']:,.0f} km.\n"
                         f"Paved share of the readable network: {res['start_asphalt_share_km']:.0%} → {res['end_asphalt_share_km']:.0%}.\n"
                         f"Surface classifier accuracy (true colour): {res['rgb_classifier']['cv_accuracy']:.0%}.",
             fontsize=8.5, color=INK2, va="top")
    save(fig, "chart_paving_census.png", "p3_f15_census.png")


if __name__ == "__main__":
    main()
