"""Paper 3, extension (4): contract value per kilometre contracted and per kilometre the satellite sees sealed.

delivered_km = contract_km x (sealed share at the latest image - sealed share at tender)
Caveats (reported with the numbers): award values can cover link roads, town roads or a second road; parts of a
route may have been sealed before the tender (town sections), which lowers the gain; route errors move both.
Values are nominal UGX; a contract is included only if it has an award value and gained at least 30 points of
sealed share after the tender.

Out: paper/road_unit_costs.json, posts/media/chart_road_unit_costs.png (+ paper/figures/p3_f12_unit_costs.png)
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCOPE = {"muyembe": "includes 25 km of link roads", "kigumba": "value may cover one of two lots",
         "hoima": "part of an oil-roads package; town sections sealed before tender",
         "rwenkunye": "two lots; not complete", "luwero": "not complete (67% sealed)"}


def main():
    R = pd.read_parquet(ROOT / "data" / "processed" / "road_progress_roads.parquet")
    R = R[R.kind.eq("upgrade") & R.award_value_bn.fillna(0).gt(0)].copy()
    R["gain"] = R.s_last - R.s_tender
    R = R[R.gain >= 0.30].copy()
    R["bn_per_contract_km"] = R.award_value_bn / R.contract_km
    R["delivered_km"] = R.contract_km * R.gain
    R["bn_per_delivered_km"] = R.award_value_bn / R.delivered_km
    R["note"] = R.id.map(SCOPE).fillna("")
    cols = ["road", "award_value_bn", "contract_km", "gain", "delivered_km", "bn_per_contract_km", "bn_per_delivered_km", "note"]
    out = R[cols].round(3).to_dict("records")
    (ROOT / "paper" / "road_unit_costs.json").write_text(json.dumps(out, indent=2))
    print(R[cols].round(2).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visuals import BLUE, INK2, title
    D = R.sort_values("bn_per_contract_km")
    H = 4.6
    fig = plt.figure(figsize=(9, H))
    title(fig, "What a kilometre of paving cost: contracted vs seen sealed",
          "UGX billion per km: award value / contracted length (light) and / length the satellite sees sealed since the tender (dark).\n"
          "Award values can include link and town roads; treat differences as prompts to check scope, not as overpayment.",
          source="PPDA open contracting data · Copernicus Sentinel-2")
    for t in fig.texts:
        x, y = t.get_position()
        t.set_y(1 - (0.3 if y > 0.9 else 0.68) / H) if y > 0.5 else t.set_y(0.2 / H)
    ax = fig.add_axes([0.3, 0.75 / H, 0.65, 1 - 1.95 / H])
    y = range(len(D))
    ax.barh([i + 0.18 for i in y], D.bn_per_contract_km, 0.34, color="#9ec5f4", label="per contracted km")
    ax.barh([i - 0.18 for i in y], D.bn_per_delivered_km, 0.34, color=BLUE, label="per km seen sealed")
    ax.set_yticks(list(y), [f"{r.road}\n({r.note})" if r.note else r.road for r in D.itertuples()], fontsize=7.5)
    ax.set_xlabel("UGX billion per km", fontsize=8.5)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.savefig(ROOT / "posts" / "media" / "chart_road_unit_costs.png", dpi=200)
    shutil.copy(ROOT / "posts" / "media" / "chart_road_unit_costs.png", ROOT / "paper" / "figures" / "p3_f12_unit_costs.png")
    print("wrote chart")


if __name__ == "__main__":
    main()
