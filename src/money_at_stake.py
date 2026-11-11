"""(1) Money at stake: what might the bunching cost, in shillings?

Logic (bunching counterfactual):
  * In the window [0.97, 1.00) of award/estimate we observe C contracts; a smooth
    counterfactual predicts C_hat. The excess share is f = (C - C_hat) / C.
  * If the excess reflects prices anchored up to the estimate, an excess contract
    would otherwise have been priced at r* x estimate. Its excess cost is
    (award - r* x estimate)+. Expected excess cost over the window:
        X(r*) = f * sum_{i in window} (A_i - r* E_i)+
  * Scenarios for r*:
      conservative  r* = 0.97                 (would have landed just below the window)
      central       r* = mean of the smooth counterfactual distribution
      competitive   r* = median ratio in open tenders outside the window
  * Bootstrap (resampling contracts) gives CIs for the central scenario.

Big caveat: if bunching comes from estimates being revised to match awards (not prices
rising to estimates) the excess cost is zero. Treat the result as a scenario bound on
what is at stake, not a measured loss.

Out: paper/money_at_stake.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import bunching  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
LO, HI, EXCL = 0.80, 1.20, (0.97, 1.00)


def excess_cost(d: pd.DataFrame, r_star: float, f: float) -> float:
    win = d[(d.ratio >= EXCL[0]) & (d.ratio < EXCL[1])]
    return float(f * np.clip(win.award - r_star * win.est, 0, None).sum())


def estimate(d: pd.DataFrame, n_boot=500, seed=0) -> dict:
    b = bunching(d.ratio, lo=LO, hi=HI, excl=EXCL, n_boot=50)
    f = b["excess_ratio"] / (1 + b["excess_ratio"])  # (C - C_hat)/C
    mids, pred = b["mids"], b["pred"].clip(min=0)
    r_cf = float((mids * pred).sum() / pred.sum())
    outside = d[(d.ratio < EXCL[0]) | (d.ratio >= EXCL[1])]
    r_open = float(outside[outside.method == "open"].ratio.median())
    scen = {"conservative_r0.97": 0.97, "central_counterfactual_mean": r_cf, "competitive_open_median": r_open}
    res = {k: dict(r_star=round(v, 4), excess_cost_ugx=excess_cost(d, v, f)) for k, v in scen.items()}

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        bb = bunching(s.ratio, lo=LO, hi=HI, excl=EXCL, n_boot=1)
        fb = bb["excess_ratio"] / (1 + bb["excess_ratio"])
        pb = bb["pred"].clip(min=0)
        rb = float((bb["mids"] * pb).sum() / pb.sum())
        boots.append(excess_cost(s, rb, fb))
    res["central_counterfactual_mean"]["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    win = d[(d.ratio >= EXCL[0]) & (d.ratio < EXCL[1])]
    return dict(n=int(len(d)), in_window=int(len(win)), excess_share_in_window=float(f),
                window_award_value_ugx=float(win.award.sum()), total_award_value_ugx=float(d.award.sum()),
                scenarios=res)


def main() -> None:
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    d = w.assign(award=w.award_value_ugx, est=w.estimate_ugx, ratio=w.award_to_est)
    d = d[(d.award > 0) & (d.est > 0) & d.ratio.between(0.3, 3)].copy()
    d["year"] = d.tender_start.dt.year.where(d.tender_start.dt.year >= 2014)
    out = {"all_works": estimate(d)}
    # robustness: drop the largest 1% of contracts (mega-projects dominate value)
    cap = d.award.quantile(0.99)
    out["excl_top1pct_value"] = estimate(d[d.award <= cap], n_boot=200)
    # by sub-sector (central scenario only)
    out["by_subsector"] = {}
    for sub, g in d.groupby("subsector"):
        if len(g) >= 200:
            e = estimate(g, n_boot=100)
            out["by_subsector"][sub] = dict(n=e["n"], central_ugx=e["scenarios"]["central_counterfactual_mean"]["excess_cost_ugx"],
                                            ci95=e["scenarios"]["central_counterfactual_mean"]["ci95"])
    c = out["all_works"]["scenarios"]["central_counterfactual_mean"]
    out["headline"] = dict(
        central_ugx_bn=round(c["excess_cost_ugx"] / 1e9, 1), ci95_ugx_bn=[round(x / 1e9, 1) for x in c["ci95"]],
        share_of_total_works_value=c["excess_cost_ugx"] / out["all_works"]["total_award_value_ugx"],
        years=f"{int(d.year.min())}-{int(d.year.max())}",
        caveat="Scenario assuming bunching reflects prices anchored to the estimate; zero if estimates were revised to awards.")
    (ROOT / "paper" / "money_at_stake.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
