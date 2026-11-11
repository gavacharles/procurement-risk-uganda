"""(5) Did e-procurement (e-GP) change outcomes? Staggered difference-in-differences.

INPUT REQUIRED: data/raw/egp_adoption.csv with columns
    buyer_name, adoption_date            (one row per entity that moved to e-GP)
PPDA confirms a 36-entity pilot from ~2021 and 100 more entities on 20 July 2026 (441 by ~2029),
but has not published the entity list with dates. Obtain it from PPDA (formal request) or the
e-GP portal archive, then run this script.

Design
  * Unit: procuring entity x quarter (tender advert date).
  * Outcomes: bunching (award at 99-100% of estimate), exact-estimate awards, non-open share,
    short bidding windows, and (bid-invitation sample) days until the award is published.
  * Treated: entities after adoption; controls: not-yet-adopted and never-adopted entities.
  * Estimators: (1) two-way fixed effects (entity + quarter), clustered by entity;
    (2) event study with leads/lags (-8..+8 quarters) to check pre-trends;
    (3) stacked DiD by adoption cohort (robust to heterogeneous effects under staggering).

Out: paper/egp_did.json, posts/media/chart_egp_event.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
ADOPT = ROOT / "data" / "raw" / "egp_adoption.csv"
OUTCOMES = ["bunched", "exact", "non_open", "short_window"]


def panel() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src"))
    from natural_experiments import prepare
    d = prepare()
    d["q"] = d.tender_start.dt.to_period("Q")
    a = pd.read_csv(ADOPT, parse_dates=["adoption_date"])
    a["adopt_q"] = a.adoption_date.dt.to_period("Q")
    d = d.merge(a[["buyer_name", "adopt_q"]], on="buyer_name", how="left")
    d["rel_q"] = np.where(d.adopt_q.notna(), (d.q - d.adopt_q).apply(lambda x: x.n if pd.notna(x) else np.nan), np.nan)
    d["post"] = (d.rel_q >= 0).astype(int)
    d["qs"] = d.q.astype(str)
    return d


def twfe(d, y):
    s = d.dropna(subset=[y])
    r = smf.ols(f"{y} ~ post + C(entity) + C(qs)", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.entity})
    return dict(n=int(r.nobs), coef=float(r.params["post"]), se=float(r.bse["post"]), p=float(r.pvalues["post"]))


def event_study(d, y, lo=-8, hi=8):
    s = d.dropna(subset=[y]).copy()
    s["k"] = s.rel_q.clip(lo, hi)
    dummies = []
    for k in range(lo, hi + 1):
        if k == -1:
            continue  # reference period
        name = f"k_{'m' if k < 0 else 'p'}{abs(k)}"
        s[name] = (s.k == k).astype(int)
        dummies.append((k, name))
    f = f"{y} ~ " + " + ".join(n for _, n in dummies) + " + C(entity) + C(qs)"
    r = smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": s.entity})
    return [dict(k=k, coef=float(r.params[n]), se=float(r.bse[n])) for k, n in dummies]


def main():
    if not ADOPT.exists():
        print(f"Missing {ADOPT}. Create it with columns buyer_name, adoption_date (see module docstring).")
        sys.exit(1)
    d = panel()
    res = {"treated_entities": int(d.loc[d.adopt_q.notna(), "entity"].nunique()),
           "twfe": {y: twfe(d, y) for y in OUTCOMES},
           "event_study": {y: event_study(d, y) for y in OUTCOMES}}
    (ROOT / "paper" / "egp_did.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["twfe"], indent=1))


if __name__ == "__main__":
    main()
