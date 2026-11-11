"""Experiment E: do estimates/awards pile up just BELOW round procurement thresholds?

For candidate thresholds T (UGX 5m ... 1bn), compare counts in the 10% band below T (0.90T to <T,
excluding exact T) with the 10% band above (>T to 1.10T). Under a smooth (log-uniform-ish) density
the ratio below/above ~ 1.0; a ratio well above 1 at a threshold = values pushed under it
(contract sizing/splitting to avoid stricter methods). Exact round values are excluded so round-
number habits do not drive the result. Binomial test for H0: P(below)=0.5. Works (ML sample).
Out: paper/threshold_bunching.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
TS = [5e6, 10e6, 20e6, 30e6, 50e6, 100e6, 200e6, 300e6, 500e6, 1e9]


def run(v):
    out = {}
    for T in TS:
        below = int(((v >= 0.9 * T) & (v < T * 0.9995)).sum())
        above = int(((v > T * 1.0005) & (v <= 1.1 * T)).sum())
        n = below + above
        if n < 30:
            continue
        # adjust for the density slope: under log-uniform, band widths in log terms differ slightly
        exp_below = np.log(1 / 0.9) / (np.log(1 / 0.9) + np.log(1.1))
        p = binomtest(below, n, exp_below, alternative="greater").pvalue
        out[f"{T/1e6:g}m"] = dict(below=below, above=above, ratio=round(below / max(above, 1), 2), p=float(p))
    return out


def main():
    t = pd.read_parquet(P / "tenders.parquet").merge(pd.read_parquet(P / "works_ml_flags.parquet"), on="ocid")
    w = t[t.ml_works]
    allt = t
    res = {"works_estimate": run(w.estimate_ugx.dropna()), "works_award": run(w.award_value_ugx.dropna()),
           "all_estimate": run(allt.estimate_ugx.dropna())}
    (ROOT / "paper" / "threshold_bunching.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print("==", k)
        for T, r in v.items():
            star = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else ""
            print(f"  {T:>6s} below={r['below']:5d} above={r['above']:5d} ratio={r['ratio']:5.2f} {star}")


if __name__ == "__main__":
    main()
