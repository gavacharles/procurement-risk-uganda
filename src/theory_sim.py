"""(8) Bidding theory + policy simulation: does a KNOWN estimate produce the bunching, and
what would a SECRET estimate save?

Setting: first-price sealed-bid procurement; bids above the reserve r (= official estimate E,
normalised to 1) are rejected. Costs c_i iid ~ F (lognormal around 0.88 x E, calibrated so the
competitive part of the award/estimate distribution resembles Uganda's). Number of bidders
n ~ Uganda's observed mix from bid records.

Regime A (known reserve). Symmetric equilibrium bid for cost c <= r with n bidders:
    b(c) = c + [ integral_c^r (1-F(x))^(n-1) dx ] / (1-F(c))^(n-1)
For n = 1 this gives b = r (the monopolist bids the reserve). Bidders shade a hair below r
in practice; we add a small shading eps ~ U(0, 1%) for bids that equal r.

Regime B (secret reserve). Bidders know only that r ~ LogNormal(0, s) around their prior of E.
A bidder maximises (b - c) * P(win | b) * P(r >= b). For n = 1 that is (b - c) * G(b); for
n > 1 we solve the equilibrium numerically by best-response iteration on a grid. Bids above the
realised r are rejected -> the tender fails (re-tender cost/delay).

Outputs: share of awards in [0.99, 1.0); mean price; failure rate; savings of B vs A.
Out: paper/theory_sim.json, posts/media/chart_theory.png
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)
N_MIX = {1: 0.415, 2: 0.049, 3: 0.479, 4: 0.029, 5: 0.028}  # works bid records (normalised)
COST = stats.lognorm(s=0.10, scale=0.88)                  # cost / estimate
GRID = np.linspace(0.55, 1.30, 1501)


def bid_known(c: np.ndarray, n: int, r: float = 1.0) -> np.ndarray:
    """Equilibrium bid with known reserve r."""
    out = np.full_like(c, np.nan)
    ok = c <= r
    if n == 1:
        out[ok] = r
        return out
    xs = GRID[GRID <= r]
    surv = (1 - COST.cdf(xs)) ** (n - 1)
    # cumulative integral from x to r
    cum = np.concatenate([[0], np.cumsum((surv[1:] + surv[:-1]) / 2 * np.diff(xs))])
    tail = cum[-1] - cum
    num = np.interp(c[ok], xs, tail)
    den = np.interp(c[ok], xs, surv)
    out[ok] = c[ok] + num / np.maximum(den, 1e-12)
    return np.minimum(out, r)


def bid_secret(n: int, s: float, iters: int = 60) -> callable:
    """Best-response iteration for symmetric bidding when r ~ LogNormal(0, s) is unknown.
    Returns a function cost -> bid (monotone, on GRID)."""
    G = stats.lognorm(s=s, scale=1.0).sf  # P(r >= b)
    cgrid = np.linspace(0.55, 1.25, 141)
    bgrid = GRID
    beta = cgrid * 1.05  # initial guess
    for _ in range(iters):
        if n == 1:
            win = np.ones_like(bgrid)
        else:
            # P(rival bids above b) = P(beta(C) > b) = P(C > beta^-1(b))
            inv = np.interp(bgrid, beta, cgrid, left=cgrid[0], right=cgrid[-1])
            win = (1 - COST.cdf(inv)) ** (n - 1)
        prof = (bgrid[None, :] - cgrid[:, None]) * win[None, :] * G(bgrid)[None, :]
        new = bgrid[np.argmax(prof, axis=1)]
        new = np.maximum.accumulate(new)  # keep monotone
        if np.max(np.abs(new - beta)) < 1e-4:
            beta = new
            break
        beta = 0.5 * beta + 0.5 * new
    return lambda c: np.interp(c, cgrid, beta)


def simulate(regime: str, n_auctions=200_000, s=0.10, shade=0.01):
    ns = RNG.choice(list(N_MIX), size=n_auctions, p=np.array(list(N_MIX.values())) / sum(N_MIX.values()))
    prices, failed = [], 0
    funcs = {n: bid_secret(n, s) for n in N_MIX} if regime == "secret" else {}
    for n in N_MIX:
        k = int((ns == n).sum())
        c = COST.rvs(size=(k, n), random_state=RNG)
        if regime == "known":
            b = np.stack([bid_known(c[:, j], n) for j in range(n)], axis=1)
            atr = np.isclose(b, 1.0)
            b = np.where(atr, 1.0 - RNG.uniform(0, shade, b.shape), b)
            win = np.nanmin(b, axis=1)
            prices.append(win[~np.isnan(win)])
        else:
            r = stats.lognorm(s=s, scale=1.0).rvs(size=k, random_state=RNG)
            b = funcs[n](c)
            win = b.min(axis=1)
            ok = win <= r
            failed += int((~ok).sum())
            prices.append(win[ok] / r[ok])  # report relative to the *realised* estimate
    p = np.concatenate(prices)
    return dict(n=int(len(p)), share_99_100=float(((p >= 0.99) & (p < 1.0)).mean()),
                share_97_100=float(((p >= 0.97) & (p < 1.0)).mean()),
                mean_price=float(p.mean()), failure_rate=failed / n_auctions, sample=p[:20000])


def main():
    A = simulate("known")
    res = {"observed_uganda": {"share_99_100": 0.231, "share_97_100": 1976 / 3652},
           "known_estimate": {k: v for k, v in A.items() if k != "sample"}, "secret_estimate": {}}
    samples = {"known": A["sample"]}
    for s in [0.05, 0.10, 0.20]:
        B = simulate("secret", s=s)
        res["secret_estimate"][f"uncertainty_sd_{s}"] = dict(
            {k: v for k, v in B.items() if k != "sample"},
            saving_vs_known_pct=100 * (A["mean_price"] - B["mean_price"]) / A["mean_price"])
        samples[f"secret_{s}"] = B["sample"]
    (ROOT / "paper" / "theory_sim.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    chart(samples)


def chart(samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "src"))
    from visuals import BLUE, ORANGE, title
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_axes([0.08, 0.18, 0.88, 0.60])
    title(fig, "Theory: a known estimate makes prices pile up below it",
          "Simulated procurement auctions with Uganda's mix of bidder numbers",
          source="Simulation: first-price auction model (src/theory_sim.py)")
    bins = np.arange(0.70, 1.15, 0.005)
    ax.hist(samples["known"], bins=bins, color=BLUE, alpha=0.85, label="Estimate known to bidders")
    ax.hist(samples["secret_0.1"], bins=bins, histtype="step", lw=2, color=ORANGE, label="Estimate kept secret (±10% uncertainty)")
    ax.set_xlabel("Winning price as a share of the estimate")
    ax.set_ylabel("Simulated contracts")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    out = ROOT / "posts" / "media" / "chart_theory.png"
    fig.savefig(out, dpi=200)
    shutil.copy(out, ROOT / "docs" / "media" / out.name)


if __name__ == "__main__":
    main()
