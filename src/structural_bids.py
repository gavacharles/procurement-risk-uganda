"""Paper 4: what does publishing the estimate cost? Structural estimation of procurement auctions (Uganda, PPDA).

Model. Lowest-price sealed-bid auctions with n bidders, symmetric independent private costs c ~ F. The
published estimate r (the engineer's estimate / budget) is used to normalise bids, beta = bid / r. Bids above
the estimate are allowed and common, so in the status quo the estimate is treated as non-binding.
A bidder may be disqualified at evaluation (the lowest bid wins in 73% of auctions); disqualification is
assumed independent of the bid, so it scales the win probability by a constant and leaves the first-order
condition unchanged.

Identification (Guerre, Perrigne & Vuong 2000, procurement form). With G_n, g_n the distribution and density
of bids in n-bidder auctions, each bid maps to a pseudo-cost
        c = beta - (1 - G_n(beta)) / ((n - 1) g_n(beta)).
Densities are kernel estimates on log bids; pseudo-values within one bandwidth of the sample boundary are
trimmed (boundary bias).

Validity conditions (reported, not assumed):
  V1 monotonicity: pseudo-costs increase with bids (necessary for equilibrium bidding);
  V2 exogenous participation: cost distributions estimated from different n coincide;
  V3 fit: the model's expected winning bid by n matches the observed lowest bid;
  V4 format: the lowest bid wins in most auctions;
  V5 robustness: works only, open bidding only, excluding bids at the estimate, other bandwidths.

Counterfactuals (pooled cost distribution, observed mix of n): status quo (non-binding public estimate);
binding public ceiling at the estimate; secret ceiling r = exp(N(0, s)); published range r ~ U(0.9, 1.1);
one more bidder. Equilibria with a ceiling are solved numerically by damped best-response iteration.
Bootstrap over auctions for confidence intervals.

Out: data/processed/structural_bids_pseudocosts.parquet, paper/structural_bids.json, figures p4_*.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))

NMAX_GROUP = 6           # n >= 6 pooled for density estimation
NMAX_CF = 8              # counterfactual n capped here
B_BOOT = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 100


# ---------------------------------------------------------------- data
def load():
    b = pd.read_parquet(P / "bids.parquet")
    t = pd.read_parquet(P / "tenders.parquet")
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    est = t[["ocid", "estimate_ugx", "award_value_ugx", "method", "tender_start", "title"]].dropna(subset=["estimate_ugx"]).drop_duplicates("ocid")
    x = b.merge(est, on="ocid")
    x = x[(x.estimate_ugx > 0) & (x.bid_value_ugx > 0)].copy()
    x["beta"] = x.bid_value_ugx / x.estimate_ugx
    raw_auctions = x.ocid.nunique()
    # unit errors: drop auctions with any bid below a quarter or above four times the estimate
    bad = x.groupby("ocid").beta.agg(lambda s: ((s < 0.25) | (s > 4)).any())
    x = x[~x.ocid.isin(bad[bad].index)]
    x = x[x.method.isin(["open", "limited", "selective"])]
    g = x.groupby("ocid").beta.agg(["size", "nunique"])
    x = x[x.ocid.isin(g[g["nunique"] >= 2].index)].copy()
    x["n"] = x.groupby("ocid").beta.transform("size").astype(int)
    x["ng"] = x.n.clip(upper=NMAX_GROUP)
    x["works"] = x.ocid.isin(w.ocid)
    x["year"] = pd.to_datetime(x.tender_start, errors="coerce", utc=True).dt.year
    return x.reset_index(drop=True), dict(raw_auctions=int(raw_auctions))


# ---------------------------------------------------------------- GPV inversion
def kde_log(y, bw_mult=1.0):
    h = 1.06 * np.std(y) * len(y) ** (-1 / 5) * bw_mult

    def pdf(v):
        return norm.pdf((v[:, None] - y[None, :]) / h).sum(1) / (len(y) * h)

    def cdf(v):
        return norm.cdf((v[:, None] - y[None, :]) / h).mean(1)
    return pdf, cdf, h


def pseudo_costs(x: pd.DataFrame, bw_mult=1.0):
    out = []
    for ng, d in x.groupby("ng"):
        y = np.log(d.beta.to_numpy())
        pdf, cdf, h = kde_log(y, bw_mult)
        G, gy = cdf(y), pdf(y)
        beta = np.exp(y)
        g = gy / beta
        # pooled top group: one common n (its mean), so the inversion stays monotone within the group
        nn = d.n.to_numpy() if ng < NMAX_GROUP else np.full(len(d), d.n.mean())
        c = beta - (1 - G) / ((nn - 1) * g)
        trim = (y < y.min() + h) | (y > y.max() - h)
        out.append(d.assign(G=G, c=c, trimmed=trim, h=h))
    r = pd.concat(out)
    r["markup"] = (r.beta - r.c) / r.beta
    return r


def monotone_share(r):
    res = {}
    for ng, d in r[~r.trimmed].groupby("ng"):
        d = d.sort_values("beta")
        res[int(ng)] = float((np.diff(d.c.to_numpy()) >= -1e-9).mean()) if len(d) > 2 else None
    return res


# ---------------------------------------------------------------- cost distribution
class CostDist:
    """Smoothed cost distribution from positive, untrimmed pseudo-costs (KDE on log cost)."""

    def __init__(self, c):
        c = np.asarray(c)
        c = c[np.isfinite(c) & (c > 0)]
        self.y = np.log(c)
        self.pdf_y, self.cdf_y, _ = kde_log(self.y)
        lo, hi = np.percentile(self.y, 0.2) - 0.3, np.percentile(self.y, 99.8) + 0.3
        self.grid_y = np.linspace(lo, hi, 2000)
        cdf = self.cdf_y(self.grid_y)
        self.grid_c, self.grid_F = np.exp(self.grid_y), np.maximum.accumulate(cdf)

    def F(self, c):
        return np.interp(np.log(np.maximum(c, 1e-9)), self.grid_y, self.grid_F, left=0.0, right=1.0)

    def sample(self, size, rng):
        u = rng.uniform(self.grid_F[0], self.grid_F[-1], size)
        return np.interp(u, self.grid_F, self.grid_c)

    def quantiles(self, q):
        return np.interp(q, self.grid_F, self.grid_c)


# ---------------------------------------------------------------- equilibrium with a (possibly secret) ceiling
def solve_bids(Fd: CostDist, n, survive, K=220, X=700, iters=80):
    """Symmetric equilibrium bid function b(c) on a cost grid, for win-acceptance probability survive(b)=Pr(r>=b)."""
    cq = Fd.quantiles(np.linspace(0.002, 0.995, K))
    Fc = Fd.F(cq)
    xs = np.linspace(cq[0], cq[-1] * 1.6 + 0.5, X)
    b = cq * 1.15 + 0.01
    S = survive(xs)
    for _ in range(iters):
        bm = np.maximum.accumulate(b)
        # probability that one rival bids above x: 1 - F(c(x)), with c(x) the inverse of b(.)
        cx = np.interp(xs, bm, cq, left=cq[0], right=np.inf)
        above = np.where(np.isinf(cx), 0.0, 1 - Fd.F(np.where(np.isinf(cx), cq[-1], cx)))
        above = np.where(xs < bm[0], 1.0, above)
        win = above ** (n - 1) * S
        prof = (xs[None, :] - cq[:, None]) * win[None, :]
        prof[xs[None, :] < cq[:, None]] = -1
        bn = xs[np.argmax(prof, axis=1)]
        b = 0.5 * b + 0.5 * bn
    return cq, np.maximum.accumulate(b)


def simulate(Fd: CostDist, n, policy, rng, sims=20000, s=0.1):
    """Expected price (relative to the estimate, conditional on award) and failure rate."""
    c = Fd.sample((sims, n), rng)
    if policy == "status_quo":           # non-binding public estimate: price = second-lowest cost
        cs = np.sort(c, axis=1)
        return float(cs[:, 1].mean()), 0.0
    if policy == "more_bidders":
        c2 = Fd.sample((sims, n + 1), rng)
        return float(np.sort(c2, axis=1)[:, 1].mean()), 0.0
    if policy == "binding_public":
        cs = np.sort(c, axis=1)
        ok = cs[:, 0] <= 1.0
        price = np.minimum(cs[:, 1], 1.0)
        return float(price[ok].mean()), float(1 - ok.mean())
    if policy == "secret":
        survive = lambda v: 1 - norm.cdf(np.log(np.maximum(v, 1e-9)) / s)
        r = np.exp(rng.normal(0, s, sims))
    elif policy == "range":
        survive = lambda v: np.clip((1.1 - v) / 0.2, 0, 1)
        r = rng.uniform(0.9, 1.1, sims)
    else:
        raise ValueError(policy)
    cq, bq = solve_bids(Fd, n, survive)
    bids = np.interp(c, cq, bq)
    bmin = bids.min(axis=1)
    ok = bmin <= r
    return float(bmin[ok].mean()) if ok.any() else float("nan"), float(1 - ok.mean())


POLICIES = [("status_quo", {}), ("binding_public", {}), ("secret", dict(s=0.05)), ("secret", dict(s=0.10)),
            ("secret", dict(s=0.20)), ("range", {}), ("more_bidders", {})]


def policy_name(p, kw):
    return p + (f"_s{kw['s']:.2f}" if "s" in kw else "")


def counterfactuals(Fd, n_weights, seed=0, sims=20000):
    rng = np.random.default_rng(seed)
    out = {}
    for p, kw in POLICIES:
        pr, fl = 0.0, 0.0
        for n, wt in n_weights.items():
            price, fail = simulate(Fd, int(n), p, rng, sims=sims, **kw)
            pr += wt * price
            fl += wt * fail
        out[policy_name(p, kw)] = dict(price=pr, fail=fl)
    base = out["status_quo"]["price"]
    for k in out:
        out[k]["price_change_vs_status_quo"] = out[k]["price"] / base - 1
    return out


# ---------------------------------------------------------------- validity checks
def fit_check(x, Fd, rng):
    rows = []
    for n, d in x.groupby(x.n.clip(upper=NMAX_CF)):
        obs = d.groupby("ocid").beta.min()
        model, _ = simulate(Fd, int(n), "status_quo", rng)
        rows.append(dict(n=int(n), auctions=int(len(obs)), observed_lowest_bid=float(obs.mean()), model_expected_price=model))
    return rows


def invariance(r):
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    ok = r[~r.trimmed & (r.c > 0)]
    tab = {int(n): [float(v) for v in np.quantile(d.c, q)] for n, d in ok.groupby("ng") if len(d) > 30}
    from scipy.stats import ks_2samp
    ns = sorted(tab)
    ks = {f"{a}v{b}": float(ks_2samp(ok[ok.ng == a].c, ok[ok.ng == b].c).statistic) for i, a in enumerate(ns) for b in ns[i + 1:]}
    return dict(quantiles=q, by_n=tab, ks=ks)


def markups_by_n(r):
    ok = r[~r.trimmed]
    return {int(n): dict(bids=int(len(d)), median_markup=float(d.markup.median()), mean_markup=float(d.markup.clip(-1, 1).mean()),
                         neg_cost_share=float((d.c <= 0).mean()), median_bid=float(d.beta.median()))
            for n, d in ok.groupby("ng")}


def lowest_wins(x):
    a = x.groupby("ocid").agg(minb=("bid_value_ugx", "min"), award=("award_value_ugx", "first"))
    a = a[a.award.notna() & (a.award > 0)]
    r = a.award / a.minb
    return dict(auctions=int(len(a)), lowest_wins=float(((r - 1).abs() <= 0.01).mean()),
                higher_bid_wins=float((r > 1.01).mean()), below_lowest=float((r < 0.99).mean()))


def anchoring(x):
    from analysis import bunching
    bb = bunching(x.beta, lo=0.80, hi=1.20, width=0.005, excl=(0.97, 1.0), deg=5, n_boot=300)
    near = ((x.beta >= 0.99) & (x.beta < 1.0)).mean()
    exact = (np.abs(x.beta - 1) < 0.001).mean()
    by_n = {int(n): dict(bids=int(len(d)), share_99_100=float(((d.beta >= 0.99) & (d.beta <= 1.0)).mean()),
                         share_above=float((d.beta > 1.0).mean())) for n, d in x.groupby("ng")}
    return dict(bunching_excess_ratio=bb["excess_ratio"], ci95=bb["ci95"], observed_in_window=bb["observed_in_window"],
                counterfactual_in_window=bb["counterfactual_in_window"], share_99_100=float(near), share_exact=float(exact),
                share_above_estimate=float((x.beta > 1.0).mean()), by_n=by_n,
                hist=dict(mids=bb["mids"].tolist(), cnt=bb["cnt"].tolist(), pred=bb["pred"].tolist()))


# ---------------------------------------------------------------- main
def main():
    x, meta = load()
    r = pseudo_costs(x)
    ok = r[~r.trimmed & (r.c > 0)]
    Fd = CostDist(ok.c)
    nw = x.drop_duplicates("ocid").n.clip(upper=NMAX_CF).value_counts(normalize=True).sort_index().to_dict()
    rng = np.random.default_rng(1)
    res = dict(
        sample=dict(**meta, auctions=int(x.ocid.nunique()), bids=int(len(x)), works_share=float(x.drop_duplicates("ocid").works.mean()),
                    years=x.drop_duplicates("ocid").year.value_counts().sort_index().to_dict(),
                    methods=x.drop_duplicates("ocid").method.value_counts().to_dict(),
                    n_auctions=x.drop_duplicates("ocid").n.value_counts().sort_index().to_dict(),
                    trimmed_share=float(r.trimmed.mean()), negative_cost_share_untrimmed=float((r[~r.trimmed].c <= 0).mean())),
        markups=markups_by_n(r),
        V1_monotone=monotone_share(r),
        V2_invariance=invariance(r),
        V3_fit=fit_check(x, Fd, rng),
        V4_format=lowest_wins(x),
        anchoring=anchoring(x),
        n_weights={int(k): float(v) for k, v in nw.items()},
    )
    # V5 robustness of markups
    rob = {}
    for name, sub, bw in [("works_only", x[x.works], 1.0), ("open_only", x[x.method == "open"], 1.0),
                          ("excl_bids_at_estimate", x[~((x.beta >= 0.99) & (x.beta <= 1.0))], 1.0),
                          ("bandwidth_x0.75", x, 0.75), ("bandwidth_x1.5", x, 1.5),
                          # unobserved heterogeneity: scale bids by the auction's geometric-mean bid instead of the estimate
                          ("heterogeneity_adjusted", x.assign(beta=x.beta / np.exp(x.groupby("ocid").beta.transform(lambda v: np.log(v).mean()))), 1.0)]:
        sub = sub[sub.ocid.isin(sub.groupby("ocid").beta.nunique().loc[lambda s: s >= 2].index)].copy()
        sub["n"] = sub.groupby("ocid").beta.transform("size").astype(int)
        sub["ng"] = sub.n.clip(upper=NMAX_GROUP)
        if sub.ocid.nunique() < 20:
            continue
        rr = pseudo_costs(sub, bw)
        rob[name] = dict(auctions=int(sub.ocid.nunique()), markups={k: v["median_markup"] for k, v in markups_by_n(rr).items()})
    res["V5_robustness"] = rob
    # counterfactuals with bootstrap
    cf = counterfactuals(Fd, nw, seed=2)
    boots = []
    auctions = x.ocid.unique()
    brng = np.random.default_rng(3)
    for bi in range(B_BOOT):
        pick = brng.choice(auctions, len(auctions), replace=True)
        xb = pd.concat([x[x.ocid == a].assign(ocid=f"{a}#{i}") for i, a in enumerate(pick)])
        rb = pseudo_costs(xb)
        okb = rb[~rb.trimmed & (rb.c > 0)]
        nwb = xb.drop_duplicates("ocid").n.clip(upper=NMAX_CF).value_counts(normalize=True).to_dict()
        boots.append(counterfactuals(CostDist(okb.c), nwb, seed=10 + bi, sims=8000))
        if (bi + 1) % 10 == 0:
            print(f"bootstrap {bi + 1}/{B_BOOT}", flush=True)
    for k in cf:
        for m in ("price_change_vs_status_quo", "fail"):
            v = [b_[k][m] for b_ in boots if np.isfinite(b_[k][m])]
            if v:
                cf[k][m + "_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res["counterfactuals"] = cf
    res["bootstrap_reps"] = B_BOOT
    # secret-ceiling trade-off curve (point estimates)
    curve = []
    for s in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        rng2 = np.random.default_rng(int(s * 1000))
        pr = fl = 0.0
        for n, wt in nw.items():
            p_, f_ = simulate(Fd, int(n), "secret", rng2, s=s)
            pr += wt * p_
            fl += wt * f_
        curve.append(dict(s=s, price_change=pr / cf["status_quo"]["price"] - 1, fail=fl))
    res["secret_curve"] = curve
    r.drop(columns=["title"], errors="ignore").to_parquet(P / "structural_bids_pseudocosts.parquet", index=False)
    (ROOT / "paper" / "structural_bids.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("sample", "markups", "V1_monotone", "V4_format")}, indent=1, default=float))
    print(json.dumps({k: {m: round(v[m], 4) for m in ("price_change_vs_status_quo", "fail")} for k, v in cf.items()}, indent=1))
    charts(x, r, res, Fd)


# ---------------------------------------------------------------- charts
def charts(x, r, res, Fd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visuals import BLUE, ORANGE, INK, INK2, title
    OUT = ROOT / "posts" / "media"
    FIG = ROOT / "paper" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    def save(fig, name, paper):
        fig.savefig(OUT / name, dpi=200)
        shutil.copy(OUT / name, FIG / paper)
        plt.close(fig)
        print("wrote", name)

    def frame(fig, H):
        for t in fig.texts:
            xx, yy = t.get_position()
            t.set_y(1 - (0.3 if yy > 0.9 else 0.68) / H) if yy > 0.5 else t.set_y(0.2 / H)

    a = res["anchoring"]
    # F1: bids relative to the estimate
    H = 4.8
    fig = plt.figure(figsize=(9, H))
    title(fig, "Bids pile up at the published estimate",
          f"All {res['sample']['bids']:,} bids in {res['sample']['auctions']} auctions with two or more bids (2018–2019). Bars: bids per 0.5% bin;\n"
          f"line: smooth counterfactual fitted outside 0.97–1.00. {a['share_above_estimate']:.0%} of bids are above the estimate.",
          source="PPDA open contracting data (bids)")
    frame(fig, H)
    ax = fig.add_axes([0.08, 0.85 / H, 0.88, 1 - 2.1 / H])
    mids, cnt, pred = np.array(a["hist"]["mids"]), np.array(a["hist"]["cnt"]), np.array(a["hist"]["pred"])
    ax.bar(mids, cnt, width=0.0045, color=[ORANGE if 0.97 <= m < 1.0 else BLUE for m in mids])
    ax.plot(mids, pred, color=INK, lw=1.5)
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.text(1.003, max(cnt) * 0.92, f"excess mass just below the estimate: {a['bunching_excess_ratio']:.1f}×\n(95% CI {a['ci95'][0]:.1f}–{a['ci95'][1]:.1f})", fontsize=8, color=INK2)
    ax.set_xlabel("Bid / published estimate", fontsize=9)
    ax.set_ylabel("Bids", fontsize=9)
    ax.tick_params(labelsize=8)
    save(fig, "chart_p4_bids_vs_estimate.png", "p4_f1_bids_vs_estimate.png")

    # F2: markups by number of bidders (JSON keys come back as strings)
    m = {int(k): v for k, v in res["markups"].items()}
    for rb_ in res["V5_robustness"].values():
        rb_["markups"] = {int(k): v for k, v in rb_["markups"].items()}
    H = 4.8
    fig = plt.figure(figsize=(9, H))
    title(fig, "More bidders, thinner margins",
          "Median markup, (bid − cost) / bid, by number of bidders; costs recovered from the bids (Guerre, Perrigne &\n"
          "Vuong 2000). Grey points: other samples and bandwidths. The level is uncertain; the fall with competition is not.",
          source="PPDA open contracting data (bids); author's estimates")
    frame(fig, H)
    ax = fig.add_axes([0.08, 0.85 / H, 0.88, 1 - 2.1 / H])
    ns = sorted(m)
    ax.plot(ns, [m[k]["median_markup"] for k in ns], "o-", color=BLUE, lw=2.5, ms=8, label="main estimate")
    for name, sty in [("works_only", "s"), ("open_only", "^"), ("excl_bids_at_estimate", "D"), ("bandwidth_x0.75", "v"), ("bandwidth_x1.5", "<"),
                      ("heterogeneity_adjusted", "P")]:
        rb = res["V5_robustness"].get(name)
        if rb:
            kk = sorted(int(k) for k in rb["markups"])
            ax.plot(kk, [rb["markups"][k] for k in kk], sty, color="#9aa0a6", ms=5, label=name.replace("_", " "))
    ax.set_xticks(ns, [f"{k}" if k < NMAX_GROUP else f"{k}+" for k in ns])
    ax.set_xlabel("Number of bidders", fontsize=9)
    ax.set_ylabel("Median markup", fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.tick_params(labelsize=8)
    save(fig, "chart_p4_markups.png", "p4_f2_markups.png")

    # F3: cost distributions by n (V2 invariance)
    H = 4.4
    fig = plt.figure(figsize=(9, H))
    title(fig, "Same costs, whatever the competition? (validity check)",
          "Distribution of recovered costs (relative to the estimate) for auctions with different numbers of bidders.\n"
          "If participation is unrelated to costs, the curves should lie on top of each other.",
          source="Author's estimates")
    frame(fig, H)
    ax = fig.add_axes([0.08, 0.85 / H, 0.88, 1 - 2.1 / H])
    okc = r[~r.trimmed & (r.c > 0)]
    cols = ["#9fc3ee", "#6ea5e3", BLUE, "#1a56a0", "#0f3768"]
    for i, (ng, d) in enumerate(okc.groupby("ng")):
        v = np.sort(d.c.to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=cols[i % len(cols)], lw=2, label=f"n = {int(ng)}{'+' if ng == NMAX_GROUP else ''}")
    ax.set_xlim(0, 2)
    ax.set_xlabel("Recovered cost / estimate", fontsize=9)
    ax.set_ylabel("Share of bidders", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    save(fig, "chart_p4_cost_invariance.png", "p4_f3_cost_invariance.png")

    # F4: fit
    fit = pd.DataFrame(res["V3_fit"])
    H = 4.8
    fig = plt.figure(figsize=(9, H))
    title(fig, "Does the model reproduce the winning bids? (validity check)",
          "Average lowest bid observed vs the model's expected winning bid, by number of bidders\n"
          "(bids relative to the published estimate; number of auctions in brackets).",
          source="Author's estimates")
    frame(fig, H)
    ax = fig.add_axes([0.08, 1.0 / H, 0.88, 1 - 2.2 / H])
    ax.bar(fit.n - 0.18, fit.observed_lowest_bid, 0.34, color="#b9bcc2", label="observed")
    ax.bar(fit.n + 0.18, fit.model_expected_price, 0.34, color=BLUE, label="model")
    ax.set_xticks(fit.n, [f"{k}\n({a_})" for k, a_ in zip(fit.n, fit.auctions)])
    ax.set_xlabel("Number of bidders (auctions)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    save(fig, "chart_p4_fit.png", "p4_f4_fit.png")

    # F5: counterfactuals
    cf = res["counterfactuals"]
    order = [("binding_public", "Estimate as a\nbinding ceiling"), ("secret_s0.05", "Secret ceiling\n(±5% uncertainty)"),
             ("secret_s0.10", "Secret ceiling\n(±10%)"), ("secret_s0.20", "Secret ceiling\n(±20%)"),
             ("range", "Publish a\n±10% range"), ("more_bidders", "One more\nbidder")]
    H = 5.0
    fig = plt.figure(figsize=(10, H))
    title(fig, "What would other rules do to prices and failed tenders?",
          "Change in the expected winning price vs today's non-binding public estimate (left) and share of tenders with no\n"
          f"acceptable bid (right), from the estimated cost distribution. Bars: 95% bootstrap intervals ({res['bootstrap_reps']} replications).",
          source="Author's estimates; PPDA open contracting data (bids)")
    frame(fig, H)
    for j, (metric, lab, col) in enumerate([("price_change_vs_status_quo", "Price change", BLUE), ("fail", "Tenders that fail", ORANGE)]):
        ax = fig.add_axes([0.2 + j * 0.42, 0.65 / H, 0.34, 1 - 1.95 / H])
        vals = [cf[k][metric] for k, _ in order]
        ci = [cf[k].get(metric + "_ci95", [v, v]) for (k, _), v in zip(order, vals)]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, color=col, height=0.6)
        ax.errorbar(vals, y, xerr=[[max(0, v - c[0]) for v, c in zip(vals, ci)], [max(0, c[1] - v) for v, c in zip(vals, ci)]],
                    fmt="none", ecolor=INK, lw=1, capsize=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, [l for _, l in order] if j == 0 else [""] * len(order), fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(lab, fontsize=10, loc="left", color=INK)
        ax.tick_params(labelsize=8)
    save(fig, "chart_p4_counterfactuals.png", "p4_f5_counterfactuals.png")

    # F6: secret-ceiling trade-off
    cv = pd.DataFrame(res["secret_curve"])
    H = 5.0
    fig = plt.figure(figsize=(9, H))
    title(fig, "The price of secrecy: cheaper contracts, more failed tenders",
          "Each point is a secret ceiling with more uncertainty for bidders\n"
          "(s = spread of the hidden ceiling around the published estimate).",
          source="Author's estimates")
    frame(fig, H)
    ax = fig.add_axes([0.1, 0.9 / H, 0.85, 1 - 2.1 / H])
    ax.plot(cv.fail, cv.price_change, "o-", color=BLUE, lw=2.5, ms=7)
    for rr_ in cv.itertuples():
        ax.annotate(f"s = {rr_.s:.2f}", (rr_.fail, rr_.price_change), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Tenders that fail (no bid under the hidden ceiling)", fontsize=9)
    ax.set_ylabel("Price change vs today", fontsize=9)
    ax.tick_params(labelsize=8)
    save(fig, "chart_p4_secrecy_tradeoff.png", "p4_f6_secrecy_tradeoff.png")


if __name__ == "__main__" and not ({"--extras", "--charts"} & set(sys.argv)):
    main()


def chart_lowest(x):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visuals import BLUE, ORANGE, INK, INK2, title
    res = json.loads((ROOT / "paper" / "structural_bids.json").read_text())
    bl = res["bunching_lowest_vs_other_bids"]
    lowest = x.groupby("ocid").beta.min()
    other = x[~x.index.isin(x.groupby("ocid").beta.idxmin())].beta
    H = 4.6
    fig = plt.figure(figsize=(10, H))
    title(fig, "Winning bids crowd just under the estimate; most losing bids sit above it",
          "Bids per 0.5% bin relative to the published estimate, lowest bid in each auction (left) vs all other bids (right).\n"
          "Excess mass in 0.97–1.00 against a smooth counterfactual, with 95% bootstrap intervals.",
          source="PPDA open contracting data (bids); author's estimates")
    for t in fig.texts:
        xx, yy = t.get_position()
        t.set_y(1 - (0.3 if yy > 0.9 else 0.68) / H) if yy > 0.5 else t.set_y(0.2 / H)
    edges = np.arange(0.80, 1.2001, 0.005)
    for j, (v, lab, key, col) in enumerate([(lowest, "Lowest bid", "lowest", ORANGE), (other, "Other bids", "other", BLUE)]):
        ax = fig.add_axes([0.07 + j * 0.48, 0.65 / H, 0.42, 1 - 1.9 / H])
        cnt, _ = np.histogram(v[(v >= 0.8) & (v < 1.2)], edges)
        mids = (edges[:-1] + edges[1:]) / 2
        ax.bar(mids, cnt, width=0.0045, color=[col if 0.97 <= m < 1.0 else "#b9bcc2" for m in mids])
        ax.axvline(1.0, color=INK2, lw=1, ls="--")
        e = bl[key]
        ax.set_title(f"{lab}: {e['excess_ratio']:.1f}× excess (95% CI {e['ci95'][0]:.1f}–{e['ci95'][1]:.1f})", fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("Bid / published estimate", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.savefig(ROOT / "posts" / "media" / "chart_p4_lowest_vs_other.png", dpi=200)
    shutil.copy(ROOT / "posts" / "media" / "chart_p4_lowest_vs_other.png", ROOT / "paper" / "figures" / "p4_f7_lowest_vs_other.png")
    plt.close(fig)
    print("wrote chart_p4_lowest_vs_other.png")


if __name__ == "__main__" and "--charts" in sys.argv:
    xx_, _ = load()
    rr_ = pseudo_costs(xx_)
    charts(xx_, rr_, json.loads((ROOT / "paper" / "structural_bids.json").read_text()), CostDist(rr_[~rr_.trimmed & (rr_.c > 0)].c))
    chart_lowest(xx_)
    sys.exit()


# ---------------------------------------------------------------- extras (added after the main run; no re-bootstrap)
def extras():
    """(a) counterfactuals with costs from auctions with 3+ bidders only (2-bidder auctions fail V2/V3);
    (b) where does the award spike come from: awards at the estimate whose lowest bid was not at the estimate."""
    f = ROOT / "paper" / "structural_bids.json"
    res = json.loads(f.read_text())
    x, _ = load()
    r = pseudo_costs(x)
    ok3 = r[~r.trimmed & (r.c > 0) & (r.ng >= 3)]
    nw3 = x[x.n >= 3].drop_duplicates("ocid").n.clip(upper=NMAX_CF).value_counts(normalize=True).to_dict()
    res["counterfactuals_n3plus"] = counterfactuals(CostDist(ok3.c), nw3, seed=5)
    a = x.groupby("ocid").agg(minb=("beta", "min"), est=("estimate_ugx", "first"), award=("award_value_ugx", "first"), n=("n", "first"))
    a = a[a.award.notna() & (a.award > 0)]
    a["award_rel"] = a.award / a.est
    in_win = lambda v: (v >= 0.97) & (v <= 1.0)
    res["award_vs_bid_spike"] = dict(
        auctions=int(len(a)),
        award_in_97_100=float(in_win(a.award_rel).mean()),
        lowest_bid_in_97_100=float(in_win(a.minb).mean()),
        award_in_window_lowest_bid_below_97=float((in_win(a.award_rel) & (a.minb < 0.97)).mean()),
        award_in_window_lowest_bid_above_100=float((in_win(a.award_rel) & (a.minb > 1.0)).mean()),
        award_exactly_estimate=float((np.abs(a.award_rel - 1) < 0.001).mean()),
        lowest_bid_exactly_estimate=float((np.abs(a.minb - 1) < 0.001).mean()),
    )
    f.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("award_vs_bid_spike",)}, indent=1))
    print({k: (round(v["price_change_vs_status_quo"], 4), round(v["fail"], 4)) for k, v in res["counterfactuals_n3plus"].items()})


if __name__ == "__main__" and "--extras" in sys.argv:
    extras()
