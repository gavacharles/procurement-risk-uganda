"""(6) Cross-country comparison: is bunching at the estimate unique to Uganda?

For each OCDS publisher (OCP Data Registry), download one recent full year (skip files
> MAX_MB), and compute on contracts with both an estimate (tender.value) and an award
(awards[].value) in the same currency:
  share_99_100 : award at 99-100% of estimate
  share_exact  : award exactly equal to estimate
  bunching b   : excess mass in [0.97,1.00) vs a smooth counterfactual (same estimator as Uganda)
Computed for all categories and, where mainProcurementCategory exists, for works.

Out: data/raw/registry/<id>_<year>.jsonl.gz, paper/cross_country.json, posts/media/chart_cross_country.png
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import bunching  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path.home() / ".cache" / "procurement-risk" / "registry"  # outside cloud-synced folder
UA = "procurement-risk-research (github.com/gavacharles/procurement-risk-uganda)"
MAX_MB = 300
PUBS = {3: "Zambia", 147: "Kenya", 85: "Ghana", 64: "Nigeria (federal)", 145: "Rwanda", 152: "Tanzania",
        156: "Liberia", 22: "Dominican Republic", 63: "Paraguay", 110: "Ecuador", 61: "Colombia"}


def get(url, method="GET"):
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method=method)
    return urllib.request.urlopen(req, timeout=300)


def pick_year(pid: int) -> str | None:
    html = get(f"https://data.open-contracting.org/en/publication/{pid}").read().decode("utf-8", "ignore")
    years = sorted(set(re.findall(r"download\?name=(\d{4})\.jsonl\.gz", html)))
    full = [y for y in years if int(y) <= 2025]
    return (full or years or [None])[-1]


def download(pid: int, year: str) -> Path | None:
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"{pid}_{year}.jsonl.gz"
    if f.exists() and f.stat().st_size > 0:
        return f
    url = f"https://data.open-contracting.org/en/publication/{pid}/download?name={year}.jsonl.gz"
    try:
        size = int(get(url, "HEAD").headers.get("Content-Length") or 0)
    except Exception:
        size = 0
    if size and size > MAX_MB * 1e6:
        print(f"  skip {pid} {year}: {size / 1e6:.0f} MB")
        return None
    tmp = f.with_suffix(".part")
    with get(url) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out, 1 << 20)
    tmp.replace(f)
    return f


def ratios(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get("tender") or {}
            tv = t.get("value") or {}
            est, cur = tv.get("amount"), tv.get("currency")
            if not est or est <= 0:
                continue
            cat = t.get("mainProcurementCategory")
            cat = cat[0] if isinstance(cat, list) and cat else cat
            for a in r.get("awards") or []:
                av = a.get("value") or {}
                amt = av.get("amount")
                if amt and amt > 0 and (av.get("currency") in (None, cur) or cur is None):
                    rows.append((amt / est, cat))
                    break
    return pd.DataFrame(rows, columns=["ratio", "cat"])


def metrics(d: pd.DataFrame) -> dict:
    r = d.ratio[(d.ratio > 0.3) & (d.ratio < 3)]
    out = dict(n=int(len(r)), share_99_100=float(r.between(0.99, 0.999999).mean()) if len(r) else None,
               share_exact=float(np.isclose(r, 1.0).mean()) if len(r) else None)
    if out["share_exact"] is not None and out["share_exact"] > 0.5:
        out["artefact"] = "award value copied into estimate field (>90% exactly equal); excluded"
        return out
    if len(r[(r >= 0.8) & (r < 1.2)]) >= 400:
        b = bunching(r, n_boot=100)
        out.update(bunching_b=b["excess_ratio"], ci95=b["ci95"])
    return out


def main():
    res = {}
    for pid, name in PUBS.items():
        try:
            y = pick_year(pid)
            print(f"{name}: year {y}", flush=True)
            f = download(pid, y) if y else None
            if not f:
                res[name] = dict(status="skipped (too large or unavailable)", year=y)
                continue
            d = ratios(f)
            res[name] = dict(year=y, all=metrics(d))
            w = d[d.cat == "works"]
            if len(w) >= 100:
                res[name]["works"] = metrics(w)
            print("  ", res[name], flush=True)
        except Exception as e:
            res[name] = dict(status=f"error: {e}")
            print("  error", e, flush=True)
    u = json.loads((ROOT / "paper" / "results.json").read_text())
    res["Uganda"] = dict(year="2015-2026", works=dict(n=u["bunching_main"]["n"], share_99_100=u["share_within_1pct_below"],
                                                      share_exact=u["share_exactly_equal"],
                                                      bunching_b=u["bunching_main"]["excess_ratio"], ci95=u["bunching_main"]["ci95"]))
    (ROOT / "paper" / "cross_country.json").write_text(json.dumps(res, indent=2, default=float))
    chart(res)


def chart(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visuals import BLUE, ORANGE, INK2, title
    rows = []
    for k, v in res.items():
        m = v.get("works") or v.get("all")
        if m and m.get("share_99_100") is not None and m["n"] >= 200 and not m.get("artefact"):
            rows.append((k, m["share_99_100"], m["n"], "works" if v.get("works") else "all"))
    rows.sort(key=lambda x: x[1])
    fig = plt.figure(figsize=(8, 0.42 * len(rows) + 2))
    ax = fig.add_axes([0.26, 0.14, 0.66, 0.68])
    title(fig, "Is Uganda unusual? Awards at 99–100% of the estimate",
          "Share of contracts; works where the data has a category, otherwise all contracts",
          source="Source: OCP Data Registry (copy-artefact feeds excluded)")
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[ORANGE if r[0] == "Uganda" else BLUE for r in rows], height=0.6)
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.005, i, f"{r[1]:.0%}  (n={r[2]:,}, {r[3]})", va="center", fontsize=8.5, color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.45)
    out = ROOT / "posts" / "media" / "chart_cross_country.png"
    fig.savefig(out, dpi=200)
    shutil.copy(out, ROOT / "docs" / "media" / out.name)


if __name__ == "__main__":
    main()
