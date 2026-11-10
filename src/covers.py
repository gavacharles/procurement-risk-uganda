"""Post covers and infographic, rendered from HTML with headless Chrome (real data, no firm names).

Out (posts/media/):
  cover_linkedin.png     1200 x 627   (LinkedIn link/post image)
  cover_substack.png     1456 x 816   (Substack header / social preview, 16:9)
  infographic_maths.png  1080 x 1350  (LinkedIn feed portrait)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import bunching  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "posts" / "media"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def uganda_svg(width: int, height: int, metric="value_per_capita") -> str:
    g = json.loads((ROOT / "docs" / "data" / "districts.geojson").read_text())
    md = json.loads((ROOT / "docs" / "data" / "district_metrics.json").read_text())
    vals = {d: v["all|all"][metric] for d, v in md["data"].items() if "all|all" in v and v["all|all"].get(metric)}
    qs = np.quantile(list(vals.values()), [1 / 7 * i for i in range(1, 7)])
    lon0, lon1, lat0, lat1 = 29.5, 35.1, -1.55, 4.3
    sx, sy = width / (lon1 - lon0), height / (lat1 - lat0)
    s = min(sx, sy)

    def pt(c):
        return f"{(c[0] - lon0) * s:.1f},{(lat1 - c[1]) * s:.1f}"

    paths = []
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        d = "".join("M" + "L".join(pt(c) for c in ring) + "Z" for poly in polys for ring in poly)
        v = vals.get(f["properties"]["district"])
        fill = RAMP[int(np.searchsorted(qs, v))] if v else "#3a4049"
        paths.append(f'<path d="{d}" fill="{fill}" stroke="#14171b" stroke-width="0.6"/>')
    return f'<svg viewBox="0 0 {(lon1 - lon0) * s:.0f} {(lat1 - lat0) * s:.0f}" width="{width}" height="{height}">{"".join(paths)}</svg>'


def spike_svg(width: int, height: int) -> str:
    w = pd.read_parquet(P / "works_tenders_scored.parquet")
    b = bunching(w.award_to_est.dropna(), lo=0.85, hi=1.10, n_boot=10)
    cnt, pred = b["cnt"], b["pred"]
    n, top = len(cnt), cnt.max() * 1.05
    bw = width / n
    bars = "".join(
        f'<rect x="{i * bw + 1:.1f}" y="{height - c / top * height:.1f}" width="{bw - 2:.1f}" height="{c / top * height:.1f}" rx="2" fill="#3987e5"/>'
        for i, c in enumerate(cnt))
    line = "M" + "L".join(f"{(i + 0.5) * bw:.1f},{height - max(p, 0) / top * height:.1f}" for i, p in enumerate(pred))
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{bars}'
            f'<path d="{line}" fill="none" stroke="#f08a5d" stroke-width="3"/></svg>')


BASE_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Source+Serif+4:wght@600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#14171b;color:#f0efec;font-family:Inter,system-ui,sans-serif;overflow:hidden}
.kicker{color:#f08a5d;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
h1{font-family:"Source Serif 4",Georgia,serif;font-weight:700;line-height:1.08}
.muted{color:#b9bcc2}
.chip{display:inline-block;border:1px solid #3a4049;border-radius:99px;padding:5px 12px;margin:0 6px 8px 0;color:#d9dbe0;font-weight:600}
.by{color:#d9dbe0;font-weight:600}.by b{color:#fff}
.url{color:#86b6ef}
.legend{display:flex;gap:3px;align-items:center;color:#b9bcc2}
.legend i{display:inline-block;width:22px;height:9px;border-radius:2px}
</style>"""


def legend_html(label):
    return ('<div class="legend" style="font-size:13px">low' + "".join(f'<i style="background:{c}"></i>' for c in RAMP)
            + f'high <span style="margin-left:8px">{label}</span></div>')


def cover(w, h, scale):
    chips = ["Statistics", "Graph theory", "Linear algebra", "Machine learning", "NLP", "Optimisation", "Number theory"]
    f = scale
    return f"""<!doctype html><html><head><meta charset="utf-8">{BASE_CSS}</head><body style="width:{w}px;height:{h}px">
<div style="position:absolute;left:{56 * f}px;top:{50 * f}px;width:{w * 0.56}px">
  <div class="kicker" style="font-size:{15 * f}px">Uganda · public works procurement</div>
  <h1 style="font-size:{56 * f}px;margin-top:{16 * f}px">Reading a country's contracts with mathematics</h1>
  <p class="muted" style="font-size:{19 * f}px;margin-top:{16 * f}px;line-height:1.4">What statistics, networks and machine learning reveal in 237,000 public procurement records</p>
  <div style="margin-top:{22 * f}px;font-size:{13 * f}px">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div>
</div>
<div style="position:absolute;left:{56 * f}px;bottom:{44 * f}px;display:flex;align-items:flex-end;gap:{22 * f}px">
  <div>{spike_svg(int(230 * f), int(92 * f))}</div>
  <div style="font-size:{15 * f}px;line-height:1.35"><b style="font-size:{34 * f}px;color:#f08a5d">5.3×</b><br>too many winning prices<br>just below the estimate</div>
</div>
<div style="position:absolute;right:{40 * f}px;top:{46 * f}px">{uganda_svg(int(w * 0.36), int(h * 0.72))}
  <div style="margin-top:{6 * f}px">{legend_html("works value per person")}</div></div>
<div style="position:absolute;right:{44 * f}px;bottom:{30 * f}px;text-align:right;font-size:{14 * f}px">
  <div class="by">By <b>Gava Charles</b></div><div class="url">gavacharles.github.io/procurement-risk-uganda</div></div>
</body></html>"""


def infographic():
    rows = [
        ("📈", "Statistics &amp; calculus", "Do prices land where competition predicts?", "<b>5.3×</b> too many awards just below the estimate"),
        ("🔢", "Benford's Law", "Do the numbers look natural?", "Marginal fit, mostly from round pricing"),
        ("🕸️", "Graph theory", "Who bids against whom?", "<b>205</b> groups that rotate wins; one trio bid together <b>255</b>×"),
        ("🧮", "Linear algebra", "Who sits at the centre?", "Eigenvectors find the hub firms"),
        ("🤖", "Machine learning", "What looks unusual?", "<b>3×</b> agreement with rules; <b>298</b> extra anomalies"),
        ("📝", "NLP", "What do titles reveal?", "<b>21</b> possible contract splits"),
        ("🔍", "Data forensics", "When does the public find out?", "Awards published <b>~3 months</b> late; <b>82%</b> entered after award"),
        ("🎯", "Optimisation", "Which contracts to audit?", "Knapsack audit plans for any budget"),
        ("🔐", "Number theory", "Can bids be judged blind?", "Anonymous bidder codes in a 2048-bit prime field"),
    ]
    items = "".join(f"""<div style="display:grid;grid-template-columns:54px 1fr;gap:14px;padding:16px 0;border-top:1px solid #2c323a">
      <div style="font-size:34px;line-height:1">{e}</div>
      <div><div style="font-size:21px;font-weight:700">{m}</div>
        <div class="muted" style="font-size:16px;margin-top:2px">{q}</div>
        <div style="font-size:18.5px;margin-top:6px;color:#f0efec">{a}</div></div></div>""" for e, m, q, a in rows)
    return f"""<!doctype html><html><head><meta charset="utf-8">{BASE_CSS}<style>b{{color:#f08a5d}}</style></head>
<body style="width:1080px;height:1350px;padding:58px 64px">
  <div class="kicker" style="font-size:15px">Uganda · public works procurement</div>
  <h1 style="font-size:50px;margin:12px 0 8px">Nine kinds of mathematics,<br>one procurement record</h1>
  <p class="muted" style="font-size:19px;margin-bottom:18px">Each tool asks a different question of 237,000 public records. Here's what each found.</p>
  {items}
  <div style="position:absolute;left:64px;right:64px;bottom:44px;display:flex;justify-content:space-between;font-size:15px">
    <span class="by">By <b style="color:#fff">Gava Charles</b> · risk indicators, not findings of wrongdoing</span><span class="url">gavacharles.github.io/procurement-risk-uganda</span></div>
</body></html>"""


def shoot(html: str, name: str, w: int, h: int):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "page.html"
        f.write_text(html)
        png = Path(td) / "out.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                        "--force-device-scale-factor=2", "--virtual-time-budget=8000", f"--user-data-dir={td}/prof",
                        f"--screenshot={png}", f.as_uri()], check=True, capture_output=True)
        dst = OUT / name
        dst.unlink(missing_ok=True)
        dst.write_bytes(png.read_bytes())
    print("wrote", name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shoot(cover(1200, 627, 1.0), "cover_linkedin.png", 1200, 627)
    shoot(cover(1456, 816, 1.2), "cover_substack.png", 1456, 816)
    shoot(infographic(), "infographic_maths.png", 1080, 1350)


if __name__ == "__main__":
    main()
