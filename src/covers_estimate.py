"""Covers for the paper 4 Substack post ("The number every bidder aims for"), in the style of src/covers.py.

Out (posts/media/): cover_substack_estimate.png (1456 x 816 @2x), cover_linkedin_estimate.png (1200 x 627 @2x)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from covers import BASE_CSS, shoot, OUT  # noqa: E402
import structural_bids as sb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def hist_svg(v, width, height, hi_col, lab):
    edges = np.arange(0.85, 1.1501, 0.0075)
    cnt, _ = np.histogram(v[(v >= 0.85) & (v < 1.15)], edges)
    mids = (edges[:-1] + edges[1:]) / 2
    top = cnt.max() * 1.08
    bw = width / len(cnt)
    bars = "".join(
        f'<rect x="{i * bw + 0.8:.1f}" y="{height - c / top * height:.1f}" width="{bw - 1.6:.1f}" height="{c / top * height:.1f}" rx="1.5" '
        f'fill="{hi_col if 0.97 <= m < 1.0 else "#3a4049"}"/>' for i, (c, m) in enumerate(zip(cnt, mids)))
    x1 = (1.0 - 0.85) / 0.30 * width
    return (f'<div><div class="muted" style="font-size:11px;margin-bottom:4px">{lab}</div>'
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{bars}'
            f'<line x1="{x1:.1f}" x2="{x1:.1f}" y1="0" y2="{height}" stroke="#b9bcc2" stroke-dasharray="3 3"/></svg></div>')


def policy_panel(cf, f):
    rows = [("binding_public", "Estimate as a binding maximum"), ("secret_s0.10", "Secret ceiling (±10%)"),
            ("range", "Publish a ±10% range"), ("more_bidders", "One more bidder")]
    out = []
    for k, lab in rows:
        p, fl = cf[k]["price_change_vs_status_quo"], cf[k]["fail"]
        pw, fw = abs(p) / 0.14 * 120 * f, fl / 0.2 * 90 * f
        out.append(f"""<div style="display:grid;grid-template-columns:{220 * f}px {175 * f}px {130 * f}px;align-items:center;gap:{10 * f}px;padding:{9 * f}px 0;border-top:1px solid #2c323a">
  <div style="font-size:{14 * f}px;color:#d9dbe0">{lab}</div>
  <div style="display:flex;align-items:center;gap:6px"><i style="display:inline-block;height:{12 * f}px;width:{pw:.0f}px;background:#3987e5;border-radius:3px"></i><span style="font-size:{13 * f}px">{p:+.0%}</span></div>
  <div style="display:flex;align-items:center;gap:6px"><i style="display:inline-block;height:{12 * f}px;width:{max(fw, 2):.0f}px;background:#f08a5d;border-radius:3px"></i><span style="font-size:{13 * f}px">{fl:.0%}</span></div>
</div>""")
    head = (f'<div style="display:grid;grid-template-columns:{220 * f}px {175 * f}px {130 * f}px;gap:{10 * f}px;font-size:{11.5 * f}px" class="muted">'
            f'<div>If the rules changed…</div><div>price</div><div>tenders that fail</div></div>')
    return head + "".join(out)


def cover(w, h, f, x, res):
    lowest = x.groupby("ocid").beta.min().to_numpy()
    other = x[~x.index.isin(x.groupby("ocid").beta.idxmin())].beta.to_numpy()
    bl = res["bunching_lowest_vs_other_bids"]
    cf = res["counterfactuals"]
    chips = ["2,725 bids", "Auction theory", "Structural estimation", "Bootstrap", "Collusion screens"]
    return f"""<!doctype html><html><head><meta charset="utf-8">{BASE_CSS}</head><body style="width:{w}px;height:{h}px">
<div style="position:absolute;left:{56 * f}px;top:{50 * f}px;width:{w * 0.47}px">
  <div class="kicker" style="font-size:{15 * f}px">Uganda · public procurement · auctions</div>
  <h1 style="font-size:{54 * f}px;margin-top:{14 * f}px">The number every bidder aims for</h1>
  <p class="muted" style="font-size:{19 * f}px;margin-top:{14 * f}px;line-height:1.4">What 2,725 bids reveal about the official cost estimate, and what would happen if it were hidden</p>
  <div style="margin-top:{20 * f}px;font-size:{13 * f}px">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div>
</div>
<div style="position:absolute;left:{56 * f}px;bottom:{40 * f}px;display:flex;align-items:flex-end;gap:{18 * f}px">
  {hist_svg(lowest, int(150 * f), int(78 * f), "#f08a5d", "winning bids")}
  {hist_svg(other, int(150 * f), int(78 * f), "#3987e5", "losing bids")}
  <div style="font-size:{14.5 * f}px;line-height:1.35;margin-left:{6 * f}px"><b style="font-size:{36 * f}px;color:#f08a5d">{bl['lowest']['excess_ratio']:.0f}×</b><br>too many winning bids<br>just under the estimate</div>
</div>
<div style="position:absolute;right:{44 * f}px;top:{150 * f}px;width:{560 * f}px">{policy_panel(cf, f)}
  <div style="margin-top:{26 * f}px;display:flex;align-items:center;gap:{14 * f}px;font-size:{14.5 * f}px;line-height:1.35"><b style="font-size:{40 * f}px;color:#86b6ef">{cf['more_bidders']['price_change_vs_status_quo']:+.0%}</b>
  <span>price change from one more bidder<br>per tender, with no failed tenders</span></div></div>
<div style="position:absolute;right:{44 * f}px;bottom:{30 * f}px;text-align:right;font-size:{14 * f}px">
  <div class="by">By <b>Gava Charles</b></div><div class="url">gavacharles.github.io/procurement-risk-uganda</div></div>
</body></html>"""


def main():
    x, _ = sb.load()
    res = json.loads((ROOT / "paper" / "structural_bids.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    shoot(cover(1456, 816, 1.2, x, res), "cover_substack_estimate.png", 1456, 816)
    shoot(cover(1200, 627, 1.0, x, res), "cover_linkedin_estimate.png", 1200, 627)


if __name__ == "__main__":
    main()
