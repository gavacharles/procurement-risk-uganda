"""Social story for paper 4 ("The number every bidder aims for"), in the same house style as
road_progress_story.py (light page, kicker, bold hook, animated graphic, big number, footer).

Writes posts/media/story_estimate/: reel_square.mp4, reel_vertical.mp4, film_estimate.mp4, highlight.gif,
clips_square/NN.mp4, carousel/CNN.png + estimate_carousel.pdf.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import road_progress_story as rs  # noqa: E402
import structural_bids as sb  # noqa: E402
from road_progress_story import BLUE, ORANGE, INK, INK2, MUT, LGREY, ease, lf, font  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "posts" / "media" / "story_estimate"
rs.STORY = "THE NUMBER BIDDERS AIM FOR"


def axes_x(d, box, lo, hi, ticks, f, fmt=lambda v: f"{v:.2f}"):
    x0, y0, x1, y1 = box
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    d.line([x0, y1, x1, y1], fill=LGREY, width=2)
    for v in ticks:
        lab = fmt(v)
        d.text((X(v) - d.textlength(lab, font=f) / 2, y1 + 8), lab, font=f, fill=MUT)
    return X


def hist(d, box, t, v, col, lab, f, top=None, share=False):
    """Bars of bids per 0.5% of the estimate, 0.85-1.15; the band 0.97-1.00 in colour; bars grow with t."""
    edges = np.arange(0.85, 1.1501, 0.005)
    v = v - 1e-9  # bids exactly at the estimate count with those just under it ("at or just under")
    cnt, _ = np.histogram(v[(v >= 0.85) & (v < 1.15)], edges)
    if share:
        cnt = cnt / len(v)
    mids = (edges[:-1] + edges[1:]) / 2
    x0, y0, x1, y1 = box
    ya = y1 - f.size - 14
    X = axes_x(d, (x0, y0, x1, ya), 0.85, 1.15, [0.90, 1.00, 1.10], f, lambda v: f"{v:.0%}")
    top = top or cnt.max() * 1.1
    bw = (x1 - x0) / len(cnt)
    g = ease(t)
    for c, m in zip(cnt, mids):
        h = c / top * (ya - y0 - f.size - 10) * g
        d.rectangle([X(m) - bw / 2 + 1, ya - h, X(m) + bw / 2 - 1, ya], fill=col if 0.97 <= m < 1.0 else LGREY)
    d.line([X(1.0), y0 + f.size + 6, X(1.0), ya], fill=INK2, width=2)
    d.text((x0, y0), lab, font=f, fill=INK2)


def g_all(d, box, t, x=None, **_):
    f = lf(box)
    hist(d, box, t, x.beta.to_numpy(), BLUE, "every bid as a share of the published estimate; colour = at or just under it", f)


def g_split(d, box, t, x=None, **_):
    lowest = x.groupby("ocid").beta.min().to_numpy()
    other = x[~x.index.isin(x.groupby("ocid").beta.idxmin())].beta.to_numpy()
    x0, y0, x1, y1 = box
    f = lf(box, 0.9)
    wide = (x1 - x0) > 1.6 * (y1 - y0)
    if wide:
        mid = (x0 + x1) / 2
        boxes = [(x0, y0, mid - 24, y1), (mid + 24, y0, x1, y1)]
    else:
        mid = (y0 + y1) / 2
        boxes = [(x0, y0, x1, mid - 20), (x0, mid + 20, x1, y1)]
    top = 1.1 * max(np.histogram(v - 1e-9, np.arange(0.85, 1.1501, 0.005))[0].max() / len(v) for v in (lowest, other))
    hist(d, boxes[0], t, lowest, ORANGE, "winning (lowest) bids, share per 0.5% band", f, top=top, share=True)
    hist(d, boxes[1], t, other, BLUE, "losing bids, same scale", f, top=top, share=True)


def g_markups(d, box, t, res=None, **_):
    m = res["markups"]
    ns = ["2", "3", "4", "5", "6"]
    vals = [m[k]["median_markup"] for k in ns]
    x0, y0, x1, y1 = box
    f = lf(box)
    ax0, ay0, ay1 = x0 + 80, y0 + f.size + 20, y1 - f.size - 40
    X = lambda i: ax0 + 30 + i * (x1 - ax0 - 90) / (len(ns) - 1)
    Y = lambda v: ay1 - v / 0.30 * (ay1 - ay0)
    for v in (0, 0.1, 0.2, 0.3):
        d.line([ax0, Y(v), x1, Y(v)], fill=(225, 225, 222), width=1)
        d.text((x0, Y(v) - f.size / 2), f"{v:.0%}", font=f, fill=MUT)
    for i, k in enumerate(ns):
        lab = "6+" if k == "6" else k
        d.text((X(i) - d.textlength(lab, font=f) / 2, ay1 + 10), lab, font=f, fill=MUT)
    d.text((ax0, ay1 + 14 + f.size), "number of bidders", font=f, fill=MUT)
    d.text((x0, y0), "median markup: how far the bid sits above the bidder's cost", font=f, fill=MUT)
    g = ease(t) * (len(ns) - 1)
    pts = []
    for i in range(len(ns)):
        if i <= g:
            pts.append((X(i), Y(vals[i])))
        elif i - 1 < g:
            fr = g - (i - 1)
            pts.append((X(i - 1) + fr * (X(i) - X(i - 1)), Y(vals[i - 1] + fr * (vals[i] - vals[i - 1]))))
    if len(pts) > 1:
        d.line(pts, fill=BLUE, width=7, joint="curve")
    for i, (px, py) in enumerate(pts[: int(g) + 1]):
        d.ellipse([px - 10, py - 10, px + 10, py + 10], fill=BLUE)
        if t > 0.9:
            d.text((px + 14, py - f.size - 6), f"{vals[i]:.0%}", font=f, fill=INK)


def g_rules(d, box, t, res=None, **_):
    cf = res["counterfactuals"]
    rows = [("binding_public", "estimate as a hard maximum"), ("secret_s0.10", "secret ceiling (±10%)"),
            ("range", "publish a ±10% range"), ("more_bidders", "one more bidder")]
    x0, y0, x1, y1 = box
    f = font(min(lf(box, 0.95).size, 26))
    lab_w = max(d.textlength(l, font=f) for _, l in rows) + 20
    colw = (x1 - x0 - lab_w) / 2 - 20
    d.text((x0 + lab_w, y0), "price change", font=f, fill=MUT)
    d.text((x0 + lab_w + colw + 40, y0), "tenders that fail", font=f, fill=MUT)
    step = (y1 - y0 - f.size - 20) / len(rows)
    g = ease(t)
    for i, (k, lab) in enumerate(rows):
        y = y0 + f.size + 20 + i * step
        bh = min(step * 0.5, 54)
        cy = y + step / 2 - bh / 2
        d.text((x0, cy + bh / 2 - f.size / 2 - 2), lab, font=f, fill=INK)
        p, fl = cf[k]["price_change_vs_status_quo"], cf[k]["fail"]
        pw = abs(p) / 0.14 * (colw - 90) * g
        fw = fl / 0.20 * (colw - 90) * g
        bx = x0 + lab_w
        d.rectangle([bx, cy, bx + pw, cy + bh], fill=BLUE)
        fx = bx + colw + 40
        d.rectangle([fx, cy, fx + max(fw, 3), cy + bh], fill=ORANGE)
        if t > 0.9:
            d.text((bx + pw + 10, cy + bh / 2 - f.size / 2 - 2), f"{p:+.1%}".replace("-", "−"), font=f, fill=INK2)
            d.text((fx + max(fw, 3) + 10, cy + bh / 2 - f.size / 2 - 2), f"{fl:.1%}", font=f, fill=INK2)


def g_secrecy(d, box, t, res=None, **_):
    C = res["secret_curve"]
    x0, y0, x1, y1 = box
    f = lf(box)
    ax0, ay0, ay1 = x0 + 90, y0 + f.size + 30, y1 - f.size - 44
    X = lambda v: ax0 + v / 0.33 * (x1 - ax0 - 20)
    Y = lambda v: ay0 + (-v) / 0.14 * (ay1 - ay0)
    for v in (0, -0.05, -0.10):
        d.line([ax0, Y(v), x1, Y(v)], fill=(225, 225, 222), width=1)
        d.text((x0, Y(v) - f.size / 2), f"{v:.0%}".replace("-", "−"), font=f, fill=MUT)
    for v in (0, 0.1, 0.2, 0.3):
        d.text((X(v) - 18, ay1 + 10), f"{v:.0%}", font=f, fill=MUT)
    d.text((ax0, ay1 + 14 + f.size), "tenders that fail (no bid under the hidden ceiling)", font=f, fill=MUT)
    d.text((x0, y0), "price change vs today, as bidders grow less sure where the secret ceiling is", font=f, fill=MUT)
    bp = res["counterfactuals"]["binding_public"]
    n = max(1, int(math.ceil(ease(t) * len(C))))
    pts = [(X(c["fail"]), Y(c["price_change"])) for c in C[:n]]
    if len(pts) > 1:
        d.line(pts, fill=ORANGE, width=6, joint="curve")
    for px, py in pts:
        d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=ORANGE)
    if t > 0.6:
        px, py = X(bp["fail"]), Y(bp["price_change_vs_status_quo"])
        d.ellipse([px - 13, py - 13, px + 13, py + 13], fill=BLUE)
        d.text((px + 18, py - f.size - 12), "public estimate as a hard maximum", font=f, fill=BLUE)


def beats(x, res):
    bl = res["bunching_lowest_vs_other_bids"]
    cf = res["counterfactuals"]
    m = res["markups"]
    s = res["sample"]
    return [
        dict(hook="Uganda publishes the estimated cost of what it buys. Bidders can see it.",
             stat=f"{s['bids']:,} bids", sub=f"in {s['auctions']} public auctions with two or more bids, almost all from 2019",
             draw=g_all, kw=dict(x=x)),
        dict(hook="Winning bids crowd just under the estimate; losing bids sit higher",
             stat=f"{bl['lowest']['excess_ratio'] + 1:.0f} times", sub="as many winning bids just under the estimate as a smooth pattern predicts",
             draw=g_split, kw=dict(x=x)),
        dict(hook="More bidders, thinner margins",
             stat=f"{m['2']['median_markup']:.0%} → {m['6']['median_markup']:.0%}", sub="median markup with two bidders vs six or more",
             draw=g_markups, kw=dict(res=res)),
        dict(hook="What if the rules changed?",
             stat=f"{cf['more_bidders']['price_change_vs_status_quo']:+.0%}".replace("-", "−"),
             sub="price from one more bidder per tender, with no failed tenders",
             draw=g_rules, kw=dict(res=res)),
        dict(hook="Hiding the estimate saves money, but tenders fail",
             stat=f"{cf['secret_s0.10']['fail']:.0%} vs {cf['binding_public']['fail']:.0%}",
             sub="tenders failing with a secret ceiling vs a published hard maximum",
             draw=g_secrecy, kw=dict(res=res)),
    ]


def main(still=False):
    x, _ = sb.load()
    res = json.loads((ROOT / "paper" / "structural_bids.json").read_text())
    B = beats(x, res)
    n = len(B)
    OUT.mkdir(parents=True, exist_ok=True)
    if still:
        for layout in rs.LAYOUTS:
            for k, b in enumerate(B):
                rs.frame(layout, b, k + 1, n, 1.0).save(OUT / f"still_{layout}_{k + 1}.png")
        return
    nf, na = int(rs.SEC * rs.FPS), int(rs.ANIM * rs.FPS)
    for d_ in ("clips_square", "carousel"):
        (OUT / d_).mkdir(parents=True, exist_ok=True)
    ts = [min(1.0, i / na) for i in range(nf)]
    for layout, name in [("square", "reel_square.mp4"), ("vertical", "reel_vertical.mp4"), ("film", "film_estimate.mp4")]:
        W, H = rs.LAYOUTS[layout]["W"], rs.LAYOUTS[layout]["H"]
        rs.encode((rs.frame(layout, b, k + 1, n, t) for k, b in enumerate(B) for t in ts), W, H, OUT / name)
        print("wrote", name, flush=True)
    pages = []
    for k, b in enumerate(B):
        rs.encode((rs.frame("square", b, k + 1, n, t) for t in ts), 1080, 1080, OUT / "clips_square" / f"{k + 1:02d}.mp4")
        pg = rs.frame("square", b, k + 1, n, 1.0)
        pg.save(OUT / "carousel" / f"C{k + 1:02d}.png")
        pages.append(pg)
    pages[0].save(OUT / "carousel" / "estimate_carousel.pdf", save_all=True, append_images=pages[1:])
    gif = [rs.frame("square", b, k + 1, n, ts[i]).resize((540, 540), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128)
           for k, b in enumerate(B) for i in range(0, nf, 3)]
    gif[0].save(OUT / "highlight.gif", save_all=True, append_images=gif[1:], duration=100, loop=0, optimize=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main(still="--still" in sys.argv)
