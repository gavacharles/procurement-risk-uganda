"""Social story for paper 3 ("Tender to tarmac"), in the house style of the Kiira last-mile stories
(~/last_mile_work/kiira/scripts/55_kiira_story_media.py): light page, grey kicker "NN / 07 · STORY · SERIES",
bold hook, animated graphic, big blue number with a one-line sub, site link and name in the footer.

Frames are drawn directly with PIL (no browser) and piped to ffmpeg.
Writes OUT/: reel_square.mp4 (1080x1080), reel_vertical.mp4 (1080x1920), film_tarmac.mp4 (1280x720),
highlight.gif, clips_square/NN.mp4, carousel/CNN.png + tarmac_carousel.pdf.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import road_progress as rp  # noqa: E402
from road_progress_charts import km_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "posts" / "media" / "story_tarmac"
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BG, INK, INK2, MUT = (244, 244, 242), (11, 11, 11), (82, 81, 78), (137, 135, 129)
BLUE, ORANGE, RED, GREY, LGREY, GREEN = (42, 120, 214), (235, 104, 52), (192, 57, 43), (74, 79, 87), (190, 192, 196), (127, 183, 126)
GRAVEL, ASPH, VEG, NOD = (217, 135, 58), (74, 79, 87), (127, 183, 126), (232, 232, 229)
FPS, SEC, ANIM = 30, 3.8, 2.2
SERIES, STORY = "PROCUREMENT RISK UGANDA", "TENDER TO TARMAC"
LINK = "gavacharles.github.io/procurement-risk-uganda"
FONTS = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in FONTS:
        FONTS[k] = ImageFont.truetype(FB if bold else FR, size)
    return FONTS[k]


def lf(box, k=1.0):
    """Label font sized to the smaller side of the graphic area."""
    x0, y0, x1, y1 = box
    return font(max(16, int(k * min(x1 - x0, y1 - y0) / 24)))


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def wrap(draw, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=f) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


# ---------------------------------------------------------------- layouts
LAYOUTS = {
    "square": dict(W=1080, H=1080, m=56, kick=(40, 22), head=(84, 54), stage=(250, 850), stat=(896, 62), sub=(966, 27), foot=(1030, 22)),
    "vertical": dict(W=1080, H=1920, m=64, kick=(70, 24), head=(120, 64), stage=(360, 1520), stat=(1600, 78), sub=(1694, 32), foot=(1840, 26)),
    "film": dict(W=1280, H=720, m=56, kick=(40, 20), head=(76, 46), stage=(150, 540), stat=(552, 50), sub=(612, 23), foot=(668, 19)),
}


# ---------------------------------------------------------------- data
def load():
    R = pd.read_parquet(P / "road_progress_roads.parquet")
    G = pd.read_parquet(P / "road_progress_segments.parquet")
    geo = json.loads((ROOT / "data" / "raw" / "geo" / "uga_districts_ubos2020.geojson").read_text())
    routes = {}
    for r in R.itertuples():
        road = next((x for x in rp.ROADS if x[0] == r.id), None)
        g = rp.geometry(road) if road else None
        if g and not str(r.status).startswith("excluded"):
            routes[r.id] = (np.array(g["coords"]), r.kind, r.status, r.contract_km)
    return R, G, geo, routes


# ---------------------------------------------------------------- graphics (drawn into box = x0, y0, x1, y1; t in [0, 1])
def g_dots(d, box, t, **_):
    """1,000 dots = 142,168 contracts; none has a start or end date."""
    x0, y0, x1, y1 = box
    n_c, n_r = 40, 25
    cell = min((x1 - x0) / n_c, (y1 - y0) / n_r)
    ox = x0 + ((x1 - x0) - cell * n_c) / 2
    oy = y0 + ((y1 - y0) - cell * n_r) / 2
    shown = int(ease(t) * n_c * n_r)
    rad = cell * 0.34
    for k in range(shown):
        i, j = k % n_c, k // n_c
        cx, cy = ox + (i + 0.5) * cell, oy + (j + 0.5) * cell
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=LGREY)
    if t > 0.85:
        f = font(max(18, int(cell * 0.9)))
        d.text((ox, oy + n_r * cell + cell * 0.5), "each dot = 142 contracts in the open data", font=f, fill=MUT)


def project(xy, box, ext=(29.5, -1.5, 35.1, 4.3)):
    x0, y0, x1, y1 = box
    lx0, ly0, lx1, ly1 = ext
    kx = math.cos(math.radians(1.3))
    w, h = (lx1 - lx0) * kx, (ly1 - ly0)
    s = min((x1 - x0) / w, (y1 - y0) / h)
    ox = x0 + ((x1 - x0) - w * s) / 2
    oy = y0 + ((y1 - y0) - h * s) / 2
    return np.c_[ox + (xy[:, 0] - lx0) * kx * s, oy + (ly1 - xy[:, 1]) * s]


STATUS_RGB = {"completed": BLUE, "substantial": (143, 184, 236), "sealed": (31, 61, 115), "incomplete": ORANGE,
              "nopaving": RED, "progress": LGREY, "control": GREY, "insufficient": (220, 220, 220)}


def skey(status):
    for pre, k in [("completed", "completed"), ("substantially", "substantial"), ("incomplete", "incomplete"),
                   ("no paving detected past", "nopaving"), ("control", "control"), ("rehabilitation", "control"),
                   ("sealed", "sealed"), ("insufficient", "insufficient")]:
        if str(status).startswith(pre):
            return k
    return "progress"


def g_map(d, box, t, geo=None, routes=None, **_):
    for f in geo["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            pts = project(np.array(poly[0]), box)
            d.polygon([tuple(p) for p in pts], fill=(236, 235, 231), outline=(214, 212, 207))
    tt = ease(t)
    colour = t > 0.75
    w = max(3, int((box[2] - box[0]) / 170))
    for rid, (xy, kind, status, km) in routes.items():
        pts = project(xy, box)
        n = max(2, int(len(pts) * tt))
        c = STATUS_RGB[skey(status)] if colour else BLUE
        d.line([tuple(p) for p in pts[:n]], fill=c, width=w + int(min(km, 140) / 45), joint="curve")
    if colour:
        f = lf(box)
        items = [("completed", "completed"), ("incomplete", "overdue"), ("nopaving", "no paving past due"),
                 ("progress", "not yet due"), ("control", "rebuilding a paved road")]
        tw = max(d.textlength(lab, font=f) for _, lab in items)
        lx = box[2] - tw - 34
        y = box[1] + 4
        for k, lab in items:
            d.rectangle([lx, y + f.size * 0.25, lx + 22, y + f.size * 0.25 + 14], fill=STATUS_RGB[k])
            d.text((lx + 30, y), lab, font=f, fill=INK2)
            y += f.size + 10


def g_strip(d, box, t, G=None, rid=None, **_):
    order = [c for c in G.columns if len(c) == 5 and c[:4].isdigit()]
    g = G[G.id == rid].sort_values("seg")
    code = {"unpaved": GRAVEL, "asphalt": ASPH, "vegetated": VEG}
    x0, y0, x1, y1 = box
    y0 += 2 * lf(box, 0.85).size + 20
    lab_h = 40
    nx, ny = len(order), len(g)
    cw, ch = (x1 - x0) / nx, (y1 - y0 - lab_h) / ny
    shown = int(math.ceil(ease(t) * nx))
    for i, c in enumerate(order[:shown]):
        for j, v in enumerate(g[c].tolist()):
            yy = y1 - lab_h - (j + 1) * ch
            d.rectangle([x0 + i * cw, yy, x0 + (i + 1) * cw - 1, yy + ch - 1], fill=code.get(v, NOD))
    box = (x0, y0 - 2 * lf(box, 0.85).size - 20, x1, y1)
    f = lf(box)
    for i, c in enumerate(order):
        if c.endswith("a") and int(c[:4]) % 2 == 0:
            d.text((x0 + i * cw, y1 - lab_h + 8), c[:4], font=f, fill=MUT)
    fs = lf(box, 0.85)
    d.text((x0, y0 - 2 * fs.size - 16), "each row = 1 km of road · each column = one reading", font=fs, fill=MUT)
    lx = x0
    for lab, col in [("asphalt", ASPH), ("unpaved", GRAVEL), ("vegetation", VEG)]:
        d.rectangle([lx, y0 - fs.size - 4, lx + 16, y0 - fs.size + 12], fill=col)
        d.text((lx + 22, y0 - fs.size - 8), lab, font=fs, fill=INK2)
        lx += 22 + d.textlength(lab, font=fs) + 26


def g_decomp(d, box, t, R=None, **_):
    D = R[R.t_launch.notna() & ~R.kind.str.startswith("control") & ~R.status.str.startswith("insufficient")].sort_values("t_tender")
    x0, y0, x1, y1 = box
    f = lf(box)
    lab_w = max(d.textlength(r.road.split("–")[0] + "–" + r.road.split("–")[-1].split(" (")[0], font=f) for r in D.itertuples()) + 16
    stacked = lab_w > 0.33 * (x1 - x0)  # narrow layouts: road name above its bar
    if stacked:
        lab_w = 0
    scale = (x1 - x0 - lab_w - d.textlength("5.5 + 4.7 y", font=f) - 20) / 10.2
    step = 1.6 if not stacked else 1.6 + (f.size + 8) / 70
    bh = min(70 if stacked else 110, (y1 - y0 - 3 * f.size - 40) / (len(D) * step))
    tt = ease(t)
    y = y0
    for k, r in enumerate(D.itertuples()):
        name = r.road.split("–")[0] + "–" + r.road.split("–")[-1].split(" (")[0]
        if stacked:
            y = y0 + k * (bh * 1.6 + f.size + 8) + f.size + 8
            d.text((x0, y - f.size - 8), name, font=f, fill=INK)
        else:
            y = y0 + k * bh * 1.6 + 10
            d.text((x0, y + bh / 2 - f.size / 2 - 2), name, font=f, fill=INK)
        pre = (r.t_launch - r.t_tender)
        end = (r.t_complete if pd.notna(r.t_complete) else r.t_last) - r.t_tender
        bx = x0 + lab_w
        grow = end * tt
        d.rectangle([bx, y, bx + min(grow, pre) * scale, y + bh], fill=LGREY)
        if grow > pre:
            col = BLUE if pd.notna(r.t_complete) else ORANGE
            d.rectangle([bx + pre * scale, y, bx + grow * scale, y + bh], fill=col)
        if t > 0.9:
            d.text((bx + end * scale + 10, y + bh / 2 - f.size / 2 - 2), f"{pre:.1f} + {end - pre:.1f} y", font=f, fill=INK2)
    fs = lf(box, 0.85)
    yy = (y + bh + 30)
    bx = x0 + lab_w
    for lab, col in [("tender to works start", LGREY), ("construction (done)", BLUE), ("construction (ongoing)", ORANGE)]:
        w_ = 22 + d.textlength(lab, font=fs)
        if bx + w_ > x1:
            bx, yy = x0 + lab_w, yy + fs.size + 12
        d.rectangle([bx, yy + 4, bx + 16, yy + 20], fill=col)
        d.text((bx + 22, yy), lab, font=fs, fill=INK2)
        bx += w_ + 30


def g_km(d, box, t, R=None, **_):
    D = R[~R.kind.str.startswith("control") & R.t_tender.notna() & ~R.status.str.startswith(("sealed", "insufficient", "excluded"))].copy()
    D["dur"] = np.where(D.t_complete.notna(), D.t_complete - D.t_tender, D.t_last - D.t_tender)
    D["event"] = D.t_complete.notna()
    D = D[D.dur > 0]
    xs, ys = km_curve(D.dur.to_numpy(), D.event.to_numpy())
    at_risk = np.array([(D.dur >= x).sum() for x in xs])
    xs, ys = xs[at_risk >= 3], ys[at_risk >= 3]
    x0, y0, x1, y1 = box
    f = lf(box)
    ax0, ax1, ay0, ay1 = x0 + 70, x1 - 20, y0 + 10, y1 - 50
    X = lambda v: ax0 + v / 10 * (ax1 - ax0)
    Y = lambda v: ay1 - v * (ay1 - ay0)
    for v in (0, 0.25, 0.5, 0.75, 1):
        d.line([ax0, Y(v), ax1, Y(v)], fill=(225, 225, 222), width=1)
        d.text((x0, Y(v) - f.size / 2), f"{v:.0%}", font=f, fill=MUT)
    for v in range(0, 11, 2):
        d.text((X(v) - 8, ay1 + 10), str(v), font=f, fill=MUT)
    d.text((ax1 - d.textlength("years since tender", font=f), ay1 + 10 + f.size + 4), "years since tender", font=f, fill=MUT)
    xm = 10 * ease(t)
    pts = []
    for i, (xv, yv) in enumerate(zip(xs, 1 - ys)):
        if xv > xm:
            break
        nxt = xs[i + 1] if i + 1 < len(xs) else xs[-1] + 0.6
        pts += [(X(xv), Y(yv)), (X(min(nxt, xm)), Y(yv))]
    if len(pts) > 1:
        d.line(pts, fill=BLUE, width=6)
    d.line([X(4), ay0, X(4), ay1], fill=MUT, width=2)
    d.text((X(4) + 8, ay0), "4 years: none finished", font=f, fill=INK2)
    d.text((x0, y0 - f.size - 8), "share of contracted roads finished", font=f, fill=MUT)


def g_media(d, box, t, R=None, **_):
    C = R[R.t_media_done.notna() & R.t_complete.notna() & ~R.early_sealed.astype(bool)]
    x0, y0, x1, y1 = box
    f = lf(box)
    side = min(x1 - x0 - 90, y1 - y0 - 70)
    ax0, ay1 = x0 + 80 + ((x1 - x0 - 90) - side) / 2, y0 + side
    lo, hi = 2019.5, 2023.5
    X = lambda v: ax0 + (v - lo) / (hi - lo) * side
    Y = lambda v: ay1 - (v - lo) / (hi - lo) * side
    d.rectangle([ax0, ay1 - side, ax0 + side, ay1], outline=(220, 220, 217), width=2)
    d.line([X(lo), Y(lo), X(hi), Y(hi)], fill=LGREY, width=3)
    for v in range(2020, 2024):
        d.text((X(v) - 22, ay1 + 8), str(v), font=f, fill=MUT)
        d.text((ax0 - 64, Y(v) - f.size / 2), str(v), font=f, fill=MUT)
    d.text((ax0, ay1 + 14 + f.size), "reported finished (news)", font=f, fill=MUT)
    d.text((ax0, ay1 - side - f.size - 10), "finished by the satellite", font=f, fill=MUT)
    shown = int(math.ceil(ease(t) * len(C)))
    for r in C.sort_values("t_media_done").head(shown).itertuples():
        cx, cy = X(r.t_media_done), Y(r.t_complete)
        d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=BLUE)
        d.text((cx + 18, cy - f.size / 2 - 2), r.road.split("–")[0], font=f, fill=INK2)


# ---------------------------------------------------------------- beats
def beats(R, G, geo, routes):
    # summary numbers computed here, from the same rules as road_progress_charts.py
    D = R[~R.kind.str.startswith("control") & R.t_tender.notna() & ~R.status.str.startswith(("sealed", "insufficient", "excluded"))].copy()
    D["dur"] = np.where(D.t_complete.notna(), D.t_complete - D.t_tender, D.t_last - D.t_tender)
    D["event"] = D.t_complete.notna()
    D = D[D.dur > 0]
    xs, ys = km_curve(D.dur.to_numpy(), D.event.to_numpy())
    med = xs[np.argmax(1 - ys >= 0.5)] if (1 - ys >= 0.5).any() else float("nan")
    km = dict(n=int(len(D)), median_years=float(med))
    C = R[R.t_media_done.notna() & R.t_complete.notna() & ~R.early_sealed.astype(bool)]
    mv = dict(completion_lag_years_median=float((C.t_complete - C.t_media_done).median()))
    L = R[R.t_launch.notna() & ~R.kind.str.startswith("control")]
    pre = float((L.t_launch - L.t_tender).median())
    return [
        dict(hook="Uganda publishes every public contract. None says when the work starts or ends.",
             stat="0 of 142,168", sub="contracts in the open data carry a start or finish date", draw=g_dots, kw={}),
        dict(hook="So we watched 22 contracted roads from space",
             stat="2017 to 2026", sub="Sentinel-2 imagery, every kilometre, twice a year", draw=g_map, kw=dict(geo=geo, routes=routes)),
        dict(hook="Hoima–Wanseko: red gravel turns into grey tarmac",
             stat="Sealed 2020–21", sub="after earthworks in 2019; commissioned January 2022", draw=g_strip, kw=dict(G=G, rid="hoima")),
        dict(hook="Koboko–Yumbe–Moyo: not one kilometre sealed yet",
             stat="0 of 103 km", sub="works began about three years after the tender; due July 2027", draw=g_strip, kw=dict(G=G, rid="koboko")),
        dict(hook="Most of the wait comes before the machines arrive",
             stat=f"{pre:.1f} years", sub="median from tender to works start, where a launch date is reported", draw=g_decomp, kw=dict(R=R)),
        dict(hook="From tender to tarmac",
             stat=f"{km['median_years']:.1f} years", sub=f"median time from tender to a finished road ({km['n']} roads)", draw=g_km, kw=dict(R=R)),
        dict(hook="Checked against the news before any claim",
             stat=f"{mv['completion_lag_years_median']:.1f} years", sub="typical gap between satellite and reported completion (39 reports, 17 roads)",
             draw=g_media, kw=dict(R=R)),
    ]


def frame(layout, b, k, n, t):
    L = LAYOUTS[layout]
    W, H, m = L["W"], L["H"], L["m"]
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    fk = font(L["kick"][1], True)
    d.text((m, L["kick"][0]), f"{k:02d} / {n:02d}  ·  {STORY}  ·  {SERIES}", font=fk, fill=MUT)
    fh = font(L["head"][1], True)
    y = L["head"][0]
    for line in wrap(d, b["hook"], fh, W - 2 * m):
        d.text((m, y), line, font=fh, fill=INK)
        y += int(fh.size * 1.12)
    s0, s1 = L["stage"]
    s0 = max(s0, y + 30)
    b["draw"](d, (m, s0 + 30, W - m, s1), t, **b["kw"])
    fs = font(L["stat"][1], True)
    d.text((m, L["stat"][0]), b["stat"], font=fs, fill=BLUE)
    fsub = font(L["sub"][1])
    yy = L["sub"][0]
    for line in wrap(d, b["sub"], fsub, W - 2 * m):
        d.text((m, yy), line, font=fsub, fill=INK2)
        yy += int(fsub.size * 1.2)
    fy = L["foot"][0]
    d.line([m, fy - 14, W - m, fy - 14], fill=(214, 214, 210), width=2)
    ff = font(L["foot"][1], True)
    d.text((m, fy), LINK, font=ff, fill=BLUE)
    fn = font(L["foot"][1])
    d.text((W - m - d.textlength("Gava Charles", font=fn), fy), "Gava Charles", font=fn, fill=MUT)
    return im


def encode(frames_iter, W, H, path, fps=FPS):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.Popen([ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)],
                         stdin=subprocess.PIPE)
    for im in frames_iter:
        p.stdin.write(im.tobytes())
    p.stdin.close()
    p.wait()


def main():
    R, G, geo, routes = load()
    B = beats(R, G, geo, routes)
    n = len(B)
    nf, na = int(SEC * FPS), int(ANIM * FPS)
    for d_ in ("clips_square", "carousel"):
        (OUT / d_).mkdir(parents=True, exist_ok=True)
    ts = [min(1.0, i / na) for i in range(nf)]
    for layout, name in [("square", "reel_square.mp4"), ("vertical", "reel_vertical.mp4"), ("film", "film_tarmac.mp4")]:
        W, H = LAYOUTS[layout]["W"], LAYOUTS[layout]["H"]
        encode((frame(layout, b, k + 1, n, t) for k, b in enumerate(B) for t in ts), W, H, OUT / name)
        print("wrote", name, flush=True)
    pages = []
    for k, b in enumerate(B):
        encode((frame("square", b, k + 1, n, t) for t in ts), 1080, 1080, OUT / "clips_square" / f"{k + 1:02d}.mp4")
        pg = frame("square", b, k + 1, n, 1.0)
        pg.save(OUT / "carousel" / f"C{k + 1:02d}.png")
        pages.append(pg)
    pages[0].save(OUT / "carousel" / "tarmac_carousel.pdf", save_all=True, append_images=pages[1:])
    gif = [frame("square", b, k + 1, n, ts[i]).resize((540, 540), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128)
           for k, b in enumerate(B) for i in range(0, nf, 3)]
    gif[0].save(OUT / "highlight.gif", save_all=True, append_images=gif[1:], duration=100, loop=0, optimize=True)
    print("wrote", OUT)


if __name__ == "__main__":
    if "--still" in sys.argv:
        R, G, geo, routes = load()
        B = beats(R, G, geo, routes)
        OUT.mkdir(parents=True, exist_ok=True)
        for layout in LAYOUTS:
            for k, b in enumerate(B):
                frame(layout, b, k + 1, len(B), 1.0).save(OUT / f"still_{layout}_{k + 1}.png")
    else:
        main()
