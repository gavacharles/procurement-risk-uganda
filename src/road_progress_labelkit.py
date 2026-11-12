"""Blind hand-labelling kit to validate the road-surface classifier (paper 3, validation (vi)).

Draws a stratified random sample of segment x period readings (half predicted asphalt, half unpaved),
cuts a 600 m x 600 m Sentinel-2 true-colour crop centred on each segment for that period, and writes:

  paper/labelkit/sheet_XX.png       numbered crops, 4 x 4 per sheet, crosshair on the road point
  paper/labelkit/labels.csv         one row per crop: fill coder1 / coder2 with A (asphalt),
                                    U (unpaved), X (cannot tell)
  paper/labelkit/key.csv            the classifier's prediction (do NOT open before labelling)

Then `python3 src/road_progress_labelkit.py --score` reports accuracy and Cohen's kappa.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import road_progress as rp  # noqa: E402
import satellite_crops as cr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "labelkit"
N = 160
HALF = 300  # metres


def crop(lon, lat, period):
    tag, start, end, _ = next(p for p in rp.PERIODS if p[0] == period)
    bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
    old = cr.HALF_M
    cr.HALF_M = HALF
    try:
        for sc_ in rp.scenes_window(bbox, start, end)[:6]:
            href = cr.tci_href(sc_)
            if not href:
                continue
            try:
                scl = cr.read_window(sc_["scl"], lon, lat)
                if scl is None or np.isin(scl[0], cr.CLOUDY).mean() > 0.05:
                    continue
                img = cr.read_window(href, lon, lat)
            except Exception:
                continue
            if img is not None and img.shape[0] == 3 and (img.sum(axis=0) == 0).mean() < 0.01:
                return np.moveaxis(img, 0, -1).astype(float), sc_["id"]
    finally:
        cr.HALF_M = old
    return None, None


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    G = pd.read_parquet(rp.P / "road_progress_segments.parquet")
    G = G[G.used.astype(bool)]
    order = [c for c in G.columns if len(c) == 5 and c[:4].isdigit()]
    L = G.melt(id_vars=["id", "seg", "km_at"], value_vars=order, var_name="period", value_name="pred")
    L = L[L.pred.isin(["asphalt", "unpaved"])]
    rng = np.random.default_rng(42)
    samp = (L.groupby("pred", group_keys=False)
             .apply(lambda d: d.sample(n=min(len(d), N // 2 + 30), random_state=int(rng.integers(1e9)))))
    geo = {}
    rows = []
    for r in samp.sample(frac=1, random_state=7).itertuples():
        if len(rows) >= N:
            break
        if r.id not in geo:
            road = next(x for x in rp.ROADS if x[0] == r.id)
            g = rp.geometry(road)
            geo[r.id] = rp.segments(g["coords"], road[3]).set_index("seg")
        s = geo[r.id].loc[int(r.seg)]
        img, sid = crop(s.lon, s.lat, r.period)
        if img is None:
            continue
        rows.append(dict(n=len(rows) + 1, id=r.id, seg=int(r.seg), km=r.km_at, period=r.period, scene=sid,
                         lon=s.lon, lat=s.lat, pred=r.pred, img=cr.stretch([img])[0]))
        print(len(rows), r.id, r.period, flush=True)
    for k in range(0, len(rows), 16):
        fig, axs = plt.subplots(4, 4, figsize=(10, 10.6))
        for ax in axs.ravel():
            ax.axis("off")
        for ax, r in zip(axs.ravel(), rows[k:k + 16]):
            ax.imshow(r["img"], interpolation="lanczos")
            h, w = r["img"].shape[:2]
            ax.plot([w / 2 - 6, w / 2 - 2], [h / 2, h / 2], color="yellow", lw=1)
            ax.plot([w / 2 + 2, w / 2 + 6], [h / 2, h / 2], color="yellow", lw=1)
            ax.plot([w / 2, w / 2], [h / 2 - 6, h / 2 - 2], color="yellow", lw=1)
            ax.plot([w / 2, w / 2], [h / 2 + 2, h / 2 + 6], color="yellow", lw=1)
            ax.set_title(f"#{r['n']}", fontsize=10)
        fig.suptitle("Is the road at the crosshair sealed (A), unpaved (U), or can't tell (X)?  600 m x 600 m, Sentinel-2 true colour",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"sheet_{k // 16 + 1:02d}.png", dpi=130)
        plt.close(fig)
    base = pd.DataFrame([{k: v for k, v in r.items() if k != "img"} for r in rows])
    base[["n", "id", "km", "period", "lon", "lat", "scene"]].assign(coder1="", coder2="").to_csv(OUT / "labels.csv", index=False)
    base[["n", "pred"]].to_csv(OUT / "key.csv", index=False)
    print("wrote", len(rows), "crops to", OUT)


def score():
    from sklearn.metrics import cohen_kappa_score
    lab = pd.read_csv(OUT / "labels.csv")
    key = pd.read_csv(OUT / "key.csv")
    d = lab.merge(key, on="n")
    out = {}
    for c in ("coder1", "coder2"):
        x = d[d[c].isin(["A", "U"])]
        if len(x):
            truth = np.where(x[c] == "A", "asphalt", "unpaved")
            out[c] = dict(n=int(len(x)), accuracy=float((truth == x.pred).mean()),
                          cannot_tell=int((d[c] == "X").sum()))
    both = d[d.coder1.isin(["A", "U"]) & d.coder2.isin(["A", "U"])]
    if len(both):
        out["inter_coder_kappa"] = float(cohen_kappa_score(both.coder1, both.coder2))
        agree = both[both.coder1 == both.coder2]
        truth = np.where(agree.coder1 == "A", "asphalt", "unpaved")
        out["accuracy_where_coders_agree"] = dict(n=int(len(agree)), accuracy=float((truth == agree.pred).mean()))
    print(out)
    import json
    (ROOT / "paper" / "road_progress_labels_score.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    score() if "--score" in sys.argv else build()
