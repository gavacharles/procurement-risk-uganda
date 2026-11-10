"""Record an animated tutorial: how to navigate the interactive map.

Drives the real site (served locally from docs/) in installed Chrome via Playwright,
with an on-screen cursor and step captions; records video and converts to MP4 + GIF.

Out: posts/media/map_tutorial.mp4, map_tutorial.gif ; docs/media/map_tutorial.mp4
"""
from __future__ import annotations

import http.server
import shutil
import subprocess
import tempfile
import threading
from functools import partial
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "posts" / "media"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1280, 900

OVERLAY = """
(() => {
  const c = document.createElement('div');
  c.id = '__cursor';
  c.style.cssText = 'position:fixed;z-index:99999;width:22px;height:22px;border-radius:50%;background:rgba(235,104,52,.35);border:2px solid #eb6834;pointer-events:none;transform:translate(-50%,-50%);transition:width .15s,height .15s;left:-50px;top:-50px';
  document.body.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; });
  document.addEventListener('mousedown', () => { c.style.width = '34px'; c.style.height = '34px'; });
  document.addEventListener('mouseup', () => { c.style.width = '22px'; c.style.height = '22px'; });
  const b = document.createElement('div');
  b.id = '__cap';
  b.style.cssText = 'position:fixed;z-index:99998;left:50%;top:58px;transform:translateX(-50%);max-width:880px;background:#14171b;color:#fff;font:600 21px/1.35 Inter,system-ui,sans-serif;padding:14px 22px;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.35);text-align:center;opacity:0;transition:opacity .3s';
  document.body.appendChild(b);
  window.__caption = (t, step) => { b.innerHTML = (step ? '<span style="color:#f08a5d">' + step + '</span>&nbsp; ' : '') + t; b.style.opacity = t ? 1 : 0; };
})();
"""


def serve():
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    httpd = serve()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    with tempfile.TemporaryDirectory() as td, sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                  record_video_dir=td, record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle")
        page.wait_for_function("typeof MD !== 'undefined' && MD && layer")
        page.evaluate(OVERLAY)
        page.evaluate("window.scrollTo(0, document.getElementById('map-section').offsetTop - 48)")
        page.wait_for_timeout(600)
        cap = lambda t, s=None: page.evaluate("([t,s]) => window.__caption(t,s)", [t, s])  # noqa: E731

        def move_to(sel, dx=0, dy=0, steps=25):
            bb = page.locator(sel).first.bounding_box()
            page.mouse.move(bb["x"] + bb["width"] / 2 + dx, bb["y"] + bb["height"] / 2 + dy, steps=steps)

        def click(sel):
            move_to(sel)
            page.wait_for_timeout(250)
            page.locator(sel).first.click()

        def district_xy(name):
            return page.evaluate("""(n) => { let pt=null; layer.eachLayer(l => { if (l.feature.properties.district===n) {
                 const c = map.latLngToContainerPoint(l.getBounds().getCenter()); const r = document.getElementById('map').getBoundingClientRect();
                 pt = [r.left + c.x, r.top + c.y]; } }); return pt; }""", name)

        page.mouse.move(W / 2, H / 2)
        cap("How to navigate the interactive map", "Tutorial")
        page.wait_for_timeout(2600)

        cap("Choose a question: where the money goes, competition, or price &amp; process", "1")
        click("#qtabs button[data-q=competition]")
        page.wait_for_timeout(2400)

        cap("Pick a measure: here, the share of works value won by each district's top supplier", "2")
        click("#metrics button:nth-child(2)")
        page.wait_for_timeout(2600)

        cap("Filter by financial year…", "3")
        move_to("#year")
        page.select_option("#year", index=3)
        page.evaluate("document.getElementById('year').dispatchEvent(new Event('change'))")
        page.wait_for_timeout(2200)

        cap("…and by sector, e.g. roads &amp; bridges", "4")
        move_to("#sub")
        page.select_option("#sub", "roads")
        page.evaluate("document.getElementById('sub').dispatchEvent(new Event('change'))")
        page.wait_for_timeout(2200)
        page.select_option("#year", "all")
        page.evaluate("document.getElementById('year').dispatchEvent(new Event('change'))")
        page.select_option("#sub", "all")
        page.evaluate("document.getElementById('sub').dispatchEvent(new Event('change'))")

        cap("Hover over a district for a quick reading", "5")
        for n in ["Lira", "Gulu"]:
            x, y = district_xy(n)
            page.mouse.move(x, y, steps=30)
            page.wait_for_timeout(1100)

        cap("Click it: the panel compares it with the national median, shows the trend and the public bodies buying works there", "6")
        x, y = district_xy("Gulu")
        page.mouse.click(x, y)
        page.wait_for_timeout(1200)
        move_to("#panel", dy=-60, steps=30)
        page.wait_for_timeout(3600)

        cap("Dashed districts have fewer than 10 contracts; you can hide them", "7")
        click("#hideSmall")
        page.wait_for_timeout(2400)
        click("#hideSmall")

        cap("Try question 3: how long until awards become public, and how often prices land just below the estimate", "8")
        click("#qtabs button[data-q=process]")
        page.wait_for_timeout(1200)
        click("#metrics button:nth-child(2)")
        page.wait_for_timeout(2600)

        cap("Explore it yourself: gavacharles.github.io/procurement-risk-uganda", "")
        page.mouse.move(W - 60, H - 120, steps=20)
        page.wait_for_timeout(3200)

        video = page.video.path()
        ctx.close()
        browser.close()
        mp4 = Path(td) / "map_tutorial.mp4"
        gif = Path(td) / "map_tutorial.gif"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", video, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "22", "-movflags", "+faststart", str(mp4)], check=True)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        "fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",
                        str(gif)], check=True)
        for f in (mp4, gif):
            dst = OUT / f.name
            dst.unlink(missing_ok=True)
            shutil.copyfile(f, dst)
        (DOCS / "media").mkdir(exist_ok=True)
        shutil.copyfile(mp4, DOCS / "media" / "map_tutorial.mp4")
    httpd.shutdown()
    print("wrote map_tutorial.mp4 / .gif")


if __name__ == "__main__":
    main()
