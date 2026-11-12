"""Build shareable versions of each paper (published separately): Word (.docx) via pandoc and PDF via HTML (MathML) +
headless Chrome. Output goes to paper/build/ (git-ignored with paper/).

Usage: python src/build_papers.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "build"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:760px;margin:40px auto;font-size:11.5pt;line-height:1.5;color:#111}
h1.title{font-size:20pt;line-height:1.25;margin-bottom:4px}
p.author,p.date{margin:2px 0;color:#444}
.abstract{font-size:10.5pt;background:#f6f5f2;padding:12px 16px;border-left:3px solid #2a78d6;margin:16px 0}
h1{font-size:15pt;margin-top:28px}h2{font-size:12.5pt;margin-top:20px}
table{border-collapse:collapse;font-size:9.5pt;margin:10px 0;width:100%}
td,th{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}
th{border-bottom:2px solid #999}
img{max-width:100%;display:block;margin:8px auto}
figure{margin:14px 0;page-break-inside:avoid}figcaption{font-size:9.5pt;color:#444;text-align:center}
blockquote{font-size:10pt;color:#333;border-left:3px solid #eb6834;margin:12px 0;padding:4px 12px;background:#fdf6f2}
code{font-size:9pt}
@page{size:A4;margin:18mm}
"""


def build(md_name: str):
    src = PAPER / md_name
    stem = src.stem
    OUT.mkdir(parents=True, exist_ok=True)
    # Word
    docx = OUT / f"{stem}.docx"
    pypandoc.convert_file(str(src), "docx", outputfile=str(docx),
                          extra_args=[f"--resource-path={PAPER}", "--toc", "--toc-depth=1"])
    # HTML (MathML) -> PDF
    with tempfile.TemporaryDirectory() as td:
        css = Path(td) / "paper.css"
        css.write_text(CSS)
        html = PAPER / f".{stem}.print.html"  # next to figures so relative paths resolve
        pypandoc.convert_file(str(src), "html5", outputfile=str(html),
                              extra_args=["--standalone", "--mathml", f"--css={css}", "--embed-resources",
                                          f"--resource-path={PAPER}", f"--metadata=pagetitle:{stem}"])
        pdf = Path(td) / f"{stem}.pdf"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={td}/prof", f"--print-to-pdf={pdf}", html.as_uri()],
                       check=True, capture_output=True)
        dst = OUT / f"{stem}.pdf"
        dst.unlink(missing_ok=True)
        shutil.copyfile(pdf, dst)
        html.unlink(missing_ok=True)
    print("built", docx.name, dst.name)


if __name__ == "__main__":
    for name in ["paper1_bunching.md", "paper2_risk_based_oversight.md",
                 "paper3_tender_to_tarmac.md", "paper4_estimate_disclosure.md"]:
        build(name)
